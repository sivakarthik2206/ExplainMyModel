import io
import base64
import traceback
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

app = FastAPI()


# ---------------------------------------------------
# HEALTH CHECK (Render uses this for service uptime)
# ---------------------------------------------------
@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "ExplainMyModel backend running. Use POST /explain"
    }


# ---------------------------------------------------
# HELPERS
# ---------------------------------------------------

def safe_read_csv(upload: UploadFile) -> pd.DataFrame:
    try:
        data = upload.file.read()
        upload.file.seek(0)
        return pd.read_csv(io.BytesIO(data))
    except Exception:
        raise HTTPException(400, "Invalid CSV file.")


def safe_load_model(upload: UploadFile):
    try:
        data = upload.file.read()
        upload.file.seek(0)
        model = joblib.load(io.BytesIO(data))
    except Exception:
        raise HTTPException(400, "Model file could not be loaded. Use joblib or pickle format.")

    if not hasattr(model, "predict"):
        raise HTTPException(400, "Uploaded file is not a valid ML model (missing predict()).")

    return model


def normalize_shap_values(values):
    """
    SHAP returns different structures:
    - TreeClassifier: list of arrays (one per class)
    - TreeRegressor: array
    - KernelExplainer: array
    - LinearExplainer: array

    We convert everything into a clean 2D numpy array.
    """
    if isinstance(values, list):
        # Classifier case → take mean magnitude across classes
        arrs = [np.abs(v) for v in values]
        merged = np.mean(arrs, axis=0)
        return merged

    values = np.array(values)

    # If SHAP outputs (samples, features, classes) → reduce classes
    if values.ndim == 3:
        values = np.mean(values, axis=2)

    return values


def pick_explainer(model, X):
    """
    Smart explainer selection with fallback.
    """
    model_name = model.__class__.__name__.lower()

    try:
        if any(k in model_name for k in ["forest", "tree", "xgb", "lgb", "cat"]):
            return shap.TreeExplainer(model)
        if hasattr(model, "coef_"):  # Linear models
            return shap.LinearExplainer(model, X)
    except Exception:
        pass

    # Kernel fallback (slow → use sampling)
    background = shap.sample(X, min(40, len(X)))
    return shap.KernelExplainer(model.predict, background)


def generate_summary_plot(shap_values, X_df):
    plt.switch_backend("Agg")

    fig = plt.figure(figsize=(8, 4))
    shap.summary_plot(shap_values, X_df, show=False)
    buf = io.BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)

    return base64.b64encode(buf.read()).decode()


def build_natural_language_explanations(shap_values, X_df, top_k=5):
    abs_vals = np.mean(np.abs(shap_values), axis=0)
    top_idx = abs_vals.argsort()[::-1][:top_k]

    explanations = []
    suggestions = []

    for idx in top_idx:
        feature = X_df.columns[idx]
        score = float(abs_vals[idx])

        explanations.append({
            "feature": feature,
            "importance": score,
            "explanation": f"'{feature}' strongly influences the model. Higher variation in this feature leads to noticeable prediction changes."
        })

        suggestions.append(
            f"Consider improving feature quality or adding more samples for '{feature}' to improve the model’s stability."
        )

    return explanations, suggestions


# ---------------------------------------------------
# MAIN EXPLAIN ENDPOINT
# ---------------------------------------------------
@app.post("/explain")
async def explain(model_file: UploadFile = File(...), csv_file: UploadFile = File(...)):
    try:
        X = safe_read_csv(csv_file)
        model = safe_load_model(model_file)

        # Auto-sample to avoid large SHAP computations
        if len(X) > 300:
            X_small = X.sample(300, random_state=42)
        else:
            X_small = X

        explainer = pick_explainer(model, X_small)

        # Compute SHAP values
        shap_values_raw = explainer.shap_values(X_small)
        shap_values = normalize_shap_values(shap_values_raw)

        # Ensure shapes match
        if shap_values.shape[1] != X_small.shape[1]:
            raise RuntimeError("SHAP mismatch: feature count does not match CSV columns.")

        # Generate plot
        plot_b64 = generate_summary_plot(shap_values, X_small)

        # Natural language
        feature_exp, suggestions = build_natural_language_explanations(shap_values, X_small)

        return JSONResponse({
            "explainer": explainer.__class__.__name__,
            "summary_plot_b64": plot_b64,
            "feature_explanations": feature_exp,
            "suggestions": suggestions
        })

    except HTTPException as e:
        raise e

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "detail": f"Unexpected error: {e}",
                "trace": traceback.format_exc()
            }
        )
