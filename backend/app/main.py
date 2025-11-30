import io
import traceback
import base64
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import joblib

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse


app = FastAPI(
    title="ExplainMyModel Backend",
    description="Upload a model + CSV → SHAP explanations + insights",
    version="1.0.0",
)

MAX_UPLOAD_MB = 30 * 1024 * 1024  # 30MB


# -----------------------------------------------------------
# UTILITIES
# -----------------------------------------------------------

def read_csv(upload: UploadFile) -> pd.DataFrame:
    content = upload.file.read()
    if len(content) > MAX_UPLOAD_MB:
        raise HTTPException(400, "CSV too large")
    try:
        upload.file.seek(0)
        df = pd.read_csv(upload.file)
        if df.empty:
            raise ValueError("CSV is empty")
        return df
    except Exception as e:
        raise HTTPException(400, f"Failed to read CSV: {e}")


def load_model(upload: UploadFile):
    content = upload.file.read()
    if len(content) > MAX_UPLOAD_MB:
        raise HTTPException(400, "Model file too large")

    try:
        upload.file.seek(0)
        model = joblib.load(upload.file)
        if not hasattr(model, "predict"):
            raise ValueError("File is not a valid ML model")
        return model
    except Exception as e:
        raise HTTPException(400, f"Failed to load model: {e}")


def safe_sample(df: pd.DataFrame, max_rows: int = 200):
    """Avoid SHAP failures by sampling oversized datasets."""
    if len(df) > max_rows:
        return df.sample(max_rows, random_state=42)
    return df


def choose_explainer(model, X_sample):
    """Pick the best SHAP explainer for the model type."""
    name = type(model).__name__.lower()

    # Tree-based
    if any(k in name for k in ["forest", "tree", "xgb", "xgboost", "lgbm", "catboost"]):
        return shap.TreeExplainer(model)

    # Linear models
    if hasattr(model, "coef_"):
        return shap.LinearExplainer(model, X_sample)

    # Generic fallback
    background = shap.sample(X_sample, min(50, len(X_sample)))
    return shap.KernelExplainer(model.predict, background)


def compute_shap_values(model, X):
    """Compute robust SHAP values without shape/index errors."""
    X_sample = safe_sample(X)
    explainer = choose_explainer(model, X_sample)

    try:
        shap_vals = explainer.shap_values(X_sample)

        # SHAP sometimes returns list for multi-class → flatten to 2D
        if isinstance(shap_vals, list):
            shap_vals = np.array(shap_vals[0])

        shap_vals = np.array(shap_vals)

        # Force 2D shape consistency
        if shap_vals.ndim == 1:
            shap_vals = shap_vals.reshape(-1, 1)

        if shap_vals.shape[1] != X_sample.shape[1]:
            raise ValueError("SHAP value feature dimension mismatch.")

        return shap_vals, X_sample
    except Exception as e:
        raise RuntimeError(f"SHAP computation failed: {e}\n{traceback.format_exc()}")


def create_summary_plot(shap_vals, X_sample):
    """Generate base64 SHAP summary plot."""
    plt.switch_backend("Agg")
    fig = plt.figure(figsize=(7, 4))
    shap.summary_plot(shap_vals, X_sample, show=False)

    buf = io.BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format="png")
    buf.seek(0)
    plt.close(fig)

    return base64.b64encode(buf.read()).decode()


def top_feature_explanations(shap_vals, X_sample, k=5):
    avg_importance = np.mean(np.abs(shap_vals), axis=0)
    idx = np.argsort(avg_importance)[::-1][:k]

    explanations = []
    suggestions = []

    for i in idx:
        feat = X_sample.columns[i]
        score = float(avg_importance[i])

        explanations.append({
            "feature": feat,
            "importance": score,
            "explanation": f"'{feat}' has a strong influence on predictions. Larger changes in this feature tend to shift the model’s output noticeably."
        })

        suggestions.append(
            f"Consider improving data quality or engineering '{feat}' further — it's one of the top drivers affecting prediction stability."
        )

    return explanations, suggestions


# -----------------------------------------------------------
# ROOT HEALTH CHECK (Render expects something on "/")
# -----------------------------------------------------------
@app.get("/")
def health():
    return {"status": "ok", "message": "ExplainMyModel backend running. POST to /explain"}


# -----------------------------------------------------------
# EXPLAIN ENDPOINT
# -----------------------------------------------------------

@app.post("/explain")
async def explain(model_file: UploadFile = File(...), csv_file: UploadFile = File(...)):
    try:
        X = read_csv(csv_file)
        model = load_model(model_file)

        shap_vals, X_sample = compute_shap_values(model, X)

        summary_plot_b64 = create_summary_plot(shap_vals, X_sample)
        explanations, suggestions = top_feature_explanations(shap_vals, X_sample)

        return JSONResponse({
            "summary_plot_b64": summary_plot_b64,
            "feature_explanations": explanations,
            "suggestions": suggestions
        })

    except HTTPException as he:
        raise he

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"detail": f"Unexpected error: {e}"}
        )
