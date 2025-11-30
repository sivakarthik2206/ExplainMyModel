# backend/app/main.py
import io
import os
import base64
import tempfile
import traceback
from typing import Tuple, Any

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

app = FastAPI()

MAX_UPLOAD_BYTES = 40 * 1024 * 1024  # 40 MB
MAX_SHAP_ROWS = 200  # cap rows used for SHAP explainers/summaries
BACKGROUND_SAMPLE = 50  # for KernelExplainer background sampling


def safe_read_bytes(upload: UploadFile) -> bytes:
    upload.file.seek(0)
    content = upload.file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="Upload too large.")
    upload.file.seek(0)
    return content


def load_model_from_bytes(b: bytes) -> Any:
    # joblib.load accepts file-like; try joblib then pickle as fallback
    buf = io.BytesIO(b)
    try:
        model = joblib.load(buf)
        return model
    except Exception:
        buf.seek(0)
        import pickle
        try:
            model = pickle.load(buf)
            return model
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to load model: {e}")


def read_csv_from_bytes(b: bytes) -> pd.DataFrame:
    try:
        df = pd.read_csv(io.BytesIO(b))
        if df.shape[0] == 0:
            raise ValueError("CSV has no rows.")
        return df
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV: {e}")


def choose_explainer_and_compute_shap(model, X: pd.DataFrame) -> Tuple[Any, Any]:
    # Work on a sampled X for SHAP speed and memory
    X_sample = X if len(X) <= MAX_SHAP_ROWS else X.sample(n=MAX_SHAP_ROWS, random_state=0)
    try:
        name = type(model).__name__.lower()
        if any(k in name for k in ("xgboost", "lgbm", "catboost", "randomforest", "decisiontree", "forest", "gbm")):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
            return explainer, shap_values
        if hasattr(model, "coef_") or any(k in name for k in ("linear", "logistic")):
            explainer = shap.LinearExplainer(model, X_sample, feature_perturbation="independent")
            shap_values = explainer.shap_values(X_sample)
            return explainer, shap_values
        # Generic fallback: KernelExplainer using small background
        background = X_sample.sample(n=min(BACKGROUND_SAMPLE, len(X_sample)), random_state=0)
        explainer = shap.KernelExplainer(model.predict, background)
        shap_values = explainer.shap_values(X_sample)
        return explainer, shap_values
    except Exception as e:
        raise RuntimeError(f"SHAP explain failed: {e}\n{traceback.format_exc()}")


def normalize_shap_values(shap_values, X_cols):
    """
    Normalize shap_values into a 2D numpy array (rows x features) that we can take mean abs over.
    Supports:
     - ndarray (rows x features)
     - list of arrays (classes x rows x features) -> average across classes
    """
    if isinstance(shap_values, list):
        try:
            # list of arrays (for multi-class): average absolute across classes and rows
            arrs = [np.array(sv) for sv in shap_values]
            # ensure same shape
            stacked = np.stack(arrs, axis=0)  # classes x rows x features
            mean_abs = np.mean(np.abs(stacked), axis=(0, 1))  # per-feature
            return mean_abs
        except Exception:
            # attempt shape fallback
            concat = np.mean([np.abs(np.array(sv)).mean(axis=0) for sv in shap_values], axis=0)
            return concat
    else:
        arr = np.array(shap_values)
        if arr.ndim == 1:
            # single prediction vector -> treat absolute values directly
            return np.abs(arr)
        if arr.ndim == 2:
            return np.mean(np.abs(arr), axis=0)
        # unexpected shape
        raise RuntimeError("Unexpected SHAP values shape.")


def plot_shap_summary(shap_values, X_sample: pd.DataFrame) -> str:
    plt.switch_backend("Agg")
    fig = plt.figure(figsize=(6, 4))
    try:
        # shap.summary_plot accepts different shap_values shapes
        shap.summary_plot(shap_values, X_sample, show=False)
        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)
        return img_b64
    except Exception:
        plt.close(fig)
        raise


def build_nl_explanations(mean_abs_importances, feature_names, top_k=6):
    idx = np.argsort(mean_abs_importances)[::-1][:top_k]
    explanations = []
    for i in idx:
        feat = feature_names[i]
        importance = float(mean_abs_importances[i])
        # Human-friendly sentence
        explanations.append({
            "feature": feat,
            "importance": importance,
            "explanation": f"'{feat}' has a strong influence on predictions — shifts in its value significantly change model output."
        })
    # suggestions heuristics
    suggestions = []
    top_features = [feature_names[i] for i in idx[:3]]
    suggestions.append(f"Inspect distributions and outliers for top features: {', '.join(top_features)}.")
    suggestions.append("Try simple feature transforms (scaling, log, binning) and re-train to reduce variance.")
    suggestions.append("If performance is unstable, gather more labeled samples for underrepresented regions.")
    suggestions.append("Consider regularization or simpler model if overfitting is suspected.")
    return explanations, suggestions


@app.get("/")
async def root():
    return {"status": "ok", "message": "ExplainMyModel backend. POST /explain"}


@app.post("/explain")
async def explain(model_file: UploadFile = File(...), csv_file: UploadFile = File(...)):
    # Basic defensive checks
    try:
        model_bytes = safe_read_bytes(model_file)
        csv_bytes = safe_read_bytes(csv_file)

        df = read_csv_from_bytes(csv_bytes)
        model = load_model_from_bytes(model_bytes)

        # Ensure model has a prediction method
        if not hasattr(model, "predict"):
            raise HTTPException(status_code=400, detail="Model object has no predict() method.")

        # Compute SHAP
        explainer, shap_vals = choose_explainer_and_compute_shap(model, df)

        # Normalize and compute top features
        mean_abs = normalize_shap_values(shap_vals, df.columns)
        explanations, suggestions = build_nl_explanations(mean_abs, list(df.columns))

        # Create summary plot (use sampled X for plotting)
        X_plot = df if len(df) <= MAX_SHAP_ROWS else df.sample(n=MAX_SHAP_ROWS, random_state=0)
        img_b64 = plot_shap_summary(shap_vals, X_plot)

        result = {
            "explainer": explainer.__class__.__name__ if explainer is not None else "unknown",
            "summary_plot_b64": img_b64,
            "feature_explanations": explanations,
            "suggestions": suggestions,
        }
        return JSONResponse(content=result)
    except HTTPException as he:
        raise he
    except RuntimeError as re:
        # SHAP-specific runtime issues
        return JSONResponse(status_code=500, content={"detail": f"Unexpected error: {str(re)}"})
    except Exception as e:
        # Generic fallback - helpful for debugging during demo; strip long traces in final prod
        tb = traceback.format_exc()
        return JSONResponse(status_code=500, content={"detail": f"Unexpected error: {str(e)}", "trace": tb})
