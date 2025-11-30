# backend/app/main.py
import io
import os
import time
import base64
import traceback
import logging
from typing import Any, Tuple

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# CONFIG
MAX_BYTES = 50 * 1024 * 1024     # 50 MB per upload
MAX_ROWS_SHAP = 200              # sample for SHAP to limit compute
SAMPLE_BACKGROUND = 50
ALLOWED_MODEL_EXT = (".pkl", ".joblib")
ALLOWED_CSV_EXT = (".csv",)

# LOGGING
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("explainmymodel")

app = FastAPI(title="ExplainMyModel API")

# Allow public frontend origins; adjust if you want tighter control
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

def _read_bytes_and_check(upload: UploadFile) -> bytes:
    content = upload.file.read()
    size = len(content)
    if size == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if size > MAX_BYTES:
        raise HTTPException(status_code=400, detail=f"File too large ({size} bytes). Max {MAX_BYTES} bytes.")
    upload.file.seek(0)
    return content

def _load_csv(csv_bytes: bytes) -> pd.DataFrame:
    try:
        return pd.read_csv(io.BytesIO(csv_bytes))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {e}")

def _load_model(model_bytes: bytes):
    try:
        # joblib supports file-like objects
        return joblib.load(io.BytesIO(model_bytes))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load model: {e}")

def _choose_explainer(model: Any, X: pd.DataFrame):
    name = type(model).__name__.lower()
    try:
        if any(k in name for k in ("xgboost", "lgbm", "catboost", "randomforest", "decisiontree", "forest", "boost")):
            return shap.TreeExplainer(model)
        if hasattr(model, "coef_") or hasattr(model, "intercept_"):
            return shap.LinearExplainer(model, X, feature_dependence="independent")
    except Exception:
        logger.exception("Fast explainer selection failed; will fallback to KernelExplainer.")

    # Fallback to KernelExplainer using a small background sample
    background = X.sample(n=min(SAMPLE_BACKGROUND, max(1, len(X))), random_state=42)
    return shap.KernelExplainer(model.predict, background)

def _compute_shap_values(explainer, X_sample: pd.DataFrame):
    # Some explainers return list (multi-class), some return array
    shap_values = explainer.shap_values(X_sample)
    return shap_values

def _summary_image_to_b64(shap_values, X_sample: pd.DataFrame) -> str:
    fig = plt.figure(figsize=(6, 4))
    try:
        # shap.summary_plot handles list/array internally
        shap.summary_plot(shap_values, X_sample, show=False)
        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150)
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)
        return img_b64
    except Exception:
        plt.close(fig)
        raise

def _mean_abs_importance(shap_values, X_sample: pd.DataFrame) -> np.ndarray:
    # Normalize various returned shapes from SHAP
    if isinstance(shap_values, list):
        # multiclass: list of arrays [n_samples, n_features] per class
        arrs = [np.array(sv) for sv in shap_values]
        # take mean absolute across samples and classes
        stacked = np.stack([np.mean(np.abs(a), axis=0) for a in arrs], axis=0)  # (classes, features)
        mean_abs = np.mean(stacked, axis=0)
    else:
        arr = np.array(shap_values)
        # arr can be (n_samples, n_features) or (n_outputs, n_samples, n_features) etc.
        if arr.ndim == 3:
            # e.g., (outputs, samples, features) — average appropriately
            mean_abs = np.mean(np.abs(arr), axis=(0,1))
        elif arr.ndim == 2:
            mean_abs = np.mean(np.abs(arr), axis=0)
        else:
            # unexpected shape
            mean_abs = np.mean(np.abs(arr).reshape(arr.shape[0], -1), axis=0)
    return mean_abs

def _generate_nl_explanations(mean_abs_importance: np.ndarray, feature_names, top_k=6):
    idx = np.argsort(mean_abs_importance)[::-1][:top_k]
    expl_list = []
    suggestions = []
    for i in idx:
        feat = feature_names[i]
        importance = float(mean_abs_importance[i])
        expl_list.append({
            "feature": str(feat),
            "importance": importance,
            "explanation": f"'{feat}' is a top driver of predictions — changes in its value noticeably shift model output."
        })
        suggestions.append(f"Consider engineering or collecting more robust measurements for '{feat}' (binning, scaling, or more samples).")
    return expl_list, suggestions

@app.get("/")
def root_health():
    return {"status": "ok", "time": time.time()}

@app.post("/explain")
async def explain(model_file: UploadFile = File(...), csv_file: UploadFile = File(...)):
    try:
        # Basic filename checks (not strict)
        if not any(str(model_file.filename).lower().endswith(ext) for ext in ALLOWED_MODEL_EXT):
            # allow other extensions but warn
            logger.info("Model filename doesn't match expected extensions; attempting to load anyway.")

        # Read and validate uploads
        model_bytes = _read_bytes_and_check(model_file)
        csv_bytes = _read_bytes_and_check(csv_file)

        # Parse inputs
        X = _load_csv(csv_bytes)
        if X.shape[0] == 0 or X.shape[1] == 0:
            raise HTTPException(status_code=400, detail="CSV has no data or columns.")

        model = _load_model(model_bytes)

        # For SHAP we sample rows for speed and stability
        X_sample = X if len(X) <= MAX_ROWS_SHAP else X.sample(n=MAX_ROWS_SHAP, random_state=42)
        
        # Build explainer and compute SHAP values
        explainer = _choose_explainer(model, X_sample)
        shap_values = _compute_shap_values(explainer, X_sample)

        # Compute mean abs importance robustly
        mean_abs = _mean_abs_importance(shap_values, X_sample)
        if len(mean_abs) != X_sample.shape[1]:
            # last-resort reshape attempt: align by columns
            mean_abs = np.resize(mean_abs, X_sample.shape[1])

        # Generate human explanations & suggestions
        feature_names = list(X_sample.columns)
        feature_explanations, suggestions = _generate_nl_explanations(mean_abs, feature_names)

        # Generate summary image (try/except but continue if image fails)
        summary_plot_b64 = None
        try:
            summary_plot_b64 = _summary_image_to_b64(shap_values, X_sample)
        except Exception as e:
            logger.exception("Failed to create SHAP summary plot; continuing without image.")

        # Return a clean JSON
        resp = {
            "explainer": explainer.__class__.__name__ if hasattr(explainer, "__class__") else str(explainer),
            "n_rows": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "summary_plot_b64": summary_plot_b64,
            "feature_explanations": feature_explanations,
            "suggestions": suggestions
        }
        return JSONResponse(content=resp)

    except HTTPException as he:
        # expected client errors
        logger.info(f"Client error: {he.detail}")
        raise he
    except Exception as exc:
        # catch-all: log internal details but return sanitized message to client
        logger.exception("Unexpected error in /explain")
        tb = traceback.format_exc()
        return JSONResponse(status_code=500, content={"detail": f"Unexpected error: {str(exc)}"})

