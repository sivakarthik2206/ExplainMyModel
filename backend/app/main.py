# backend/app/main.py
import io
import os
import base64
import hashlib
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Tuple, List

import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

app = FastAPI(title="ExplainMyModel")
@app.get("/")
async def root():
    return {"status": "ok", "message": "ExplainMyModel backend. POST /explain"}
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow calls from anywhere for demo (restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Allow the frontend (Streamlit) to call this API during demos or hosted deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["*"],
)

MAX_UPLOAD_BYTES = 30 * 1024 * 1024  # 30 MB
SHAP_SAMPLE_ROWS = 200                # sample size used for SHAP plotting/background
KERNEL_BACKGROUND = 50                # background size for KernelExplainer
TOP_K_FEATURES = 6

_executor = ThreadPoolExecutor(max_workers=1)
_cache: Dict[str, Dict[str, Any]] = {}  # very small in-memory cache for repeat uploads


def _hash_bytes(*parts: bytes) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p)
    return h.hexdigest()


def _read_bytes_limited(upfile: UploadFile) -> bytes:
    content = upfile.file.read()
    size = len(content)
    if size > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="Uploaded file exceeds size limit.")
    upfile.file.seek(0)
    return content


def _load_model_from_bytes(b: bytes):
    bio = io.BytesIO(b)
    try:
        model = joblib.load(bio)
    except Exception as e:
        # try pickle fallback
        bio.seek(0)
        raise HTTPException(status_code=400, detail=f"Failed to load model: {e}")
    if not hasattr(model, "predict"):
        raise HTTPException(status_code=400, detail="Loaded object is not a model with a predict method.")
    return model


def _read_csv_from_bytes(b: bytes) -> pd.DataFrame:
    bio = io.BytesIO(b)
    try:
        df = pd.read_csv(bio)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV: {e}")
    if df.empty:
        raise HTTPException(status_code=400, detail="CSV appears to be empty.")
    return df


def _choose_explainer(model, X: pd.DataFrame):
    mname = type(model).__name__.lower()
    if any(k in mname for k in ("xgboost", "lgbm", "catboost", "randomforest", "decisiontree", "forest", "gradientboost")):
        return "tree"
    if hasattr(model, "coef_") or hasattr(model, "intercept_"):
        return "linear"
    return "kernel"


def _prepare_sample(X: pd.DataFrame) -> pd.DataFrame:
    if len(X) > SHAP_SAMPLE_ROWS:
        return X.sample(n=SHAP_SAMPLE_ROWS, random_state=42).reset_index(drop=True)
    return X.reset_index(drop=True)


def _shap_values_for_model(model, X: pd.DataFrame) -> Tuple[Any, str]:
    # This function runs SHAP synchronously; it's executed in a thread to avoid blocking.
    expl_type = _choose_explainer(model, X)
    X_sample = _prepare_sample(X)
    try:
        if expl_type == "tree":
            expl = shap.TreeExplainer(model)
            shap_values = expl.shap_values(X_sample)
            return shap_values, expl.__class__.__name__
        if expl_type == "linear":
            expl = shap.LinearExplainer(model, X_sample, feature_dependence="independent")
            shap_values = expl.shap_values(X_sample)
            return shap_values, expl.__class__.__name__
        # fallback
        bg = shap.sample(X_sample, min(KERNEL_BACKGROUND, len(X_sample)))
        expl = shap.KernelExplainer(model.predict, bg)
        shap_values = expl.shap_values(X_sample)
        return shap_values, expl.__class__.__name__
    except Exception as e:
        raise RuntimeError(f"SHAP computation failed: {e}\n{traceback.format_exc()}")


def _avg_abs_shap(shap_values) -> np.ndarray:
    # shap_values may be ndarray (num_samples x num_features) or list (multi-class)
    if isinstance(shap_values, list):
        # average across class outputs by magnitude
        abs_means = [np.mean(np.abs(sv), axis=0) for sv in shap_values]
        return np.mean(np.vstack(abs_means), axis=0)
    if isinstance(shap_values, np.ndarray):
        return np.mean(np.abs(shap_values), axis=0)
    # unknown shape -> try numpy conversion
    try:
        arr = np.asarray(shap_values)
        return np.mean(np.abs(arr), axis=0)
    except Exception:
        raise RuntimeError("Unexpected SHAP values format.")


def _generate_summary_plot(shap_values, X: pd.DataFrame) -> str:
    plt.tight_layout()
    fig = plt.figure(figsize=(7, 4.5))
    try:
        # shap handles arrays or list-of-arrays
        shap.summary_plot(shap_values, X, show=False)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode("utf-8")
        return encoded
    finally:
        plt.close(fig)


def _nl_feature_explanations(avg_abs: np.ndarray, columns: List[str], top_k: int = TOP_K_FEATURES):
    idx = np.argsort(avg_abs)[::-1][:top_k]
    explanations = []
    suggestions = []
    for i in idx:
        feat = columns[i]
        score = float(avg_abs[i])
        explanations.append({
            "feature": feat,
            "importance": round(score, 6),
            "explanation": f"'{feat}' strongly influences the model — changes in this feature have a notable effect on predictions."
        })
        suggestions.append(f"Consider engineering '{feat}' (scaling, binning, or adding interaction terms) or collecting more samples for robust estimates.")
    return explanations, suggestions


async def _compute_explanation_async(model_bytes: bytes, csv_bytes: bytes) -> Dict[str, Any]:
    cache_key = _hash_bytes(model_bytes, csv_bytes)
    if cache_key in _cache:
        return _cache[cache_key]

    model = _load_model_from_bytes(model_bytes)
    X = _read_csv_from_bytes(csv_bytes)
    # For heavy CPU-bound SHAP work, run in threadpool
    shap_values, explainer_name = await run_in_threadpool(_shap_values_for_model, model, X)

    avg_abs = _avg_abs_shap(shap_values)
    explanations, suggestions = _nl_feature_explanations(avg_abs, list(X.columns))

    # Use a small sample for plotting to keep image stable/fast
    plot_X = _prepare_sample(X)
    plot_b64 = await run_in_threadpool(_generate_summary_plot, shap_values, plot_X)

    result = {
        "explainer": explainer_name,
        "summary_plot_b64": plot_b64,
        "feature_explanations": explanations,
        "suggestions": suggestions,
    }

    # keep small cache (last N)
    if len(_cache) > 16:
        _cache.pop(next(iter(_cache)))
    _cache[cache_key] = result
    return result


@app.post("/explain")
async def explain_endpoint(model_file: UploadFile = File(...), csv_file: UploadFile = File(...)):
    try:
        model_bytes = _read_bytes_limited(model_file)
        csv_bytes = _read_bytes_limited(csv_file)
        result = await _compute_explanation_async(model_bytes, csv_bytes)
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except RuntimeError as re:
        raise HTTPException(status_code=500, detail=str(re))
    except Exception as e:
        # keep the error informative for demo/troubleshooting
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")
