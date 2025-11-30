# frontend/streamlit_app.py
import io
import os
import base64
import json
import time
from typing import Optional

import pandas as pd
import requests
import streamlit as st

# Configuration
API_URL = "http://localhost:8000/explain"
REQUEST_TIMEOUT = 120

st.set_page_config(page_title="ExplainMyModel", layout="centered")
st.title("ExplainMyModel")
st.write("Upload a trained model (.pkl/.joblib) and a matching CSV sample → get SHAP explanations and human-friendly suggestions.")

# Helpers
def post_files(model_bytes: bytes, model_name: str, csv_bytes: bytes, csv_name: str):
    files = {
        "model_file": (model_name, io.BytesIO(model_bytes), "application/octet-stream"),
        "csv_file": (csv_name, io.BytesIO(csv_bytes), "text/csv"),
    }
    resp = requests.post(API_URL, files=files, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp.json()

def render_report_html(title: str, explainer: str, img_b64: Optional[str], explanations, suggestions):
    html = [f"<h1>{title}</h1>", f"<p><strong>Explainer:</strong> {explainer}</p>"]
    if img_b64:
        html.append(f'<img src="data:image/png;base64,{img_b64}" style="max-width:100%;height:auto;"/>')
    html.append("<h2>Top Feature Explanations</h2><ul>")
    for e in explanations:
        html.append(f"<li><strong>{e['feature']}</strong> (importance={e['importance']:.4f}): {e['explanation']}</li>")
    html.append("</ul><h2>Suggestions — What to improve</h2><ul>")
    for s in suggestions:
        html.append(f"<li>{s}</li>")
    html.append("</ul>")
    return "\n".join(html)

# UI
with st.form("upload_form", clear_on_submit=False):
    left, right = st.columns([1, 1])
    with left:
        model_file = st.file_uploader("Model (.pkl / .joblib)", type=["pkl", "joblib"], help="Pickle or joblib-serialized sklearn/xgboost models.")
        st.markdown("**Tip:** Use `joblib.dump(model, 'model.joblib')` to export.")
    with right:
        csv_file = st.file_uploader("Sample CSV", type=["csv"], help="A small sample (10-200 rows) that matches model features.")
        if csv_file:
            try:
                df_preview = pd.read_csv(csv_file)
                st.write("CSV preview (first 5 rows):")
                st.dataframe(df_preview.head())
            except Exception:
                st.warning("Couldn't preview CSV; ensure it's valid.")
    submitted = st.form_submit_button("Explain (run SHAP)")

if submitted:
    if not model_file or not csv_file:
        st.error("Both a model and a CSV are required.")
    else:
        st.info("Submitting files. Processing may take a few seconds.")
        try:
            model_bytes = model_file.getvalue()
            csv_bytes = csv_file.getvalue()
            spinner_text = st.empty()
            with st.spinner("Running explainability pipeline..."):
                start = time.time()
                resp = post_files(model_bytes, model_file.name, csv_bytes, csv_file.name)
                elapsed = time.time() - start
            spinner_text.success(f"Completed in {elapsed:.1f}s")
            explainer = resp.get("explainer", "unknown")
            img_b64 = resp.get("summary_plot_b64")
            explanations = resp.get("feature_explanations", [])
            suggestions = resp.get("suggestions", [])

            st.success(f"Explainer used: {explainer}")
            if img_b64:
                st.image(base64.b64decode(img_b64), caption="SHAP summary", use_column_width=True)

            st.markdown("### Top feature explanations")
            for e in explanations:
                st.markdown(f"**{e['feature']}** (importance={e['importance']:.4f}) — {e['explanation']}")

            st.markdown("### Suggestions — What to improve")
            for s in suggestions:
                st.write("• " + s)

            # Downloadable assets
            report_html = render_report_html("ExplainMyModel Report", explainer, img_b64, explanations, suggestions)
            report_bytes = report_html.encode("utf-8")
            st.download_button("Download HTML report", data=report_bytes, file_name="explain_report.html", mime="text/html")

            json_bytes = json.dumps(resp, indent=2).encode("utf-8")
            st.download_button("Download JSON output", data=json_bytes, file_name="explain_output.json", mime="application/json")

        except requests.exceptions.HTTPError as e:
            st.error(f"Server error: {e.response.text if e.response is not None else e}")
        except requests.exceptions.RequestException as e:
            st.error(f"Request failed: {e}")
        except Exception as e:
            st.error(f"Unexpected error: {e}")
