import io
import os
import base64
import json
import time
import requests
import streamlit as st
import pandas as pd

# ============================================================
# CONFIG
# ============================================================
API_URL = "https://explainmymodel.onrender.com/explain"
REQUEST_TIMEOUT = 180

st.set_page_config(
    page_title="ExplainMyModel",
    page_icon="✨",
    layout="centered",
)


# ============================================================
# APPLE-STYLE POLISHED THEME
# ============================================================
def load_css():
    st.markdown(
        """
        <style>

        /* Global UI */
        body, input, textarea {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        }

        /* Centered Title */
        .title-container {
            text-align: center;
            padding-top: 15px;
        }

        /* Frosted-glass card (Apple aesthetic) */
        .apple-card {
            background: rgba(255, 255, 255, 0.55);
            backdrop-filter: saturate(180%) blur(20px);
            -webkit-backdrop-filter: saturate(180%) blur(20px);
            border-radius: 18px;
            padding: 30px;
            box-shadow: 0 8px 25px rgba(0,0,0,0.08);
            margin-top: 20px;
        }

        @media (prefers-color-scheme: dark) {
            .apple-card {
                background: rgba(35, 35, 35, 0.45) !important;
                box-shadow: 0 8px 30px rgba(0,0,0,0.55);
                color: #f1f1f1;
            }
        }

        /* Upload Button */
        .stButton>button {
            background: linear-gradient(135deg, #007aff, #0051a8);
            color: white;
            padding: 0.7rem 1.5rem;
            border-radius: 12px;
            border: none;
            font-size: 1rem;
            font-weight: 500;
        }

        .stButton>button:hover {
            background: linear-gradient(135deg, #409cff, #007aff);
        }

        /* Feature explanation cards */
        .feature-card {
            background: rgba(255,255,255,0.45);
            border-radius: 14px;
            padding: 15px 22px;
            margin-bottom: 14px;
        }

        @media (prefers-color-scheme: dark) {
            .feature-card {
                background: rgba(60,60,60,0.35);
            }
        }

        /* Center images */
        .center-image {
            display: flex;
            justify-content: center;
            margin-top: 15px;
        }

        </style>
        """,
        unsafe_allow_html=True
    )


load_css()


# ============================================================
# HELPERS
# ============================================================
def send_to_backend(model_bytes, model_name, csv_bytes, csv_name):
    files = {
        "model_file": (model_name, io.BytesIO(model_bytes), "application/octet-stream"),
        "csv_file": (csv_name, io.BytesIO(csv_bytes), "text/csv"),
    }
    response = requests.post(API_URL, files=files, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()


def decode_image(b64_str):
    return base64.b64decode(b64_str) if b64_str else None


# ============================================================
# LANDING PAGE
# ============================================================
st.markdown(
    """
    <div class="title-container">
        <h1>✨ ExplainMyModel</h1>
        <p style="font-size:1.15rem;color:#666;">
            Transform any ML model into clear, human-friendly insights.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# UPLOAD SECTION
# ============================================================
with st.container():
    st.markdown("<div class='apple-card'>", unsafe_allow_html=True)

    with st.form("upload_form", clear_on_submit=False):
        col1, col2 = st.columns(2)

        with col1:
            model_file = st.file_uploader(
                "Upload Model (.pkl / .joblib)",
                type=["pkl", "joblib"]
            )
        with col2:
            csv_file = st.file_uploader(
                "Upload Sample CSV",
                type=["csv"]
            )

        submit = st.form_submit_button("Explain")

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# EXECUTION
# ============================================================
if submit:
    if not model_file or not csv_file:
        st.error("Please upload both the model and the CSV file.")
        st.stop()

    with st.spinner("✨ Running SHAP explainability pipeline…"):
        try:
            start = time.time()
            result = send_to_backend(
                model_file.getvalue(), model_file.name,
                csv_file.getvalue(), csv_file.name
            )
            runtime = round(time.time() - start, 2)
        except Exception as e:
            st.error(f"Request failed: {e}")
            st.stop()

    # ============================================================
    # RESULTS
    # ============================================================
    st.markdown(f"<h3>Completed in {runtime}s</h3>", unsafe_allow_html=True)

    # SHAP Image
    img_b64 = result.get("summary_plot_b64")
    if img_b64:
        st.markdown("<div class='center-image'>", unsafe_allow_html=True)
        st.image(decode_image(img_b64), caption="SHAP Summary Plot", use_column_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Feature-level explanations
    st.markdown("<h2>Top Feature Explanations</h2>", unsafe_allow_html=True)

    for item in result.get("feature_explanations", []):
        st.markdown(
            f"""
            <div class='feature-card'>
                <strong>{item['feature']}</strong><br>
                Importance: {item['importance']:.4f}<br>
                {item['explanation']}
            </div>
            """,
            unsafe_allow_html=True
        )

    # Suggestions
    st.markdown("<h2>Suggestions — What to Improve</h2>", unsafe_allow_html=True)

    for s in result.get("suggestions", []):
        st.markdown(f"<div class='feature-card'>{s}</div>", unsafe_allow_html=True)

    # Download Output JSON
    st.download_button(
        "Download JSON Output",
        json.dumps(result, indent=2).encode("utf-8"),
        file_name="explanation.json",
        mime="application/json"
    )
