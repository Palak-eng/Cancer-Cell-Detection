import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array


st.set_page_config(
    page_title="Cancer Cell Detection",
    page_icon="🎗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

import base64

MODEL_PATH = os.path.join("models", "best_model.keras")

# Place your own logo/icon here (PNG or JPG) and it will be used automatically
# in the sidebar and page headers instead of the emoji fallback.
LOGO_PATH = os.path.join("assets", "logo.png")


def get_logo_html(height=48):
    """Returns an <img> tag for the logo if assets/logo.png exists, else ''."""
    if os.path.exists(LOGO_PATH):
        with open(LOGO_PATH, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        ext = os.path.splitext(LOGO_PATH)[1].lstrip(".").lower() or "png"
        if ext == "jpg":
            ext = "jpeg"
        return (
            f'<img src="data:image/{ext};base64,{b64}" '
            f'style="height:{height}px; width:auto; display:block; border-radius:8px;" />'
        )
    return ""

# ---------------------------------------------------------------------------
# Global styling — forced pastel lavender theme (ignores OS dark mode)
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"], .stApp, [data-testid="stAppViewContainer"] {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(180deg, #faf5fd 0%, #f2e4fa 100%) !important;
        color: #3b1f4d !important;
    }
    [data-testid="stHeader"] {
        background: rgba(0,0,0,0) !important;
    }
    .block-container {
        padding-top: 1.6rem;
        max-width: 1100px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #3b1f4d !important;
    }
    section[data-testid="stSidebar"] * {
        color: #f3e8fb !important;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] label {
        background-color: rgba(255,255,255,0.07);
        border-radius: 8px;
        padding: 0.5rem 0.8rem;
        margin-bottom: 0.4rem;
    }

    /* Headings */
    h1, h2, h3, h4, h5 {
        font-family: 'Poppins', sans-serif;
        color: #3b1f4d !important;
        letter-spacing: -0.3px;
    }
    p, span, label, li, div {
        color: #3b1f4d;
    }

    /* Hero */
    .hero {
        background: linear-gradient(135deg, #f2e4fb 0%, #e6cdf5 100%);
        border-radius: 20px;
        padding: 2rem 2.4rem;
        margin-bottom: 1.4rem;
        border: 1px solid #e2c9f2;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .hero-eyebrow {
        display: inline-block;
        background-color: #8b2f7a;
        color: white !important;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 1px;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        margin-bottom: 0.7rem;
        text-transform: uppercase;
    }
    .hero-title {
        font-family: 'Poppins', sans-serif;
        font-size: 2.1rem;
        font-weight: 800;
        color: #2d1b3d !important;
        margin: 0;
        line-height: 1.15;
    }
    .hero-subtitle {
        color: #6b4a7a !important;
        font-size: 0.98rem;
        margin-top: 0.5rem;
        max-width: 480px;
    }

    /* Card containers (real Streamlit containers) */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #ffffff;
        border-radius: 18px;
        border: 1px solid #efe2f7 !important;
        box-shadow: 0 4px 18px rgba(90, 40, 110, 0.07);
    }
    div[data-testid="stVerticalBlockBorderWrapper"] > div {
        padding: 0.4rem 0.3rem;
    }
    /* Higher-specificity override so the sidebar stays purple, not white */
    section[data-testid="stSidebar"] div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }

    /* Result badges */
    .result-badge {
        display: inline-block;
        padding: 0.55rem 1.5rem;
        border-radius: 30px;
        font-size: 1.3rem;
        font-family: 'Poppins', sans-serif;
        font-weight: 700;
        margin-bottom: 0.7rem;
    }
    .badge-benign {
        background-color: #e5f6ec;
        color: #1e8449 !important;
        border: 1px solid #9fe0ba;
    }
    .badge-malignant {
        background-color: #fbe6f2;
        color: #a3216f !important;
        border: 1px solid #f0a8d3;
    }

    .confidence-number {
        font-family: 'Poppins', sans-serif;
        font-size: 2.4rem;
        font-weight: 800;
        color: #3b1f4d !important;
        line-height: 1;
    }

    .pill {
        display: inline-block;
        background-color: #f2e6fa;
        color: #6b2d5c !important;
        padding: 0.32rem 0.95rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 0.5rem;
        margin-top: 0.3rem;
    }

    .disclaimer {
        background-color: #fdf3e0;
        border: 1px solid #f0d18a;
        border-radius: 12px;
        padding: 0.9rem 1.1rem;
        font-size: 0.85rem;
        color: #7a5b00 !important;
        margin-top: 1rem;
    }
    .disclaimer * { color: #7a5b00 !important; }

    .stat-tile {
        background-color: #faf5fd;
        border: 1px solid #ecdcf7;
        border-radius: 14px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.7rem;
    }
    .stat-tile .label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        color: #9b6bab !important;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .stat-tile .value {
        font-family: 'Poppins', sans-serif;
        font-size: 1.02rem;
        font-weight: 600;
        color: #3b1f4d !important;
    }

    div.stButton > button {
        background: linear-gradient(135deg, #8b2f7a 0%, #6b2d5c 100%);
        color: white !important;
        border-radius: 10px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        font-family: 'Poppins', sans-serif;
        border: none;
        box-shadow: 0 3px 10px rgba(139, 47, 122, 0.22);
        width: 100%;
    }
    div.stButton > button:hover {
        background: linear-gradient(135deg, #a3378f 0%, #7c3569 100%);
        color: white !important;
    }

    section[data-testid="stFileUploaderDropzone"] {
        background-color: #faf5fd !important;
        border: 1.5px dashed #d3aee6 !important;
        border-radius: 14px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()


def hero(eyebrow, title, subtitle):
    logo = get_logo_html(56) or '<span style="font-size:2.6rem;">🎗️</span>'
    html = (
        '<div class="hero">'
        '<div>'
        f'<span class="hero-eyebrow">{eyebrow}</span>'
        f'<p class="hero-title">{title}</p>'
        f'<p class="hero-subtitle">{subtitle}</p>'
        '</div>'
        f'<div>{logo}</div>'
        '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
_sidebar_logo = get_logo_html(32) or '<span style="font-size:1.5rem;">🎗️</span>'
_sidebar_html = (
    '<div style="display:flex; align-items:center; gap:0.6rem; margin-bottom:0.2rem;">'
    f'{_sidebar_logo}'
    '<span style="font-family:\'Poppins\',sans-serif; font-size:1.15rem; font-weight:700;">Cell Detection</span>'
    '</div>'
    '<div style="opacity:0.75; font-size:0.85rem; margin-bottom:0.8rem;">AI-assisted histopathology screening</div>'
)
st.sidebar.markdown(_sidebar_html, unsafe_allow_html=True)

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["Prediction", "Model Information", "About Project"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<div style='font-size:0.78rem; opacity:0.65;'>Built with EfficientNetB0 + TensorFlow</div>",
    unsafe_allow_html=True
)


# ---------------------------------------------------------------------------
# Prediction page
# ---------------------------------------------------------------------------
if page == "Prediction":

    hero(
        "Spot Early",
        "Cancer Cell Detection",
        "Upload a histopathology image and get an instant AI-assisted "
        "classification — Benign or Malignant — powered by EfficientNetB0."
    )

    uploaded_file = st.file_uploader(
        "Upload a histopathology image (JPG / PNG)",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:

        image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns([1, 1], gap="large")

        with col1:
            with st.container(border=True):
                st.markdown("#### 🖼️ Uploaded Image")
                st.image(image, use_container_width=True)

            st.write("")
            predict_clicked = st.button("🔬 Run Prediction", use_container_width=True)

        with col2:
            with st.container(border=True):
                st.markdown("#### 📊 Prediction Result")

                if predict_clicked:

                    img = image.resize((224, 224))
                    img = img_to_array(img)
                    # EfficientNetB0 normalizes internally (expects raw 0-255
                    # pixel values) — do not divide by 255 here.
                    img = np.expand_dims(img, axis=0)

                    prediction = model.predict(img, verbose=0)[0][0]

                    if prediction >= 0.5:
                        label = "Malignant"
                        confidence = float(prediction)
                        badge_class = "badge-malignant"
                    else:
                        label = "Benign"
                        confidence = float(1 - prediction)
                        badge_class = "badge-benign"

                    st.markdown(
                        f'<span class="result-badge {badge_class}">{label}</span>',
                        unsafe_allow_html=True
                    )

                    st.markdown("**Confidence**")
                    st.markdown(
                        f'<div class="confidence-number">{confidence*100:.2f}%</div>',
                        unsafe_allow_html=True
                    )
                    st.progress(confidence)

                    st.markdown(
                        '<span class="pill">Model: EfficientNetB0</span>'
                        '<span class="pill">Input: 224 × 224 RGB</span>',
                        unsafe_allow_html=True
                    )

                    st.markdown(
                        '<div class="disclaimer">⚠️ This tool is a student/research '
                        'project prototype and is <b>not a medical diagnostic device</b>. '
                        'Results should never be used to make real clinical decisions. '
                        'Always consult a qualified pathologist or physician.</div>',
                        unsafe_allow_html=True
                    )

                else:
                    st.info("Upload an image and click **Run Prediction** to see results here.")


# ---------------------------------------------------------------------------
# Model information page
# ---------------------------------------------------------------------------
elif page == "Model Information":

    hero(
        "Under the Hood",
        "Model Information",
        "Technical details of the classification model."
    )

    col1, col2 = st.columns(2, gap="large")

    with col1:
        with st.container(border=True):
            st.markdown(
                '<div class="stat-tile"><div class="label">Architecture</div>'
                '<div class="value">EfficientNetB0 (ImageNet, transfer learning)</div></div>'
                '<div class="stat-tile"><div class="label">Task</div>'
                '<div class="value">Binary classification — Benign vs. Malignant</div></div>'
                '<div class="stat-tile"><div class="label">Classes</div>'
                '<div class="value">0 → Benign &nbsp;|&nbsp; 1 → Malignant</div></div>',
                unsafe_allow_html=True
            )

    with col2:
        with st.container(border=True):
            st.markdown(
                '<div class="stat-tile"><div class="label">Input Size</div>'
                '<div class="value">224 × 224 RGB</div></div>'
                '<div class="stat-tile"><div class="label">Loss Function</div>'
                '<div class="value">Binary Crossentropy</div></div>'
                '<div class="stat-tile"><div class="label">Optimizer</div>'
                '<div class="value">Adam</div></div>',
                unsafe_allow_html=True
            )


# ---------------------------------------------------------------------------
# About page
# ---------------------------------------------------------------------------
else:

    hero(
        "About",
        "About This Project",
        "A quick overview of what this tool does — and doesn't — do."
    )

    with st.container(border=True):
        st.markdown(
            """
This project uses **EfficientNetB0** and TensorFlow to classify histopathology
images into two categories:

- 🟢 **Benign**
- 🔴 **Malignant**

Upload an image on the **Prediction** page to receive a classification and
confidence score.

This is a learning / portfolio project and is **not intended for clinical use**.
            """
        )