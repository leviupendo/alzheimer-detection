"""
app.py — Streamlit web UI for Alzheimer's Detection from MRI scans.

Run:
    streamlit run app/app.py
"""

import sys
from pathlib import Path
import numpy as np
import streamlit as st
from PIL import Image
import io
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from preprocess import load_image, extract_features
from utils import IDX_TO_CLASS, CLASSES, MODELS_DIR
import joblib
import cv2

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Alzheimer Detection",
    page_icon="🧠",
    layout="centered",
)

# ── Styling ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { max-width: 800px; margin: 0 auto; }
    .stProgress > div > div > div { background-color: #3b82f6; }
    .result-box {
        background: #f0f9ff;
        border-left: 4px solid #3b82f6;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        margin-top: 1rem;
    }
    .warning-box {
        background: #fff7ed;
        border-left: 4px solid #f97316;
        padding: 1rem 1.5rem;
        border-radius: 8px;
    }
    .class-label { font-size: 1.6rem; font-weight: 700; color: #1e40af; }
    .confidence  { font-size: 1.1rem; color: #374151; }
</style>
""", unsafe_allow_html=True)

# ── Load model ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    candidates = sorted(MODELS_DIR.glob("best_randomforest.pkl"))
    if not candidates:
        candidates = sorted(MODELS_DIR.glob("best_*.pkl"))
    if not candidates:
        return None
    return joblib.load(candidates[-1])


# ── UI ─────────────────────────────────────────────────────────────────────
st.title("🧠 Alzheimer's Detection")
st.caption("MRI Brain Scan Classifier — Research & Educational Use Only")

st.markdown("""
Upload a brain MRI scan and the model will classify the level of dementia:
**NonDemented · VeryMildDemented · MildDemented · ModerateDemented**
""")

# Sidebar info
with st.sidebar:
    st.header("About")
    st.markdown("""
**Model:** Random Forest on handcrafted features

**Features extracted:**
- HOG (Histogram of Oriented Gradients)
- LBP (Local Binary Patterns)
- Intensity statistics

**Dataset:** Alzheimer's 4-class MRI dataset (Kaggle)

---
⚠️ *This tool is for research purposes only and is not a clinical diagnostic tool.*
    """)
    st.header("Model Status")
    model = load_model()
    if model:
        st.success("✅ Model loaded")
    else:
        st.error("❌ No trained model found.\nRun `python src/train.py` first.")

# File uploader
uploaded = st.file_uploader(
    "Upload MRI scan (.jpg / .png)",
    type=["jpg", "jpeg", "png"],
    help="Use grayscale T1-weighted MRI images for best results.",
)

if uploaded:
    col1, col2 = st.columns([1, 1])

    # Show original
    pil_img = Image.open(uploaded).convert("L")
    with col1:
        st.subheader("Uploaded Scan")
        st.image(pil_img, use_column_width=True, caption="Input MRI")

    # Preprocess & show resized
    img_bytes = uploaded.read()
    nparr = np.frombuffer(img_bytes, np.uint8)

    # Re-read with cv2
    uploaded.seek(0)
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img_cv = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img_cv, (128, 128))

    with col2:
        st.subheader("Preprocessed (128×128)")
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.imshow(img_resized, cmap="bone")
        ax.axis("off")
        st.pyplot(fig, use_container_width=True)

    # Predict
    if model is None:
        st.error("No model loaded. Please train the model first.")
    else:
        with st.spinner("Extracting features & classifying…"):
            feats = extract_features(img_resized).reshape(1, -1)
            pred_idx = model.predict(feats)[0]
            predicted = IDX_TO_CLASS[pred_idx]

            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(feats)[0]
                confidence = float(proba[pred_idx])
                all_probs = {CLASSES[i]: float(p) for i, p in enumerate(proba)}
            else:
                confidence = 1.0
                all_probs = {predicted: 1.0}

        # Severity colour mapping
        severity_color = {
            "NonDemented":       "#22c55e",
            "VeryMildDemented":  "#eab308",
            "MildDemented":      "#f97316",
            "ModerateDemented":  "#ef4444",
        }
        color = severity_color.get(predicted, "#3b82f6")

        st.markdown(f"""
<div class="result-box">
  <div class="class-label" style="color:{color}">🔍 {predicted}</div>
  <div class="confidence">Confidence: <strong>{confidence*100:.1f}%</strong></div>
</div>
""", unsafe_allow_html=True)

        # Probability bar chart
        st.subheader("Class Probabilities")
        fig2, ax2 = plt.subplots(figsize=(7, 3))
        colors = [severity_color[c] for c in CLASSES]
        vals   = [all_probs.get(c, 0) for c in CLASSES]
        bars   = ax2.barh(CLASSES, vals, color=colors, edgecolor="white", linewidth=0.8)
        ax2.set_xlim(0, 1)
        ax2.set_xlabel("Probability")
        for bar, val in zip(bars, vals):
            ax2.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                     f"{val*100:.1f}%", va="center", fontsize=9)
        ax2.spines[["top", "right"]].set_visible(False)
        ax2.invert_yaxis()
        plt.tight_layout()
        st.pyplot(fig2, use_container_width=True)

        st.markdown("""
<div class="warning-box">
⚠️ <strong>Disclaimer:</strong> This prediction is generated by a research model and should 
<strong>not</strong> be used for clinical diagnosis. Please consult a qualified medical professional.
</div>
""", unsafe_allow_html=True)

else:
    st.info("👆 Upload an MRI brain scan image to get started.")

    # Sample class descriptions
    st.subheader("Classification Guide")
    cols = st.columns(4)
    info = [
        ("NonDemented",      "🟢", "No detectable signs of cognitive decline"),
        ("VeryMild",         "🟡", "Very subtle early-stage changes"),
        ("Mild",             "🟠", "Noticeable memory and cognitive issues"),
        ("Moderate",         "🔴", "Significant impairment requiring care"),
    ]
    for col, (label, icon, desc) in zip(cols, info):
        col.metric(f"{icon} {label}", "", desc)