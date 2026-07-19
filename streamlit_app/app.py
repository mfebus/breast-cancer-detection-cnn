"""
Breast Cancer Detection — Streamlit Prototype
Loads the trained EfficientNetB0 classifier and runs real predictions on
uploaded histopathology image patches.

Run locally with:
    streamlit run app.py

Requires the trained model files to be present at:
    model/breast_cancer_efficientnet_final.keras
    model/threshold.json
(copy these out of your Drive's breast_cancer_project/models/ folder)
"""

import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# --------------------------------------------------------------------------
# Page config -- must be the first Streamlit call
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Diagnostic Triage Dash — Breast Cancer CNN",
    page_icon="🔬",
    layout="wide",
)

MODEL_PATH = Path("model/breast_cancer_efficientnet_final.keras")
THRESHOLD_PATH = Path("model/threshold.json")
IMG_SIZE = 224  # must match the size the model was trained on

# --------------------------------------------------------------------------
# Custom styling -- matches the companion dashboard artifact's visual
# language (navy header, cream background, brick accent, monospace labels)
# so the static mockup and this live app read as one cohesive product.
# --------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&display=swap');

    .triage-header {
        background: #10173a;
        color: #eee8da;
        padding: 28px 32px 22px;
        border-radius: 8px;
        margin-bottom: 24px;
    }
    .triage-header p {
        color: #d8d3c2;
        font-size: 15px;
        line-height: 1.55;
        margin: 0 0 14px;
    }
    .triage-byline {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 11px;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #8b85a0;
    }
    .stat-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 11px;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #7a7466;
        display: block;
        margin-bottom: 4px;
    }
    .stat-value {
        font-size: 28px;
        font-weight: 800;
        color: #10173a;
        border-left: 2px solid #a83e28;
        padding-left: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Model loading (cached so it only loads once per session, not per upload)
# --------------------------------------------------------------------------
@st.cache_resource
def load_model_and_threshold():
    """Load the trained Keras model and its calibrated decision threshold.

    Returns:
        (model, threshold, target_recall) or (None, None, None) if the
        model files aren't present -- the app falls back to a clear
        on-screen error instead of crashing.
    """
    if not MODEL_PATH.exists():
        return None, None, None

    model = tf.keras.models.load_model(str(MODEL_PATH))

    threshold = 0.5
    target_recall = None
    if THRESHOLD_PATH.exists():
        with open(THRESHOLD_PATH) as f:
            meta = json.load(f)
        threshold = meta.get("threshold", 0.5)
        target_recall = meta.get("target_recall")

    return model, threshold, target_recall


def preprocess_image(pil_image: Image.Image) -> np.ndarray:
    """Convert an uploaded PIL image into the (1, 224, 224, 3) array the model expects.

    Mirrors the notebook's load_image() function: resize to IMG_SIZE and
    leave pixel values in raw [0, 255] range -- the model's internal
    EfficientNet preprocessing layer handles scaling, so we must NOT
    normalize to [0,1] here or predictions will be wrong.
    """
    img = pil_image.convert("RGB")                       # ensure 3 channels, no alpha
    img = img.resize((IMG_SIZE, IMG_SIZE))                 # match training resolution
    arr = np.array(img, dtype=np.float32)                  # raw [0, 255] range, as the model expects
    arr = np.expand_dims(arr, axis=0)                       # add batch dimension -> (1, 224, 224, 3)
    return arr


# --------------------------------------------------------------------------
# Sidebar -- model card / context, so this reads as a real clinical-support
# tool rather than a bare demo
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("About this model")
    st.markdown(
        """
        **Architecture:** EfficientNetB0 (ImageNet-pretrained, fine-tuned)

        **Training data:** IDC breast histopathology patches, patient-level
        train/val/test split (no patient overlap between splits)

        **Test performance (at tuned threshold):**
        - Malignant recall: **0.90**
        - Malignant precision: 0.72
        - Overall accuracy: 0.79

        **Why recall-optimized:** the decision threshold was deliberately
        tuned to prioritize catching malignant cases over minimizing false
        alarms, since a missed cancer case is far more costly than a false
        positive in this clinical context.
        """
    )
    st.divider()
    st.caption(
        "👩‍💻 **Portfolio project by Maria Febus** — Master of Applied Data Science, "
        "University of Michigan (2026)"
    )
    st.caption(
        "⚠️ Educational / demonstration prototype only. Not a diagnostic device, not "
        "clinically validated, and not a substitute for pathologist review."
    )


# --------------------------------------------------------------------------
# Header -- styled to match the companion dashboard artifact
# --------------------------------------------------------------------------
st.markdown(
    """
    <div class="triage-header">
        <p>A convolutional neural network for classifying breast histopathology patches
        as benign or malignant, optimized for recall to minimize missed diagnoses and
        support faster, more consistent specimen review.</p>
        <span class="triage-byline">M. Febus · Diagnostic Triage Dash — Live Prediction</span>
    </div>
    """,
    unsafe_allow_html=True,
)

st.info(
    "🎓 **Educational / Portfolio Demonstration Project** — This app showcases an "
    "end-to-end machine learning pipeline (data acquisition, transfer learning, "
    "class imbalance handling, threshold tuning) built as part of a Master of Applied "
    "Data Science program. It is **not** a medical device, has **not** been clinically "
    "validated, and must **not** be used for actual diagnosis or clinical decision-making."
)

# Real stats strip -- same numbers as the companion dashboard artifact
stat_cols = st.columns(4)
stat_cols[0].markdown(
    '<span class="stat-label">Recall (Malignant)</span><span class="stat-value">90.0%</span>',
    unsafe_allow_html=True,
)
stat_cols[1].markdown(
    '<span class="stat-label">Precision</span><span class="stat-value">72.2%</span>',
    unsafe_allow_html=True,
)
stat_cols[2].markdown(
    '<span class="stat-label">Decision Threshold</span><span class="stat-value">0.449</span>',
    unsafe_allow_html=True,
)
stat_cols[3].markdown(
    '<span class="stat-label">Test Set Size</span><span class="stat-value">8,717</span>',
    unsafe_allow_html=True,
)

st.write("")
st.subheader("Upload a Specimen Patch")
st.markdown(
    "Upload a histopathology image patch (50x50px IDC-style tissue image) "
    "for a **real prediction** from the trained model — this is the live, functional "
    "counterpart to the static dashboard mockup."
)

model, threshold, target_recall = load_model_and_threshold()

if model is None:
    st.error(
        f"Model file not found at `{MODEL_PATH}`.\n\n"
        "Copy `breast_cancer_efficientnet_final.keras` and `threshold.json` "
        "from your Drive's `breast_cancer_project/models/` folder into a "
        "local `model/` folder next to this app, then restart Streamlit."
    )
    st.stop()

uploaded_files = st.file_uploader(
    "Choose one or more image files",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True,
)

if uploaded_files:
    st.markdown(f"**{len(uploaded_files)} patch(es) uploaded** — running inference on each...")

    results = []  # collected for the summary table below
    cols_per_row = 3

    for row_start in range(0, len(uploaded_files), cols_per_row):
        row_files = uploaded_files[row_start : row_start + cols_per_row]
        row_cols = st.columns(cols_per_row)

        for col, file in zip(row_cols, row_files):
            pil_image = Image.open(file)
            input_array = preprocess_image(pil_image)
            probability = float(model.predict(input_array, verbose=0)[0][0])
            is_malignant = probability > threshold
            results.append(
                {
                    "File": file.name,
                    "Prediction": "Malignant" if is_malignant else "Benign",
                    "P(malignant)": round(probability, 3),
                }
            )

            with col:
                st.image(pil_image, use_container_width=True)
                if is_malignant:
                    st.error(f"**Malignant** — {probability:.3f}")
                else:
                    st.success(f"**Benign** — {probability:.3f}")
                st.progress(probability)

    st.divider()
    st.subheader("Summary")
    st.dataframe(results, use_container_width=True, hide_index=True)
    st.caption(
        f"Decision threshold: {threshold:.3f}"
        + (f" (tuned for recall ≥ {target_recall:.0%})" if target_recall else "")
    )

    st.divider()
    st.markdown(
        """
        **How to read this:** the model outputs a probability between 0 and 1
        that each tissue patch is malignant. That probability is compared
        against a **tuned decision threshold** (not the naive 0.5 default) —
        chosen specifically to catch as many true malignant cases as
        possible, at the cost of some false positives on benign tissue.
        """
    )
else:
    st.info("Upload one or more images above to see predictions.")
