"""
Breast Cancer Detection — Streamlit Prototype
Loads the trained EfficientNetB0 classifier and runs real predictions on
uploaded histopathology image patches.

Run locally with:
    streamlit run app.py

Requires the trained model files to be present at:
    model/breast_cancer_efficientnet_final.keras
    model/threshold.json
"""

import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input

# --------------------------------------------------------------------------
# Page config -- must be the first Streamlit call
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="HistoTriage — Breast Cancer CNN",
    page_icon="🔬",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "model" / "breast_cancer_efficientnet_final.keras"
THRESHOLD_PATH = BASE_DIR / "model" / "threshold.json"
LOGO_PATH = BASE_DIR / "assets" / "histotriage_logo.png"
IMG_SIZE = 224  # must match the size the model was trained on

# --------------------------------------------------------------------------
# Custom styling -- matches the companion dashboard artifact's visual
# language (navy header, cream background, brick accent, monospace labels)
# so the static mockup and this live app read as one cohesive product.
# --------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

    :root {
        --navy: #07182d;
        --navy-2: #12334a;
        --teal: #2d8b8c;
        --teal-soft: #dcefed;
        --rust: #c86443;
        --rust-soft: #f6e5df;
        --cream: #f7f3eb;
        --paper: #fffdf8;
        --ink: #172033;
        --muted: #657184;
        --line: #e5ded2;
        --shadow: 0 12px 32px rgba(16, 35, 55, 0.08);
    }

    html, body, [class*="css"], .stApp, .stMarkdown, p, li, span, label {
        font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at 8% 0%, rgba(45,139,140,0.08), transparent 28%),
            radial-gradient(circle at 96% 12%, rgba(200,100,67,0.07), transparent 25%),
            var(--cream);
        color: var(--ink);
    }

    .block-container {
        max-width: 1550px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    /* Hero container that holds the generated logo and compact supporting copy. */
    .logo-banner {
        background: linear-gradient(135deg, var(--navy) 0%, var(--navy-2) 100%);
        border-radius: 18px;
        padding: 10px;
        margin-bottom: 8px;
        box-shadow: 0 16px 38px rgba(7,24,45,0.16);
        border: 1px solid rgba(255,255,255,0.07);
    }

    .hero-summary {
        background: rgba(255,253,248,0.96);
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 15px 18px;
        margin: 0 0 18px;
        box-shadow: 0 8px 22px rgba(23,32,51,0.045);
    }

    .hero-copy {
        color: #445064;
        font-size: 15px;
        line-height: 1.6;
        margin: 0 0 10px;
        max-width: 960px;
    }

    .hero-meta {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        padding: 6px 10px;
        border-radius: 999px;
        background: var(--teal-soft);
        border: 1px solid #a9cfcc;
        color: #246d6e;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 9.5px;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    .notice-card {
        display: flex;
        gap: 13px;
        align-items: flex-start;
        background: rgba(255,253,248,0.96);
        border: 1px solid #eadbd4;
        border-left: 5px solid var(--rust);
        border-radius: 14px;
        padding: 16px 18px;
        margin-bottom: 20px;
        box-shadow: 0 8px 24px rgba(23,32,51,0.05);
    }

    .notice-icon {
        min-width: 34px;
        height: 34px;
        display: grid;
        place-items: center;
        border-radius: 10px;
        background: var(--rust-soft);
        font-size: 17px;
    }

    .notice-title {
        color: var(--rust);
        font-weight: 700;
        margin-bottom: 3px;
    }

    .notice-copy {
        color: #604f49;
        font-size: 13.5px;
        line-height: 1.55;
        margin: 0;
    }

    .stat-card {
        background: rgba(255,253,248,0.96);
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 17px 17px 15px;
        min-height: 112px;
        box-shadow: 0 8px 25px rgba(23,32,51,0.055);
        transition: transform 0.18s ease, box-shadow 0.18s ease;
    }

    .stat-card:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow);
    }

    .stat-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 10px;
        font-weight: 600;
        letter-spacing: 0.10em;
        text-transform: uppercase;
        color: var(--muted);
        display: block;
        margin-bottom: 8px;
    }

    .stat-value {
        font-size: 30px;
        line-height: 1;
        font-weight: 700;
        color: var(--navy);
        letter-spacing: -0.035em;
    }

    .stat-accent {
        width: 34px;
        height: 3px;
        border-radius: 999px;
        background: linear-gradient(90deg, var(--teal), var(--rust));
        margin-top: 13px;
    }

    h1, h2, h3 {
        color: var(--navy) !important;
        letter-spacing: -0.025em;
    }

    h2 { font-size: 25px !important; }
    h3 { font-size: 19px !important; }

    .section-intro {
        color: var(--muted);
        font-size: 15.5px;
        line-height: 1.65;
        margin-top: -4px;
        margin-bottom: 12px;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f2eee5 0%, #ece6da 100%);
        border-right: 1px solid #ded6c9;
    }

    section[data-testid="stSidebar"] h2 {
        font-family: 'DM Sans', sans-serif;
        font-size: 16px !important;
        font-weight: 700;
        color: var(--navy);
    }

    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] li {
        font-size: 13.5px;
        line-height: 1.55;
        color: #3f4958;
    }

    .sidebar-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 9.5px;
        font-weight: 600;
        letter-spacing: 0.11em;
        text-transform: uppercase;
        color: var(--teal);
        display: block;
        margin-top: 17px;
        margin-bottom: 5px;
    }

    div[data-testid="stFileUploader"] {
        background: rgba(255,253,248,0.96);
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 10px;
        box-shadow: 0 8px 24px rgba(23,32,51,0.045);
    }

    div[data-testid="stFileUploader"] section {
        border: 1.5px dashed #8eb9b7;
        border-radius: 12px;
        background: #f7fbfa;
    }

    div[data-testid="stFileUploader"] button {
        border-radius: 999px !important;
        border: 1px solid var(--teal) !important;
        color: var(--teal) !important;
        font-weight: 600 !important;
    }

    div[data-testid="stAlert"] {
        border-radius: 12px;
    }

    div[data-testid="stImage"] img {
        border-radius: 14px;
        border: 1px solid #e5ddd1;
        box-shadow: 0 10px 24px rgba(23,32,51,0.08);
    }

    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, var(--teal), var(--rust));
    }

    div[data-testid="stDataFrame"] {
        border-radius: 14px;
        overflow: hidden;
        border: 1px solid var(--line);
        box-shadow: 0 8px 22px rgba(23,32,51,0.045);
    }

    .queue-card {
        background: rgba(255,253,248,0.96);
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 20px 22px;
        margin-top: 14px;
        box-shadow: 0 8px 26px rgba(23,32,51,0.05);
    }

    .queue-heading {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 11px;
        letter-spacing: 0.11em;
        text-transform: uppercase;
        color: var(--navy);
        font-weight: 600;
    }

    .coming-soon-pill {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 9.5px;
        letter-spacing: 0.07em;
        text-transform: uppercase;
        color: var(--rust);
        background: var(--rust-soft);
        border: 1px solid #dfae9e;
        border-radius: 999px;
        padding: 4px 9px;
        margin-left: 9px;
    }

    .queue-copy {
        margin: 12px 0 0;
        font-size: 14px;
        color: var(--muted);
        line-height: 1.55;
        max-width: 880px;
    }

    .filter-preview-note {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 9.5px;
        letter-spacing: 0.07em;
        text-transform: uppercase;
        color: var(--teal);
        background: var(--teal-soft);
        border: 1px solid #a9cfcc;
        border-radius: 999px;
        padding: 4px 9px;
        display: inline-block;
        margin: 14px 0 10px;
    }

    div[data-baseweb="select"] > div,
    div[data-testid="stTextInput"] input {
        border-radius: 10px !important;
        border-color: #d7d0c4 !important;
        background: rgba(255,253,248,0.96) !important;
    }

    @media (max-width: 720px) {
        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }

        .logo-banner {
            border-radius: 14px;
            padding: 7px;
        }

        .hero-copy {
            font-size: 14px;
        }
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

    # Keras 3 requires explicit help to reload the Lambda layer that wraps
    # EfficientNet's preprocess_input function: custom_objects tells it WHERE
    # to find that function, and safe_mode=False is required because Keras 3
    # blocks Lambda-layer deserialization by default (a security measure,
    # since Lambda layers can technically run arbitrary code -- safe here
    # since this is our own trained model, not an untrusted download).
    model = tf.keras.models.load_model(
        str(MODEL_PATH),
        custom_objects={'preprocess_input': preprocess_input},
        safe_mode=False,
    )

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

    st.markdown('<span class="sidebar-label">Architecture</span>', unsafe_allow_html=True)
    st.markdown("EfficientNetB0 (ImageNet-pretrained, fine-tuned)")

    st.markdown('<span class="sidebar-label">Training Data</span>', unsafe_allow_html=True)
    st.markdown(
        "IDC breast histopathology patches, patient-level train/val/test "
        "split (no patient overlap between splits)"
    )

    st.markdown('<span class="sidebar-label">Test Performance (Tuned Threshold)</span>', unsafe_allow_html=True)
    st.markdown(
        """
        - Malignant recall: **0.90**
        - Malignant precision: 0.72
        - Overall accuracy: 0.79
        """
    )

    st.markdown('<span class="sidebar-label">Why Recall-Optimized</span>', unsafe_allow_html=True)
    st.markdown(
        "The decision threshold was deliberately tuned to prioritize "
        "catching malignant cases over minimizing false alarms, since a "
        "missed cancer case is far more costly than a false positive in "
        "this clinical context."
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
# Header -- generated HistoTriage logo plus supporting product statement
# --------------------------------------------------------------------------
if LOGO_PATH.exists():
    st.markdown('<div class="logo-banner">', unsafe_allow_html=True)

    st.image(str(LOGO_PATH), width=2000
    )

    st.markdown("</div>", unsafe_allow_html=True)
else:
    st.markdown(
        """
        <div class="logo-banner" style="color:white; font-size:36px; font-weight:700;">
            Histo<span style="color:#66bec0;">Triage</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <div class="notice-card">
        <div class="notice-icon">🎓</div>
        <div>
            <div class="notice-title">Educational / Portfolio Demonstration Project</div>
            <p class="notice-copy">
                Upload breast histopathology patches and explore AI-assisted breast cancer triage using a recall-optimized EfficientNetB0 model. This interactive prototype showcases an end-to-end machine learning workflow, including transfer learning, class-imbalance handling, threshold optimization, and live model inference.
                This application is intended for educational and portfolio demonstration purposes only. It has not been clinically validated and must not be used for diagnosis or clinical decision-making.
            </p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Real stats strip -- same numbers as the companion dashboard artifact
stat_cols = st.columns(4)
stat_cols[0].markdown(
    '<div class="stat-card"><span class="stat-label">Recall · Malignant</span><span class="stat-value">90.0%</span><div class="stat-accent"></div></div>',
    unsafe_allow_html=True,
)
stat_cols[1].markdown(
    '<div class="stat-card"><span class="stat-label">Precision</span><span class="stat-value">72.2%</span><div class="stat-accent"></div></div>',
    unsafe_allow_html=True,
)
stat_cols[2].markdown(
    '<div class="stat-card"><span class="stat-label">Decision Threshold</span><span class="stat-value">0.449</span><div class="stat-accent"></div></div>',
    unsafe_allow_html=True,
)
stat_cols[3].markdown(
    '<div class="stat-card"><span class="stat-label">Test Set Size</span><span class="stat-value">8,717</span><div class="stat-accent"></div></div>',
    unsafe_allow_html=True,
)

st.write("")
st.subheader("Upload a Specimen Patch")
st.markdown(
    """
    <p class="section-intro">
        Upload one or more 50×50 IDC-style histopathology patches to receive a
        live prediction from the trained model. This is the functional counterpart
        to the project's application prototype.
    </p>
    """,
    unsafe_allow_html=True,
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

# --------------------------------------------------------------------------
# Review Queue -- honest "coming soon" placeholder, not a fake working
# feature. Sample names are illustrative only, clearly labeled as such.
# --------------------------------------------------------------------------
st.write("")
st.markdown(
    """
    <div class="queue-card">
        <span class="queue-heading">Review Queue</span>
        <span class="coming-soon-pill">Coming Soon</span>
        <p class="queue-copy">
            A future version could retain uploaded cases, sort them by confidence,
            and help a pathologist prioritize the patches most likely to require
            immediate review. The interactive preview below uses illustrative data only.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

_sample_queue = [
    {"Patient": "Jane Doe", "Confidence": 0.94, "Class": "Malignant", "Status": "Pending"},
    {"Patient": "John Smith", "Confidence": 0.88, "Class": "Malignant", "Status": "Pending"},
    {"Patient": "Maria Garcia", "Confidence": 0.81, "Class": "Malignant", "Status": "In Review"},
    {"Patient": "Robert Chen", "Confidence": 0.73, "Class": "Malignant", "Status": "Pending"},
    {"Patient": "Emily Johnson", "Confidence": 0.61, "Class": "Malignant", "Status": "Pending"},
    {"Patient": "Ava Patel", "Confidence": 0.42, "Class": "Benign", "Status": "Reviewed"},
    {"Patient": "Daniel Kim", "Confidence": 0.27, "Class": "Benign", "Status": "Reviewed"},
]

st.markdown(
    '<span class="filter-preview-note">Interactive Preview · Future Feature</span>',
    unsafe_allow_html=True,
)

filter_cols = st.columns([1.1, 1.1, 1.35, 1.55])

with filter_cols[0]:
    class_filter = st.selectbox(
        "Predicted class", ["All", "Malignant", "Benign"], key="queue_class_filter"
    )

with filter_cols[1]:
    status_filter = st.selectbox(
        "Review status", ["All", "Pending", "In Review", "Reviewed"], key="queue_status_filter"
    )

with filter_cols[2]:
    min_confidence = st.slider(
        "Minimum confidence", min_value=0.0, max_value=1.0, value=0.0, step=0.05,
        key="queue_confidence_filter"
    )

with filter_cols[3]:
    patient_search = st.text_input(
        "Search illustrative patient", placeholder="Type a sample name", key="queue_patient_search"
    )

filtered_queue = [
    row for row in _sample_queue
    if (class_filter == "All" or row["Class"] == class_filter)
    and (status_filter == "All" or row["Status"] == status_filter)
    and row["Confidence"] >= min_confidence
    and (not patient_search or patient_search.lower() in row["Patient"].lower())
]

display_queue = [{**row, "Confidence": f'{row["Confidence"]:.2f}'} for row in filtered_queue]

st.dataframe(display_queue, use_container_width=True, hide_index=True)
st.caption(
    "Interactive preview using illustrative sample data only. "
    "No real patient records are connected, stored, or processed."
)
