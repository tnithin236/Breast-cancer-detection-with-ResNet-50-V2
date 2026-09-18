import os
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.resnet_v2 import preprocess_input

# ============================================================
# Configuration
# ============================================================

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cancer_model.h5")

CLASSES = [
    {
        "name": "Benign",
        "color": "#19d99b",
        "note": "Non-cancerous tissue indicated."
    },
    {
        "name": "Malignant",
        "color": "#ff3f70",
        "note": "Cancerous tissue indicated."
    },
]

CLASS_NAMES = [c["name"] for c in CLASSES]

KNOWN_METRICS = {
    "Macro AUC": None,
}

# ============================================================
# Page
# ============================================================

st.set_page_config(
    page_title="Breast Cancer Classifier",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# Modern UI styling
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --bg: #020b1d;
        --panel: #06162f;
        --panel-2: #071b39;
        --border: #12345f;
        --blue: #38a8ff;
        --purple: #9b6cff;
        --text: #edf5ff;
        --muted: #8ca6c8;
        --green: #19d99b;
        --red: #ff3f70;
    }

    * {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at 80% 0%, rgba(50, 106, 255, 0.13), transparent 32%),
            radial-gradient(circle at 20% 100%, rgba(123, 68, 255, 0.08), transparent 30%),
            #020b1d;
        color: var(--text);
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    section[data-testid="stSidebar"] {
        background:
            linear-gradient(180deg, #020b1d 0%, #04122a 100%);
        border-right: 1px solid #102b4f;
    }

    section[data-testid="stSidebar"] > div {
        padding-top: 1.4rem;
    }

    .block-container {
        max-width: 1500px;
        padding: 2.1rem 2.5rem 3rem 2.2rem;
    }

    /* Hide Streamlit chrome that is not part of the application */
    #MainMenu, footer {
        visibility: hidden;
    }

    /* Sidebar */
    .brand {
        padding: 0.2rem 0.4rem 1.4rem 0.4rem;
        border-bottom: 1px solid #173457;
        margin-bottom: 1.3rem;
    }

    .brand-row {
        display: flex;
        align-items: center;
        gap: 13px;
    }

    .ribbon {
        width: 42px;
        height: 42px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 31px;
        filter: drop-shadow(0 0 9px rgba(255, 70, 145, 0.35));
    }

    .brand-title {
        color: #f2f7ff;
        font-size: 20px;
        line-height: 1.15;
        font-weight: 800;
        letter-spacing: -0.5px;
    }

    .brand-title span {
        color: #72b9ff;
    }

    .brand-sub {
        color: #7894b9;
        font-size: 10px;
        margin-top: 5px;
        letter-spacing: 0.1px;
    }

    .side-card {
        background: linear-gradient(145deg, rgba(8, 29, 60, 0.96), rgba(3, 17, 39, 0.96));
        border: 1px solid #12345f;
        border-radius: 13px;
        padding: 17px 16px 14px;
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.18);
    }

    .side-stat {
        display: flex;
        align-items: center;
        gap: 13px;
        padding: 11px 0;
        border-bottom: 1px solid #102d51;
    }

    .side-stat:last-of-type {
        border-bottom: none;
    }

    .stat-icon {
        width: 38px;
        height: 38px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: rgba(48, 151, 255, 0.09);
        border: 1px solid rgba(48, 151, 255, 0.2);
        color: #42adff;
        font-size: 20px;
    }

    .stat-label {
        color: #7591b5;
        font-size: 10px;
        letter-spacing: 0.7px;
        text-transform: uppercase;
        margin-bottom: 3px;
    }

    .stat-value {
        color: #edf5ff;
        font-size: 15px;
        font-weight: 600;
    }

    .side-section-title {
        color: #8ca6c8;
        font-size: 11px;
        letter-spacing: 0.8px;
        text-transform: uppercase;
        margin: 19px 0 11px;
    }

    .class-item {
        display: flex;
        align-items: center;
        gap: 10px;
        color: #dbe9fa;
        font-size: 13px;
        margin: 11px 0;
    }

    .class-dot {
        width: 12px;
        height: 12px;
        border-radius: 50%;
        display: inline-block;
        box-shadow: 0 0 9px currentColor;
    }

    .research-box {
        margin-top: 22px;
        border: 1px solid #304b86;
        border-radius: 12px;
        padding: 15px 14px;
        background: linear-gradient(145deg, rgba(20, 29, 77, 0.42), rgba(5, 17, 43, 0.5));
    }

    .research-title {
        color: #e8efff;
        font-size: 12px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .research-text {
        color: #8199bc;
        font-size: 10.5px;
        line-height: 1.65;
    }

    /* Main header */
    .topline {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 7px;
    }

    .page-title {
        font-size: 32px;
        font-weight: 800;
        letter-spacing: -1px;
        margin: 0;
        color: #f3f7ff;
    }

    .page-title span {
        color: #a979ff;
    }

    .page-subtitle {
        color: #89a5c8;
        font-size: 13px;
        margin-bottom: 25px;
    }

    .project-badge {
        border: 1px solid #163f72;
        background: rgba(14, 49, 93, 0.55);
        color: #71baff;
        border-radius: 22px;
        padding: 9px 15px;
        font-size: 11px;
        font-weight: 600;
        white-space: nowrap;
    }

    /* Upload */
    .upload-label {
        color: #d9e9ff;
        font-size: 13px;
        font-weight: 600;
        margin-bottom: 8px;
    }

    div[data-testid="stFileUploader"] {
        background: linear-gradient(110deg, rgba(5, 26, 57, 0.9), rgba(4, 20, 46, 0.75));
        border: 1px dashed #318dff;
        border-radius: 14px;
        padding: 6px 10px;
        box-shadow: inset 0 0 30px rgba(39, 127, 255, 0.035);
    }

    div[data-testid="stFileUploader"] section {
        border: none !important;
        background: transparent !important;
    }

    div[data-testid="stFileUploader"] button {
        background: linear-gradient(135deg, #277cff, #7754e8);
        border: none;
        color: white;
        border-radius: 8px;
        font-weight: 600;
    }

    div[data-testid="stFileUploader"] small {
        color: #718db3 !important;
    }

    /* Panels */
    .panel {
        background: linear-gradient(145deg, rgba(6, 24, 52, 0.97), rgba(3, 16, 37, 0.97));
        border: 1px solid #113762;
        border-radius: 14px;
        padding: 16px;
        min-height: 560px;
        box-shadow: 0 20px 50px rgba(0, 0, 0, 0.16);
    }

    .panel-heading {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 13px;
    }

    .panel-title {
        color: #edf5ff;
        font-size: 16px;
        font-weight: 700;
    }

    .ready {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        color: #21e3a4;
        background: rgba(25, 217, 155, 0.09);
        border: 1px solid rgba(25, 217, 155, 0.25);
        padding: 6px 10px;
        border-radius: 20px;
        font-size: 10px;
        font-weight: 700;
    }

    .ready-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: #19d99b;
        box-shadow: 0 0 8px #19d99b;
    }

    .scan-wrap {
        background: #000;
        border: 1px solid #102f55;
        border-radius: 11px;
        min-height: 390px;
        display: flex;
        align-items: center;
        justify-content: center;
        overflow: hidden;
    }

    .file-meta {
        position: absolute;
    }

    .control-bar {
        display: flex;
        align-items: center;
        gap: 9px;
        background: #071c3a;
        border: 1px solid #12365f;
        border-radius: 10px;
        margin-top: 8px;
        padding: 9px;
    }

    .control {
        color: #a9c2e2;
        background: #0c2a50;
        border-radius: 8px;
        padding: 7px 12px;
        font-size: 11px;
    }

    .control.download {
        margin-left: auto;
    }

    /* Result */
    .result-card {
        border: 1px solid;
        border-radius: 13px;
        padding: 24px 22px;
        margin-bottom: 23px;
    }

    .result-card.malignant {
        background: linear-gradient(115deg, rgba(118, 18, 62, 0.33), rgba(35, 16, 53, 0.28));
        border-color: #ff3268;
        box-shadow: inset 0 0 30px rgba(255, 48, 105, 0.04);
    }

    .result-card.benign {
        background: linear-gradient(115deg, rgba(9, 99, 75, 0.25), rgba(8, 31, 51, 0.28));
        border-color: #19d99b;
    }

    .result-row {
        display: flex;
        align-items: center;
        gap: 16px;
    }

    .result-icon {
        width: 52px;
        height: 52px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 25px;
        font-weight: 800;
    }

    .malignant .result-icon {
        background: #ff466f;
        color: white;
        box-shadow: 0 0 22px rgba(255, 70, 111, 0.2);
    }

    .benign .result-icon {
        background: #19d99b;
        color: #021a1b;
    }

    .result-label {
        font-size: 23px;
        font-weight: 800;
        letter-spacing: 0.2px;
    }

    .malignant .result-label {
        color: #ff4772;
    }

    .benign .result-label {
        color: #19d99b;
    }

    .result-confidence {
        color: #eef5ff;
        font-size: 14px;
        margin-top: 5px;
    }

    .result-note {
        color: #8ca6c8;
        font-size: 11px;
        margin-top: 17px;
        line-height: 1.5;
    }

    .prob-title {
        color: #dceaff;
        font-size: 15px;
        font-weight: 700;
        margin: 7px 0 20px;
    }

    .prob-label {
        display: flex;
        justify-content: space-between;
        color: #d9e7f8;
        font-size: 12px;
        margin-bottom: 7px;
    }

    .prob-track {
        width: 100%;
        height: 9px;
        background: #152844;
        border-radius: 20px;
        overflow: hidden;
        margin-bottom: 17px;
    }

    .prob-fill {
        height: 100%;
        border-radius: 20px;
    }

    .disclaimer {
        border: 1px solid #806500;
        background: rgba(65, 52, 4, 0.34);
        border-radius: 10px;
        padding: 13px;
        color: #d6b632;
        font-size: 10px;
        line-height: 1.55;
        margin-top: 24px;
    }

    .empty-state {
        min-height: 390px;
        display: flex;
        align-items: center;
        justify-content: center;
        text-align: center;
        color: #617d9f;
        font-size: 12px;
        border: 1px dashed #173b65;
        border-radius: 11px;
        background: rgba(0, 0, 0, 0.13);
    }

    /* Prediction button */
    .stButton > button {
        width: 100%;
        border: none;
        border-radius: 9px;
        padding: 0.65rem 1rem;
        background: linear-gradient(135deg, #ff4668, #ff3d58);
        color: white;
        font-weight: 700;
        box-shadow: 0 8px 20px rgba(255, 62, 91, 0.14);
    }

    .stButton > button:hover {
        border: none;
        background: linear-gradient(135deg, #ff5977, #ff4962);
        color: white;
    }


    /* Robust Streamlit-native panels */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: linear-gradient(145deg, rgba(6, 24, 52, 0.98), rgba(3, 16, 37, 0.98));
        border: 1px solid #123762 !important;
        border-radius: 14px !important;
        box-shadow: 0 20px 50px rgba(0,0,0,.16);
    }

    [data-testid="stVerticalBlockBorderWrapper"] > div {
        padding: 14px 16px !important;
    }

    .simple-stat {
        display:flex;
        align-items:center;
        gap:11px;
        padding:11px 10px;
        margin:7px 0;
        border:1px solid #12345f;
        border-radius:10px;
        background:rgba(6,26,54,.75);
    }

    .simple-stat-icon {
        width:34px;
        height:34px;
        display:flex;
        align-items:center;
        justify-content:center;
        border-radius:9px;
        color:#43adff;
        background:rgba(48,151,255,.09);
        border:1px solid rgba(48,151,255,.18);
        font-size:18px;
    }

    .simple-stat-text {
        display:flex;
        flex-direction:column;
        gap:2px;
    }

    .simple-stat-text small {
        color:#718caf;
        font-size:8px;
        letter-spacing:.8px;
    }

    .simple-stat-text b {
        color:#eef5ff;
        font-size:14px;
    }

    .class-line {
        color:#dceaff;
        font-size:12px;
        margin:9px 3px;
        display:flex;
        align-items:center;
        gap:9px;
    }

    .green-dot,.red-dot {
        width:9px;
        height:9px;
        border-radius:50%;
        display:inline-block;
    }

    .green-dot { background:#19d99b; box-shadow:0 0 8px #19d99b; }
    .red-dot { background:#ff3f70; box-shadow:0 0 8px #ff3f70; }

    .image-meta {
        display:flex;
        gap:7px;
        align-items:center;
        padding:8px;
        margin-top:8px;
        border-radius:9px;
        background:#071c3a;
        border:1px solid #12365f;
        color:#9bb5d5;
        font-size:10px;
    }

    .image-meta .file-pill {
        margin-left:auto;
        background:#0b2a51;
        border-radius:7px;
        padding:6px 9px;
        max-width:65%;
        overflow:hidden;
        text-overflow:ellipsis;
        white-space:nowrap;
    }

    .gallery-header {
        margin-top:30px;
        margin-bottom:13px;
    }

    .gallery-title {
        color:#edf5ff;
        font-size:20px;
        font-weight:800;
    }

    .gallery-subtitle {
        color:#7894b9;
        font-size:11px;
        margin-top:4px;
    }

    .gallery-label {
        font-size:9px;
        font-weight:800;
        letter-spacing:.7px;
        margin-bottom:8px;
    }

    .gallery-name {
        color:#e8f1ff;
        font-size:11px;
        font-weight:700;
        margin-top:8px;
    }

    .gallery-source {
        color:#617d9f;
        font-size:9px;
        margin-top:3px;
    }

    /* Mobile */
    @media (max-width: 900px) {
        .block-container {
            padding: 1.2rem 1rem 2rem;
        }

        .page-title {
            font-size: 26px;
        }

        .project-badge {
            display: none;
        }

        .panel {
            min-height: auto;
            margin-bottom: 15px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# Model loading
# ============================================================

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}. "
            "Place cancer_model.h5 in the same folder as this app."
        )
    return tf.keras.models.load_model(MODEL_PATH)


def get_input_spec(model):
    shape = model.input_shape
    if isinstance(shape, list):
        shape = shape[0]

    _, h, w, c = shape
    return int(h or 224), int(w or 224), int(c or 3)


def preprocess_image(img: Image.Image, height, width, channels):
    if channels == 1:
        proc = img.convert("L").resize((width, height))
        arr = np.array(proc).astype("float32") / 255.0
        arr = np.expand_dims(arr, axis=-1)
    else:
        proc = img.convert("RGB").resize((width, height))
        arr = np.array(proc).astype("float32")
        arr = preprocess_input(arr)

    return np.expand_dims(arr, axis=0), proc


def predict(model, img_array):
    preds = np.array(model.predict(img_array, verbose=0)).squeeze()

    if preds.ndim == 0:
        prob_malignant = float(preds)
        probs = {
            "Benign": 1 - prob_malignant,
            "Malignant": prob_malignant,
        }
    else:
        probs = {
            CLASS_NAMES[i]: float(preds[i])
            for i in range(len(CLASS_NAMES))
        }

    predicted = max(probs, key=probs.get)
    return predicted, probs[predicted], probs


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    st.markdown(
        """
        <div class="brand">
            <div class="brand-row">
                <div class="ribbon">🎀</div>
                <div>
                    <div class="brand-title">Breast Cancer<br><span>Classifier</span></div>
                    <div class="brand-sub">AI Powered • Medical Imaging</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    model_loaded = False

    try:
        model = load_model()
        height, width, channels = get_input_spec(model)
        model_loaded = True
    except Exception as e:
        st.error("Failed to load model.")
        st.exception(e)

    if model_loaded:
        size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
        params = model.count_params()

        st.markdown(
            f"""
            <div class="simple-stat">
                <span class="simple-stat-icon">◇</span>
                <span class="simple-stat-text"><small>MODEL SIZE</small><b>{size_mb:.1f} MB</b></span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="simple-stat">
                <span class="simple-stat-icon">✣</span>
                <span class="simple-stat-text"><small>PARAMETERS</small><b>{params/1e6:.1f}M</b></span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="simple-stat">
                <span class="simple-stat-icon">▣</span>
                <span class="simple-stat-text"><small>ARCHITECTURE</small><b>ResNet-50 V2</b></span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="side-section-title">Classes</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="class-line"><span class="green-dot"></span>Benign</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="class-line"><span class="red-dot"></span>Malignant</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="research-box">
                <div class="research-title">ⓘ &nbsp; For research purposes only.</div>
                <div class="research-text">
                    Not intended for clinical diagnosis.<br>
                    Always consult a qualified radiologist.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ============================================================
# Main
# ============================================================

if not model_loaded:
    st.stop()

st.markdown(
    """
    <div class="topline">
        <div>
            <div class="page-title">Upload <span>Mammogram</span></div>
        </div>
        <div class="project-badge">⚗ &nbsp; Research Project</div>
    </div>
    <div class="page-subtitle">
        Our AI model analyzes the scan and predicts whether it is Benign or Malignant.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="upload-label">Upload a breast scan image</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Drag and drop your mammogram image here",
    type=["png", "jpg", "jpeg", "bmp"],
    label_visibility="collapsed",
)

if uploaded_file is not None:
    try:
        original_img = Image.open(uploaded_file).convert("RGB")

        st.markdown(
            f"""
            <div style="
                margin-top:-5px;
                margin-bottom:12px;
                color:#7592b5;
                font-size:10px;">
                Supported formats: JPG, PNG, JPEG, BMP &nbsp; • &nbsp;
                File: {uploaded_file.name}
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Clear an old result when a different file is uploaded.
        current_file_id = f"{uploaded_file.name}_{uploaded_file.size}"
        if st.session_state.get("file_id") != current_file_id:
            st.session_state["prediction"] = None
            st.session_state["file_id"] = current_file_id

        if st.button("Run Prediction", type="primary"):
            with st.spinner("Analyzing mammogram..."):
                img_array, _ = preprocess_image(
                    original_img, height, width, channels
                )

                raw_preds = model.predict(img_array, verbose=0)
                raw_preds = np.asarray(raw_preds).squeeze()

                if raw_preds.ndim == 0:
                    # Binary sigmoid: 0 = Benign, 1 = Malignant.
                    malignant_prob = float(np.clip(raw_preds, 0.0, 1.0))
                    probs = {
                        "Benign": 1.0 - malignant_prob,
                        "Malignant": malignant_prob,
                    }
                elif raw_preds.size == 2:
                    # Two-class output. Convert logits to probabilities if needed.
                    values = raw_preds.astype("float32").reshape(-1)
                    if (
                        np.any(values < 0)
                        or np.any(values > 1)
                        or not np.isclose(np.sum(values), 1.0, atol=0.01)
                    ):
                        exp_values = np.exp(values - np.max(values))
                        values = exp_values / np.sum(exp_values)

                    probs = {
                        "Benign": float(values[0]),
                        "Malignant": float(values[1]),
                    }
                else:
                    raise ValueError(
                        f"Unexpected model output shape: {raw_preds.shape}. "
                        f"Output: {raw_preds}"
                    )

                predicted_class = max(probs, key=probs.get)
                confidence = float(probs[predicted_class])

                st.session_state["prediction"] = {
                    "class": predicted_class,
                    "confidence": confidence,
                    "probs": probs,
                }

            st.success("Prediction completed successfully.")

    except Exception as e:
        st.error("Unable to generate prediction.")
        with st.expander("Show technical error"):
            st.exception(e)

# ============================================================
# Results
# ============================================================

prediction = st.session_state.get("prediction")

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    with st.container(border=True):
        heading_col, status_col = st.columns([3, 1])
        with heading_col:
            st.markdown("### ▧  Uploaded Scan")
        with status_col:
            st.markdown(
                '<div class="ready"><span class="ready-dot"></span>Ready</div>',
                unsafe_allow_html=True,
            )

        if uploaded_file is not None:
            st.image(
                original_img,
                use_container_width=True,
                output_format="PNG",
            )

            st.markdown(
                f"""
                <div class="image-meta">
                    <span>⌕ &nbsp; Zoom</span>
                    <span>⛶ &nbsp; Fit</span>
                    <span class="file-pill">⇩ &nbsp; {uploaded_file.name}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="empty-state">Upload a mammogram image to preview the scan.</div>',
                unsafe_allow_html=True,
            )

with col2:
    with st.container(border=True):
        st.markdown("### ▥  Classification Result")

        if prediction is not None:
            predicted_class = prediction["class"]
            confidence = prediction["confidence"]
            probs = prediction["probs"]

            is_malignant = predicted_class == "Malignant"
            result_class = "malignant" if is_malignant else "benign"
            icon = "!" if is_malignant else "✓"

            note = (
                "Cancerous tissue indicated. Clinical follow-up is advised."
                if is_malignant
                else
                "Non-cancerous tissue indicated. Follow-up as advised by your radiologist."
            )

            st.markdown(
                f"""
                <div class="result-card {result_class}">
                    <div class="result-row">
                        <div class="result-icon">{icon}</div>
                        <div>
                            <div class="result-label">{predicted_class.upper()}</div>
                            <div class="result-confidence">Predicted probability: {confidence * 100:.1f}%</div>
                        </div>
                    </div>
                    <div class="result-note">{note}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("**◔  Class Probabilities**")

            for class_name in ["Benign", "Malignant"]:
                probability = float(probs.get(class_name, 0.0))
                class_color = "#ff3f70" if class_name == "Malignant" else "#19d99b"

                st.markdown(
                    f"""
                    <div class="prob-label">
                        <span>{class_name}</span>
                        <span style="color:{class_color};font-weight:700;">{probability * 100:.1f}%</span>
                    </div>
                    <div class="prob-track">
                        <div class="prob-fill"
                             style="width:{probability * 100:.1f}%;background:{class_color};"></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown(
                """
                <div class="disclaimer">
                    <b>ⓘ &nbsp; For research purposes only.</b><br>
                    Not intended for clinical diagnosis. Always consult a qualified radiologist.
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="empty-state">Upload a mammogram and click Run Prediction to generate the classification result.</div>',
                unsafe_allow_html=True,
            )

# ============================================================
# Reference Gallery
# ============================================================

st.markdown(
    """
    <div class="gallery-header">
        <div class="gallery-title">Mammogram Reference Gallery</div>
        <div class="gallery-subtitle">
            Educational examples only. These reference images are not used for prediction.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

gallery = [
    {
        "label": "BENIGN / NORMAL",
        "title": "Normal mammogram",
        "url": "https://upload.wikimedia.org/wikipedia/commons/9/98/Normal_mammogram.jpg",
        "color": "#19d99b",
        "source": "NCI / NIH",
    },
    {
        "label": "BENIGN / NORMAL",
        "title": "Normal fatty breast",
        "url": "https://upload.wikimedia.org/wikipedia/commons/d/d2/Mammogram_showing_normal_fatty_breast.jpg",
        "color": "#19d99b",
        "source": "NCI / NIH",
    },
    {
        "label": "MALIGNANT",
        "title": "Mammogram showing cancer",
        "url": "https://upload.wikimedia.org/wikipedia/commons/d/dc/Mammogram_showing_cancer.jpg",
        "color": "#ff3f70",
        "source": "NCI / NIH",
    },
    {
        "label": "MALIGNANT",
        "title": "Mammogram with obvious cancer",
        "url": "https://upload.wikimedia.org/wikipedia/commons/3/35/Mammogram_with_obvious_cancer.jpg",
        "color": "#ff3f70",
        "source": "NCI / NIH",
    },
]

gallery_cols = st.columns(4, gap="medium")

for col, item in zip(gallery_cols, gallery):
    with col:
        with st.container(border=True):
            st.markdown(
                f'<div class="gallery-label" style="color:{item["color"]};">● {item["label"]}</div>',
                unsafe_allow_html=True,
            )
            st.image(item["url"], use_container_width=True)
            st.markdown(
                f'<div class="gallery-name">{item["title"]}</div><div class="gallery-source">{item["source"]}</div>',
                unsafe_allow_html=True,
            )

st.caption(
    "Reference examples sourced from National Cancer Institute / NIH images hosted on Wikimedia Commons."
)

