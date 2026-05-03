import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import os

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(
    page_title="Food Image Classifier",
    page_icon="🍽️",
    layout="wide"
)


# -----------------------------
# Custom CSS - Pastel Theme
# -----------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800;900&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at 10% 20%, rgba(255, 183, 197, 0.55), transparent 30%),
            radial-gradient(circle at 90% 15%, rgba(186, 230, 253, 0.65), transparent 32%),
            radial-gradient(circle at 55% 90%, rgba(220, 252, 231, 0.70), transparent 35%),
            linear-gradient(135deg, #fff7ed 0%, #fdf2f8 42%, #eff6ff 100%);
        background-size: 180% 180%;
        animation: gradientMove 14s ease infinite;
        color: #1f2937;
    }

    @keyframes gradientMove {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    .hero {
        position: relative;
        overflow: hidden;
        padding: 2.4rem 2.3rem;
        border-radius: 30px;
        background:
            linear-gradient(135deg, rgba(255, 228, 230, 0.95), rgba(219, 234, 254, 0.95), rgba(220, 252, 231, 0.95));
        border: 1px solid rgba(255,255,255,0.85);
        margin-bottom: 1.6rem;
        box-shadow: 0 22px 55px rgba(148, 163, 184, 0.28);
        animation: fadeUp 0.8s ease both, heroGlow 4s ease-in-out infinite;
    }

    .hero::before {
        content: "🍕 🥗 🍔 🍣 🍝";
        position: absolute;
        top: 14px;
        right: 24px;
        font-size: 2rem;
        opacity: 0.35;
        animation: floatFood 4s ease-in-out infinite;
    }

    .hero::after {
        content: "";
        position: absolute;
        width: 280px;
        height: 280px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(255,255,255,0.75), transparent 68%);
        right: -100px;
        bottom: -120px;
        animation: pulseBlob 5s ease-in-out infinite;
    }

    .hero-title {
        position: relative;
        z-index: 1;
        font-size: 3.05rem;
        font-weight: 900;
        margin-bottom: 0.75rem;
        color: #1f2937;
        letter-spacing: -1px;
    }

    .hero-subtitle {
        position: relative;
        z-index: 1;
        font-size: 1.12rem;
        color: #374151;
        max-width: 1000px;
        line-height: 1.75;
    }

    .metric-card {
        padding: 1.3rem;
        border-radius: 24px;
        background: rgba(255, 255, 255, 0.72);
        border: 1px solid rgba(255,255,255,0.9);
        box-shadow: 0 14px 32px rgba(148, 163, 184, 0.22);
        transition: transform 0.25s ease, box-shadow 0.25s ease, border 0.25s ease;
        animation: fadeUp 0.85s ease both;
        color: #1f2937;
    }

    .metric-card:hover {
        transform: translateY(-6px) scale(1.015);
        border: 1px solid rgba(251, 191, 36, 0.8);
        box-shadow: 0 18px 42px rgba(251, 191, 36, 0.18);
    }

    .prediction-card {
        position: relative;
        overflow: hidden;
        padding: 1.8rem;
        border-radius: 28px;
        background:
            linear-gradient(135deg, #bbf7d0 0%, #a7f3d0 45%, #bfdbfe 100%);
        border: 1px solid rgba(255,255,255,0.95);
        box-shadow: 0 18px 46px rgba(52, 211, 153, 0.22);
        margin-bottom: 1rem;
        animation: popIn 0.55s ease both, predictionGlow 3s ease-in-out infinite;
    }

    .prediction-card::after {
        content: "✓";
        position: absolute;
        right: 26px;
        top: 18px;
        font-size: 5rem;
        color: rgba(255,255,255,0.55);
        font-weight: 900;
    }

    .prediction-label {
        color: #065f46;
        font-size: 1rem;
        font-weight: 800;
        margin-bottom: 0.45rem;
    }

    .prediction-class {
        color: #064e3b;
        font-size: 2.45rem;
        font-weight: 900;
        margin-bottom: 0.35rem;
    }

    .prediction-confidence {
        color: #065f46;
        font-size: 1.15rem;
        font-weight: 700;
    }

    .section-title {
        font-size: 1.65rem;
        font-weight: 850;
        color: #1f2937;
        margin-top: 1.3rem;
        margin-bottom: 0.9rem;
        animation: fadeUp 0.7s ease both;
    }

    .small-text {
        color: #374151;
        line-height: 1.75;
        padding: 1.1rem 1.2rem;
        border-radius: 20px;
        background: rgba(255,255,255,0.72);
        border: 1px solid rgba(255,255,255,0.92);
        box-shadow: 0 12px 28px rgba(148, 163, 184, 0.18);
    }

    .sidebar-card {
        padding: 1rem;
        border-radius: 20px;
        background: rgba(255,255,255,0.68);
        border: 1px solid rgba(255,255,255,0.9);
        margin-bottom: 1rem;
        transition: transform 0.25s ease, border 0.25s ease, box-shadow 0.25s ease;
        color: #1f2937;
        box-shadow: 0 10px 22px rgba(148, 163, 184, 0.14);
    }

    .sidebar-card:hover {
        transform: translateX(5px);
        border: 1px solid rgba(244, 114, 182, 0.45);
        box-shadow: 0 14px 28px rgba(244, 114, 182, 0.15);
    }

    div[data-testid="stSidebar"] {
        background:
            linear-gradient(180deg, rgba(255, 241, 242, 0.95), rgba(239, 246, 255, 0.95), rgba(240, 253, 244, 0.95));
        border-right: 1px solid rgba(255,255,255,0.8);
    }

    div[data-testid="stSidebar"] * {
        color: #1f2937 !important;
    }

    div[data-testid="stFileUploader"] {
        padding: 1rem;
        border-radius: 24px;
        background: rgba(255,255,255,0.68);
        border: 1px dashed rgba(148, 163, 184, 0.8);
        box-shadow: inset 0 0 24px rgba(255,255,255,0.55), 0 12px 28px rgba(148, 163, 184, 0.15);
        animation: fadeUp 0.9s ease both;
        color: #1f2937;
    }

    div[data-testid="stFileUploader"]:hover {
        border: 1px dashed rgba(251, 146, 60, 0.9);
    }

    div[data-testid="stImage"] img {
        border-radius: 26px;
        box-shadow: 0 20px 46px rgba(148, 163, 184, 0.28);
        border: 1px solid rgba(255,255,255,0.95);
        animation: imageFloat 4s ease-in-out infinite;
    }

    div[data-testid="stDataFrame"] {
        border-radius: 18px;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,0.9);
        box-shadow: 0 12px 26px rgba(148, 163, 184, 0.16);
    }

    .stAlert {
        border-radius: 18px;
    }

    /* Make Streamlit text elements readable on pastel background */
    h1, h2, h3, h4, h5, h6, p, label, span {
        color: #1f2937;
    }

    /* Chart container softness */
    div[data-testid="stVegaLiteChart"] {
        background: rgba(255,255,255,0.60);
        border-radius: 22px;
        padding: 0.8rem;
        box-shadow: 0 12px 28px rgba(148, 163, 184, 0.16);
    }

    @keyframes fadeUp {
        from {
            opacity: 0;
            transform: translateY(18px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    @keyframes popIn {
        from {
            opacity: 0;
            transform: scale(0.96) translateY(10px);
        }
        to {
            opacity: 1;
            transform: scale(1) translateY(0);
        }
    }

    @keyframes heroGlow {
        0%, 100% {
            box-shadow: 0 22px 55px rgba(148, 163, 184, 0.28);
        }
        50% {
            box-shadow: 0 24px 70px rgba(244, 114, 182, 0.22);
        }
    }

    @keyframes predictionGlow {
        0%, 100% {
            box-shadow: 0 18px 46px rgba(52, 211, 153, 0.22);
        }
        50% {
            box-shadow: 0 18px 56px rgba(96, 165, 250, 0.30);
        }
    }

    @keyframes floatFood {
        0%, 100% {
            transform: translateY(0px) rotate(0deg);
        }
        50% {
            transform: translateY(10px) rotate(2deg);
        }
    }

    @keyframes pulseBlob {
        0%, 100% {
            transform: scale(1);
            opacity: 0.55;
        }
        50% {
            transform: scale(1.18);
            opacity: 0.85;
        }
    }

    @keyframes imageFloat {
        0%, 100% {
            transform: translateY(0px);
        }
        50% {
            transform: translateY(-4px);
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Class names
# Make sure this order matches train_data.classes from ImageFolder
# -----------------------------
class_names = ["Burger", "Pasta", "Pizza", "Salad", "Spaghetti", "Sushi"]

# -----------------------------
# Model loading
# -----------------------------
@st.cache_resource
def load_model():
    model = models.efficientnet_b0(weights=None)

    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, len(class_names))

    model_path = "best_model_aug.pth"

    if not os.path.exists(model_path):
        st.error(
            "Model file not found. Please make sure best_model_combined.pth is in the same folder as app.py."
        )
        return None

    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model

model = load_model()

# -----------------------------
# Image preprocessing
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.markdown("## 🍽️ Project Details")

    st.markdown(
        """
        <div class="sidebar-card">
        <b>Backbone</b><br>
        EfficientNet-B0
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="sidebar-card">
        <b>Pretraining</b><br>
        ImageNet
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="sidebar-card">
        <b>Best Configuration</b><br>
        Original Data + Augmentation
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="sidebar-card">
        <b>Classes</b><br>
        Burger, Pasta, Pizza, Salad, Spaghetti, Sushi
        </div>
        """,
        unsafe_allow_html=True
    )

# -----------------------------
# Hero section
# -----------------------------
st.markdown(
    """
    <div class="hero">
        <div class="hero-title">🍽️ Food Image Classification App</div>
        <div class="hero-subtitle">
            Upload a food image and the fine-tuned EfficientNet-B0 model will classify it into one of six categories.
            This demo uses the best-performing model trained with original data and online data augmentation.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Top project metrics
# -----------------------------
m1, m2, m3, m4 = st.columns(4)

with m1:
    st.markdown(
        """
        <div class="metric-card">
        <b>Model</b><br>
        EfficientNet-B0
        </div>
        """,
        unsafe_allow_html=True
    )

with m2:
    st.markdown(
        """
        <div class="metric-card">
        <b>Task</b><br>
        Food Classification
        </div>
        """,
        unsafe_allow_html=True
    )

with m3:
    st.markdown(
        """
        <div class="metric-card">
        <b>Classes</b><br>
        6 Categories
        </div>
        """,
        unsafe_allow_html=True
    )

with m4:
    st.markdown(
        """
        <div class="metric-card">
        <b>Best Test Accuracy</b><br>
        93.33%
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------
# File upload
# -----------------------------
st.markdown('<div class="section-title">Upload Image</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Choose a food image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

# -----------------------------
# Prediction
# -----------------------------
if uploaded_file is not None and model is not None:
    image = Image.open(uploaded_file).convert("RGB")

    left_col, right_col = st.columns([1.05, 1])

    with left_col:
        st.markdown('<div class="section-title">Input Image</div>', unsafe_allow_html=True)
        st.image(image, caption="Uploaded food image", use_container_width=True)

    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        predicted_index = torch.argmax(probabilities).item()
        predicted_class = class_names[predicted_index]
        confidence = probabilities[predicted_index].item() * 100

    prob_df = pd.DataFrame({
        "Class": class_names,
        "Confidence (%)": [round(p.item() * 100, 2) for p in probabilities]
    }).sort_values(by="Confidence (%)", ascending=False)

    with right_col:
        st.markdown('<div class="section-title">Prediction Result</div>', unsafe_allow_html=True)

        st.markdown(
            f"""
            <div class="prediction-card">
                <div class="prediction-label">Predicted Class</div>
                <div class="prediction-class">{predicted_class}</div>
                <div class="prediction-confidence">Confidence: {confidence:.2f}%</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown('<div class="section-title">Confidence Scores</div>', unsafe_allow_html=True)
        st.bar_chart(prob_df.set_index("Class"))
        st.dataframe(prob_df, use_container_width=True, hide_index=True)

else:
    st.info("Upload a food image to see the model prediction and class confidence scores.")

# -----------------------------
# Explanation section
# -----------------------------
st.markdown("---")
st.markdown('<div class="section-title">How This Demo Works</div>', unsafe_allow_html=True)

st.markdown(
    """
    <div class="small-text">
    The uploaded image is resized to 224 × 224, normalized using ImageNet statistics, and passed through a fine-tuned
    EfficientNet-B0 model. The model outputs probability scores for all six food classes, and the class with the
    highest probability is selected as the final prediction.
    </div>
    """,
    unsafe_allow_html=True
)