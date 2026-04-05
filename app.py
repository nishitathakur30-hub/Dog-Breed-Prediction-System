"""
Dog Breed Prediction - Streamlit Web App
==========================================
Run: streamlit run app.py

Author: Nishita Thakur
"""

import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
import io
import os

# ── Config ────────────────────────────────────
IMG_SIZE    = 224
MODEL_PATH  = "dog_breed_model.h5"
LABELS_PATH = "class_labels.npy"

st.set_page_config(
    page_title="Dog Breed Predictor 🐾",
    page_icon="🐶",
    layout="centered"
)

# ── Load Model (cached) ───────────────────────
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file '{MODEL_PATH}' not found. Please train the model first.")
        st.stop()
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

@st.cache_resource
def load_labels():
    if not os.path.exists(LABELS_PATH):
        st.error(f"Labels file '{LABELS_PATH}' not found. Please train the model first.")
        st.stop()
    return np.load(LABELS_PATH, allow_pickle=True).item()

# ── Preprocessing ──────────────────────────────
def preprocess(pil_image):
    img = np.array(pil_image.convert("RGB"))
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)

# ── UI ─────────────────────────────────────────
st.title("🐾 Dog Breed Prediction System")
st.markdown(
    "Upload a dog photo and the model will predict the **top 3 most likely breeds** "
    "using a CNN trained with TensorFlow & MobileNetV2."
)
st.divider()

uploaded_file = st.file_uploader("📸 Upload a Dog Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True)

    with col2:
        with st.spinner("Analyzing breed..."):
            model       = load_model()
            idx_to_class = load_labels()
            tensor      = preprocess(image)
            preds       = model.predict(tensor)[0]
            top_indices = np.argsort(preds)[::-1][:3]
            results     = [
                (idx_to_class[i].replace("_", " ").title(), float(preds[i]) * 100)
                for i in top_indices
            ]

        st.subheader("🏆 Prediction Results")
        medals = ["🥇", "🥈", "🥉"]
        for medal, (breed, conf) in zip(medals, results):
            st.metric(label=f"{medal}  {breed}", value=f"{conf:.2f}%")
            st.progress(conf / 100)

st.divider()
st.markdown(
    "<small>Built with TensorFlow · MobileNetV2 · OpenCV · Streamlit &nbsp;|&nbsp; "
    "By Nishita Thakur</small>",
    unsafe_allow_html=True
)
