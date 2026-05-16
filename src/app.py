import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Page config
st.set_page_config(
    page_title="Breast Cancer Detection",
    page_icon="🔬",
    layout="centered"
)

# Title
st.title("🔬 Breast Cancer Detection CNN")
st.write("Upload a histopathology image to classify it as **Benign** or **Malignant**.")

# Load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('models/best_model.keras')
    return model

model = load_model()

# Image upload
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = image.resize((96, 96))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    probability = float(prediction[0][0])

    # Display result
    st.subheader("Prediction Result:")
    if probability > 0.5:
        st.error(f"🔴 Malignant (confidence: {probability:.2%})")
    else:
        st.success(f"🟢 Benign (confidence: {1 - probability:.2%})")

    # Disclaimer
    st.warning("⚠️ This is a portfolio demo only. Not for clinical use.")