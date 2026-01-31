import streamlit as st
from tensorflow import keras
import numpy as np
from PIL import Image

# Minimal upload-only frontend
st.set_page_config(page_title="MNIST Upload", page_icon="🔢")


@st.cache_resource
def load_model():
    try:
        return keras.models.load_model("handWritten.keras")
    except Exception:
        return None


model = load_model()

st.title("MNIST Digit Recognizer — Upload Only")

if model is None:
    st.error("Model not found. Ensure 'handWritten.keras' is in the app directory.")
else:
    uploaded_file = st.file_uploader("Upload an image (jpg, png, bmp)", type=["jpg", "jpeg", "png", "bmp", "gif"]) 

    if uploaded_file is not None:
        # Display original
        img_display = Image.open(uploaded_file).convert("RGB")
        st.image(img_display, caption="Uploaded Image", use_column_width=True)

        # Preprocess: grayscale -> 28x28 -> normalize
        img_gray = Image.open(uploaded_file).convert("L")
        try:
            resample = Image.Resampling.LANCZOS
        except AttributeError:
            resample = Image.LANCZOS
        img_resized = img_gray.resize((28, 28), resample)
        img_array = np.array(img_resized) / 255.0
        img_input = img_array.reshape(1, 28, 28, 1)

        # Predict
        prediction = model.predict(img_input, verbose=0)
        predicted_digit = int(np.argmax(prediction))
        confidence = float(prediction[0][predicted_digit]) * 100.0

        st.subheader("Prediction")
        st.write(f"Predicted digit: **{predicted_digit}**")
        st.write(f"Confidence: **{confidence:.2f}%**")

        st.image(img_resized, caption="Processed (28x28)", width=150)
