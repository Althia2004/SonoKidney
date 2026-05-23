import json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

MODEL_PATH = "models/best_sonokidney_model.keras"
CLASS_INDICES_PATH = "models/class_indices.json"
IMG_SIZE = (224, 224)

st.set_page_config(
    page_title="SonoKidney AI",
    page_icon="🩺",
    layout="centered"
)

st.markdown("""
<style>
.main {
    background-color: #f7f9fc;
}
.title {
    text-align: center;
    color: #1f4e79;
}
.result-box {
    padding: 20px;
    border-radius: 12px;
    background-color: #eef6ff;
    border-left: 6px solid #1f77b4;
    margin-top: 20px;
}
.warning-box {
    padding: 15px;
    border-radius: 10px;
    background-color: #fff3cd;
    border-left: 6px solid #ffc107;
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_data
def load_class_names():
    with open(CLASS_INDICES_PATH, "r") as f:
        class_indices = json.load(f)

    index_to_class = {v: k for k, v in class_indices.items()}
    return index_to_class

def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def get_description(prediction):
    descriptions = {
        "normal": "The model predicts that the kidney ultrasound image appears normal.",
        "cyst": "The model predicts signs consistent with a kidney cyst.",
        "stone": "The model predicts signs consistent with a kidney stone.",
        "tumor": "The model predicts signs consistent with a kidney tumor."
    }
    return descriptions.get(prediction.lower(), "No description available.")

st.markdown("<h1 class='title'>SonoKidney AI</h1>", unsafe_allow_html=True)
st.markdown("### Kidney Ultrasound Image Classification")
st.write("Upload a kidney ultrasound image and the model will classify it as **Normal, Cyst, Stone, or Tumor**.")

try:
    model = load_model()
    index_to_class = load_class_names()
except Exception as e:
    st.error("App failed while loading the model or class indices.")
    st.exception(e)
    st.stop()

uploaded_file = st.file_uploader(
    "Upload ultrasound image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Ultrasound Image", use_container_width=True)

    if st.button("Analyze Image"):
        processed_image = preprocess_image(image)

        predictions = model.predict(processed_image)[0]
        predicted_index = int(np.argmax(predictions))
        confidence = min(float(predictions[predicted_index]), 0.999)
        predicted_class = index_to_class[predicted_index]

        st.markdown(
            f"""
            <div class="result-box">
                <h3>Prediction Result</h3>
                <p><b>Predicted Class:</b> {predicted_class.upper()}</p>
                <p><b>Confidence:</b> {confidence * 100:.1f}%</p>
                <p>{get_description(predicted_class)}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.subheader("Class Confidence Scores")

        for idx, score in enumerate(predictions):
            class_name = index_to_class[idx]
            st.write(f"**{class_name.upper()}**: {score * 100:.2f}%")
            st.progress(float(score))

        st.markdown(
            """
            <div class="warning-box">
                <b>Disclaimer:</b> This system is for academic and research purposes only.
                It is not a replacement for professional medical diagnosis.
            </div>
            """,
            unsafe_allow_html=True
        )
else:
    st.info("Please upload an ultrasound image to begin.")