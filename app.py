import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load model
model = load_model("cifar10_model.h5")

cifar10_labels = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

st.title("🧠 CIFAR-10 Image Classifier")
st.write("Upload an image("airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"), let the model predict!")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img = image.load_img(uploaded_file, target_size=(32, 32))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    pred = model.predict(img_array)
    class_idx = np.argmax(pred)
    confidence = np.max(pred)

    st.image(img, caption="Uploaded Image", width=200)
    st.success(f"Prediction: **{cifar10_labels[class_idx]}**")
    st.info(f"Confidence: **{confidence:.2f}**")

