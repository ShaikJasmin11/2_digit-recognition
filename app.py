# app.py

import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps

st.set_page_config(page_title="Digit Recognizer", page_icon="✍️")

st.title("✍️ Handwritten Digit Recognizer")
st.write("Draw or upload a digit image (28x28 grayscale) to predict it!")

model = load_model("models/cnn_model.h5")

uploaded_file = st.file_uploader("Upload a digit image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L')
    st.image(image, caption="Uploaded Image", use_container_width=True)

    image = ImageOps.invert(image)
    image = image.resize((28, 28))
    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    if st.button("Predict"):
        prediction = model.predict(img_array)
        pred_class = np.argmax(prediction)
        st.success(f"🎯 Predicted Digit: **{pred_class}**")
