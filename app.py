# app.py
import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os

st.set_page_config(page_title="Indian Currency Note Recognizer", layout="centered")
st.title("Indian Currency Note Recognition (Single-note)")

st.write("Upload an image of a single Indian currency note (₹10, ₹20, ₹50, ₹100, ₹200, ₹500, ₹2000).")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"])

@st.cache(allow_output_mutation=True)
def load_trained_model(path="model.h5"):
    return load_model(path)

CLASS_NAMES = ['10', '20', '50', '100', '200', '500', '2000']

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded image', use_column_width=True)

    img_size = (128,128)
    img = image.resize(img_size)
    img_arr = np.array(img)/255.0
    img_arr = np.expand_dims(img_arr, axis=0)

    if not os.path.exists("model.h5"):
        st.error("Model file model.h5 not found in repo root. Place model.h5 in the repo and redeploy.")
    else:
        model = load_trained_model("model.h5")
        pred = model.predict(img_arr)[0]
        class_idx = int(np.argmax(pred))
        label = CLASS_NAMES[class_idx] if class_idx < len(CLASS_NAMES) else str(class_idx)
        confidence = float(pred[class_idx])
        st.markdown(f"### Predicted: **₹{label}**")
        st.markdown(f"**Confidence:** {confidence*100:.2f}%")
