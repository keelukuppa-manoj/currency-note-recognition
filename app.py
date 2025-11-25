# app.py
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import os

# Streamlit page config
st.set_page_config(page_title="Indian Currency Note Recognizer", layout="centered")
st.title("Indian Currency Note Recognition (Single-note)")
st.write("Upload an image of a single Indian currency note (₹10, ₹20, ₹50, ₹100, ₹200, ₹500, ₹2000).")

# Path to TFLite model in repo root
TFLITE_MODEL_PATH = "model.tflite"  # Make sure model.tflite is in repo root

CLASS_NAMES = ['Tennote', 'Twentynote', 'Fiftynote', '1Hundrednote', 
               '2Hundrednote', '5Hundrednote', '2Thousandnote']

# Load TFLite model
@st.cache(allow_output_mutation=True)
def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Load and display uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Resize and normalize image
    img_size = (128, 128)
    img = image.resize(img_size)
    img_arr = np.array(img, dtype=np.float32) / 255.0
    img_arr = np.expand_dims(img_arr, axis=0)  # Add batch dimension

    if not os.path.exists(TFLITE_MODEL_PATH):
        st.error(f"Model file not found: {TFLITE_MODEL_PATH}. Please place it in the repo root and redeploy.")
    else:
        interpreter = load_tflite_model(TFLITE_MODEL_PATH)
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Ensure input dtype matches TFLite model
        input_dtype = input_details[0]['dtype']
        img_arr = img_arr.astype(input_dtype)

        # Run inference
        interpreter.set_tensor(input_details[0]['index'], img_arr)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])[0]

        # Log raw output
        st.write(f"Raw model output: {output_data}")

        # Get predicted class
        class_idx = int(np.argmax(output_data))
        label = CLASS_NAMES[class_idx]
        confidence = float(output_data[class_idx])

        st.markdown(f"### Predicted: **₹{label.replace('note','')}**")
        st.markdown(f"**Confidence:** {confidence*100:.2f}%")
