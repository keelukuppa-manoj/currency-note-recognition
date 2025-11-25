# app.py
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import os

st.set_page_config(page_title="Indian Currency Note Recognizer", layout="centered")
st.title("Indian Currency Note Recognition (Single-note)")

st.write("Upload an image of a single Indian currency note (₹10, ₹20, ₹50, ₹100, ₹200, ₹500, ₹2000).")

CLASS_NAMES = ['10', '20', '50', '100', '200', '500', '2000']

# Load TFLite model
@st.cache(allow_output_mutation=True)
def load_trained_model(path="model.tflite"):
    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    return interpreter

# Ensure model file exists
if not os.path.exists("model.tflite"):
    st.error("Model file model.tflite not found in repo root. Place model.tflite in the repo and redeploy.")
else:
    interpreter = load_trained_model("model.tflite")
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']  # e.g., [1, 224, 224, 3]
    input_dtype = input_details[0]['dtype']

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"])

# Optional: test with your uploaded file from history
test_image_path = "/mnt/data/909ead23-8c4b-402e-aaca-40def6e66ebe.png"
use_test_image = st.checkbox("Use sample test image")

if uploaded_file or use_test_image:
    if use_test_image:
        image = Image.open(test_image_path).convert("RGB")
    else:
        image = Image.open(uploaded_file).convert("RGB")
    
    st.image(image, caption='Input image', use_column_width=True)

    # Resize to model input
    img_resized = image.resize((input_shape[1], input_shape[2]))
    img_arr = np.expand_dims(np.array(img_resized, dtype=np.float32)/255.0, axis=0)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], img_arr)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]

    class_idx = int(np.argmax(output_data))
    label = CLASS_NAMES[class_idx] if class_idx < len(CLASS_NAMES) else str(class_idx)
    confidence = float(output_data[class_idx])

    st.markdown(f"### Predicted: **₹{label}**")
    st.markdown(f"**Confidence:** {confidence*100:.2f}%")
