import streamlit as st
import numpy as np
from PIL import Image
import os
import tensorflow as tf

# Path ke model
model_path = "waste_classifier_model.tflite"

# Fungsi untuk memuat model TFLite hanya sekali
@st.cache_resource
def load_tflite_model():
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_tflite_model()

# Ambil detail input dan output dari model
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Daftar kelas
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Judul aplikasi
st.title("Aplikasi Klasifikasi Gambar Sampah")
st.write("Upload gambar dan model akan memprediksi jenis sampah.")

# Upload gambar
uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Gambar yang diunggah", use_column_width=True)

    # Preprocessing sesuai dengan input model
    img = img.resize((150, 150))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Inference
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    predicted_class = class_names[np.argmax(output_data)]
    confidence = np.max(output_data)

    st.success(f"Prediksi: {predicted_class} ({confidence * 100:.2f}%)")
