# app.py
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os
import gdown

model_path = "waste_classifier_model.h5"
if not os.path.exists(model_path):
    url = "https://drive.google.com/file/d/1ftXwJDsJgFm6suy7R115IeZPTgXyYPCw"  # Ganti FILE_ID
    gdown.download(url, model_path, quiet=False)

# Fungsi untuk memuat model hanya sekali (cache)
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('waste_classifier_model.h5')
    return model

# Muat model
model = load_model()

# Daftar kelas yang dikenali model
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Judul aplikasi
st.title("Aplikasi Klasifikasi Gambar Sampah")
st.write("Upload gambar dan model akan memprediksi jenis sampah.")

# Form upload gambar
uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

# Jika ada gambar yang diunggah
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Gambar yang diunggah", use_column_width=True)

    # Preprocessing
    img = img.resize((150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediksi
    pred = model.predict(img_array)
    predicted_class = class_names[np.argmax(pred)]
    confidence = np.max(pred)

    # Tampilkan hasil prediksi
    st.success(f"Prediksi: {predicted_class} ({confidence*100:.2f}%)")
