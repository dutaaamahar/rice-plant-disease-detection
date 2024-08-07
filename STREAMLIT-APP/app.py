"""
Aplikasi Web Identifikasi Penyakit Tanaman Padi

Modul ini mengimplementasikan aplikasi web untuk mengidentifikasi penyakit pada
tanaman padi menggunakan model Convolutional Neural Network (CNN). Aplikasi ini
memungkinkan pengguna untuk mengunggah gambar penyakit tanaman padi, memproses gambar,
dan memprediksi jenis penyakit. Jika gambar yang diunggah tidak termasuk dalam kategori 
penyakit yang dikenal, maka hasil prediksi akan menunjukkan bahwa gambar tersebut tidak 
mengandung penyakit tanaman padi.
"""

import os
import numpy as np
import tensorflow as tf
from PIL import Image
import streamlit as st

# Memuat model yang telah dilatih sebelumnya
model_path = os.path.join(os.path.dirname(__file__), 'models', 'saved_model_256_9010.h5')
model = tf.keras.models.load_model(model_path)

# Mapping indeks ke label kelas
class_labels = ['Bacterialblight', 'Blast', 'Brownspot', 'Tungro', 'Unknown']

def main():
    """
    Fungsi utama untuk menjalankan aplikasi web Streamlit.

    Fungsi ini mengatur antarmuka Streamlit, termasuk judul, deskripsi, uploader file
    untuk mengunggah gambar, dan tombol prediksi. Fungsi ini menangani pengunggahan gambar,
    menampilkan gambar, pra-pemrosesan, dan prediksi.
    """
    st.write("""
        # Identifikasi Penyakit pada Tanaman Padi menggunakan Metode Convolutional Neural Network
        *Web App* ini dibuat sebagai syarat dalam menyelesaikan Program Studi Strata Satu (S-1)
        Program Studi Teknik Informatika. Web App ini dapat mengidentifikasi 4 jenis penyakit pada
        tanaman padi yaitu *Bacterial Blight*, *Blast*, *Brown Spot*, dan *Tungro*, objek diluar keempat
        penyakit akan diprediksi sebagai *Unknown*. Pengguna dapat mengunggah gambar penyakit tanaman
        padi yang ingin diidentifikasi dengan format .jpg, .jpeg, dan .png dengan ukuran maksimal 200MB.
        """
    )

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Menampilkan gambar yang diunggah
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", use_column_width=False)

        if st.button("Predict"):
            # Pra-pemrosesan gambar
            img = tf.keras.utils.load_img(uploaded_file, target_size=(150, 150))
            x = tf.keras.utils.img_to_array(img)
            x /= 255.0
            x = np.expand_dims(x, axis=0)
            images = np.vstack([x])

            # Melakukan prediksi
            classes = model.predict(images, batch_size=10)

            # Mengasumsikan bentuk output model adalah (1, 5) untuk 5 kelas
            predicted_label = class_labels[np.argmax(classes[0])]
            predicted_proba = classes[0][np.argmax(classes[0])]

            st.write("## Prediction:")
            if predicted_label == 'Unknown':
                st.write("This image does not include rice plant diseases.")
            else:
                st.write(f"The image is classified as: {predicted_label} with confidence: "
                         f"{predicted_proba:.2f}")

            st.write("## Prediction Probabilities:")
            for label, proba in zip(class_labels[:-1], classes[0][:-1]):
                st.write(f"{label}: {proba:.2f}")

if __name__ == '__main__':
    main()
