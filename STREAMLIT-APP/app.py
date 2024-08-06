import os
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# Load the pre-trained model
model_path = os.path.join(os.path.dirname(__file__), 'models', 'saved_model_256_9010.h5')
model = tf.keras.models.load_model(model_path)

# Mapping index to class labels
class_labels = ['Bacterialblight', 'Blast', 'Brownspot', 'Tungro', 'Unknown']

# Streamlit app
def main():
    st.write("""
        # Identifikasi Penyakit pada Tanaman Padi menggunakan Metode Convolutional Neural Network
        *Web App* ini dibuat sebagai syarat dalam menyelesaikan Program Studi Strata Satu (S-1)
        Program Studi Teknik Informatika. Web App ini dapat mengidentifikasi 4 jenis penyakit pada
        tanaman padi yaitu *Bacterial Blight*, *Blast*, *Brown Spot*, dan *Tungro*. Pengguna dapat 
        mengunggah gambar penyakit tanaman padi yang ingin diidentifikasi dengan format .jpg, .jpeg,
        dan .png dengan ukuran maksimal 200MB.
        """
    )

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", use_column_width=False)

        # Button for prediction
        if st.button("Predict"):
            # Preprocess the image
            img = tf.keras.utils.load_img(uploaded_file, target_size=(150, 150))
            x = tf.keras.utils.img_to_array(img)
            x /= 255
            x = np.expand_dims(x, axis=0)
            images = np.vstack([x])

            # Make predictions
            classes = model.predict(images, batch_size=10)

            # Assuming model output shape is (1, 5) for the 5 classes
            predicted_label = class_labels[np.argmax(classes[0])]
            predicted_proba = classes[0][np.argmax(classes[0])]

            st.write("## Prediction:")
            st.write(f"The image is classified as: {predicted_label} with confidence: {predicted_proba:.2f}")

            st.write("## Prediction Probabilities:")
            for label, proba in zip(class_labels, classes[0]):
                st.write(f"{label}: {proba:.2f}")

if __name__=='__main__':
    main()