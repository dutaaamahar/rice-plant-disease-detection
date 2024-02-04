import os
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# Load the pre-trained model
model_path = os.path.join(os.path.dirname(__file__), 'models', 'saved_model.h5')
model = tf.keras.models.load_model(model_path)

# Mapping index to class labels
class_labels = ['Bacterialblight', 'Blast', 'Brownspot', 'Tungro']

# Streamlit app
def main():
    st.write("""
        # Identifikasi Penyakit pada Tanaman Padi menggunakan Metode Convolutional Neural Network
        Tugas Akhir ini dibuat sebagai syarat dalam menyelesaikan Program Studi Strata Satu (S-1) Program Studi Teknik Informatika
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
            img = tf.keras.utils.load_img(uploaded_file, target_size=(300, 300))
            x = tf.keras.utils.img_to_array(img)
            x /= 255
            x = np.expand_dims(x, axis=0)
            images = np.vstack([x])

            # Make predictions
            classes = model.predict(images, batch_size=10)

            # Assuming model output shape is (1, 4) for the 4 classes
            predicted_label = class_labels[np.argmax(classes[0])]

            st.write("## Prediction:")
            st.write(f"The image is classified as: {predicted_label} with confidence: {classes[0][np.argmax(classes[0])]}")

if __name__=='__main__':
    main()