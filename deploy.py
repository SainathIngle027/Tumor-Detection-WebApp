import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

model = tf.keras.models.load_model('/workspaces/codespaces-blank/tumor_detection_model.h5')

def preprocess_image(img):
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def predict_tumor(image_path):
    img = image.load_img(image_path, target_size=(128, 128))
    processed_image = preprocess_image(img)
    prediction = model.predict(processed_image)
    return prediction[0][0]

def main():
    st.title("Tumor Detection App")
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        image_path = "temp.jpg"
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        
        img = mpimg.imread(image_path)
        st.image(img, caption="Uploaded Image.", use_column_width=True)

        prediction = predict_tumor(image_path)
        prediction_label = "Tumor" if prediction >= 0.5 else "No Tumor"
        st.subheader(f"Prediction: {prediction_label}")

if __name__ == "__main__":
    main()
