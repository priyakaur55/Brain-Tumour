import streamlit as st
import numpy as np
from keras.models import load_model
from keras.preprocessing import image

# Load the pre-trained model
model = load_model('./my_model.h5')

# Function to preprocess an image for prediction
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize
    return img_array

# Streamlit app
def main():
    st.title("Brain Tumor Classification App")

    # Upload image through Streamlit
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)

        # Classify the uploaded image
        if st.button("Classify"):
            # Preprocess image
            img_array = preprocess_image(uploaded_file)

            # Make prediction
            prediction = model.predict(img_array)

            # Display prediction
            tumor_types = ["No Tumor", "Glioma Tumor", "Meningioma Tumor", "Pituitary Tumor"]
            predicted_tumor = tumor_types[np.argmax(prediction)]

            st.subheader("Prediction:")
            st.write(f"The predicted tumor type is: {predicted_tumor}")

           
main: main()
