## Import necessary libraries ##
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

## Define functions ##
def set_up_page():
    """ Set up the header for the page"""
    st.header("Computer Vision: X-Ray Classifier")
    st.info("""This program uses Computer Vision and Convolutional Neural Networks (CNN) to predict is there is a fracture in the X-Ray image.
            \n The Machine Learning model was built using Tensorflow and trained with 3000 X-ray images.\n """)

def load_model():
    """
    This function loads the keras model that will be used to predicts the fractures

    Returns:
        - model: keras CNN model
    """
    
    model = tf.keras.models.load_model('app/pages/provider/best_model-0.95.keras')

    return model

def preprocess_image(uploaded_img):
    """
    Preprocesses the image to make it compatible with the model.
    - Reads the image directly from uploaded bytes.
    - Resizes the image to (100, 100).
    - Normalizes pixel values to the range [0, 1].

    Args:
        uploaded_img (UploadedFile): The uploaded image file from Streamlit.

    Returns:
        tf.Tensor: Preprocessed image tensor.
    """
    # Decode the uploaded image
    image = Image.open(uploaded_img).convert("RGB")

    # Resize to the model's input shape
    image = image.resize((100, 100))

    # Convert to numpy array and normalize to [0, 1]
    image_array = np.array(image) / 255.0

    # Expand dimensions to match the model's expected input shape
    image_tensor = tf.expand_dims(image_array, axis=0)

    return image_tensor

def predict_fracture(preprocessed_image, model):
    """
    Predicts whether the image represents a fractured or non-fractured bone.

    Args:
        preprocessed_image (tf.Tensor): The preprocessed image tensor.

    Returns:
        str: 'Fractured' or 'Not Fractured' based on the model's prediction.
    """
    # Make prediction
    prediction = model.predict(preprocessed_image)

    # Interpret the result
    if prediction[0] > 0.75:
        return f"Fractured, {int(prediction[0]*100)} %" 
    elif prediction[0] < 0.25:
        return f"Not Fractured, {int(prediction[0]*100)} %"
    else: 
        return f"Inconclusive diagnosis with {int(prediction[0]*100)} % possibility of fracture... \n Please check with Radiologist"
def main():

    model = load_model()
    set_up_page()

    # File uploader for images
    uploaded_img = st.file_uploader("Upload an X-ray image", type=["jpg", "jpeg", "png"])
    if uploaded_img is not None:
        try:
            # Preprocess the uploaded image
            preprocessed_image = preprocess_image(uploaded_img)

            # Make a prediction using the model
            result = predict_fracture(preprocessed_image, model)

            # Display the prediction result
            st.markdown(f"##### Prediction: {result}")

            # Display the uploaded image
            st.image(uploaded_img, caption=f"Prediction: {result}", use_container_width=False, width=500)

        except Exception as e:
            st.error(f"An error occurred: {e}")

## Initialize the app ##
main()
