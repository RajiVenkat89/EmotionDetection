import streamlit as st
from tensorflow.keras.models import load_model  
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image

st.set_page_config(page_title='EMOTION DETECTION',layout="wide")

st.markdown(f'<h1 style= "text-align:center;size:24px;color:blue;">EMOTION DETECTION</h1>',unsafe_allow_html=True)
st.header(":blue[**Welcome!!!**]")
#Load Model 
model=load_model('C:/Users/Dharmarajan/Documents/Guvi/Project/Final Project/Emotion Detection/emotion_cnn_new.h5',compile=False)

np.random.seed(42)
tf.random.set_seed(42)
#Provision to Upload File for Prediction
uploaded_file=st.file_uploader("Upload an Image",type=["jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    #image = image.convert("RGB")
    resize_image= image.resize((224,224))
    st.image(resize_image, caption='Resized Image (224x224).', use_column_width=False)
    select=st.button(":blue[**PREDICT EMOTION**]")
    if select:      
        image_array = img_to_array(resize_image)  # Convert the image to a numpy array
        image_array = image_array / 255.0  # Rescale the image
        image_array = tf.expand_dims(image_array, axis=0) #Adding one dimension for batch size
        predictions = model.predict(image_array)
        #st.write(predictions)
        #st.write(predictions.shape)
        if predictions.size>0:
            predicted_class_index =int(np.argmax(predictions))
            class_names = ['Angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']  # Replace with your actual class names
            predicted_class = class_names[predicted_class_index]
            st.write(f":green[The predicted class is: {predicted_class}]")
        else:
            st.write("No predictions were made. Please try again.")