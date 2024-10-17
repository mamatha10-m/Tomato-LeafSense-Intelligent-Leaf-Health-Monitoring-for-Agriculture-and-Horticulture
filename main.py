import os
import keras
from keras.models import load_model
import streamlit as st 
import tensorflow as tf
import numpy as np


st.header('Tomato leaf disease Classification ')
flower_names = ['Tomato___Bacterial_spot',
 'Tomato___Early_blight',
 'Tomato___Late_blight',
 'Tomato___Leaf_Mold',
 'Tomato___Septoria_leaf_spot',
 'Tomato___Spider_mites Two-spotted_spider_mite',
 'Tomato___Target_Spot',
 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
 'Tomato___Tomato_mosaic_virus',
 'Tomato___healthy']

model = load_model('densenet121_custom_model (1).keras')

def classify_images(image_path):
    input_image = tf.keras.utils.load_img(image_path, target_size=(256, 256))
    input_image_array = tf.keras.utils.img_to_array(input_image)
    input_image_exp_dim = tf.expand_dims(input_image_array,0)

    predictions = model.predict(input_image_exp_dim)
    result = tf.nn.softmax(predictions[0])
    max_result_percentage = np.max(result) * 100

    # Convert to an integer after multiplying by 100
    if int(max_result_percentage) < 40:
        outcome = 'Not detected'
    else: 
        outcome = 'The Image belongs to ' + flower_names[np.argmax(result)] + ' with a score of '+ str(np.max(result)*100)
    return outcome

uploaded_file = st.file_uploader('Upload an Image')
if uploaded_file is not None:
   with open(os.path.join('upload', uploaded_file.name), 'wb') as f:
       f.write(uploaded_file.getbuffer())
       st.image(uploaded_file, width = 200)

       st.markdown(classify_images(uploaded_file))
    
  

uploaded_fstreamlit run c:/pppp/main.pyile = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    upload_dir = 'upload'
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)
    
    save_path = os.path.join(upload_dir, uploaded_file.name)
    with open(save_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    
    st.write("File saved at:", save_path)
