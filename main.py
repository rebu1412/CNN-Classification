import streamlit as st
import tensorflow as tf
from tensorflow import keras
from PIL import Image
from util import classify, set_background


set_background('doctor3.jpg')


# set title
st.title("Pneumonia classification")

# set header
st.header("Please upload a chest X-ray image")

# #upload file
file = st.file_uploader('', type=['png', 'jpg', 'jpeg'])

# #load model
model = tf.keras.models.load_model('pneumonia.h5')

#load class names
with open('chest_xray/labels.txt', 'r') as f:
    class_names = [a[:-1].split(' ')[1] for a in f.readlines()]
    f.close()

# display image
if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image, use_column_width=True)

# classify image
    class_name, conf_score = classify(image, model, class_names)

    st.write("## {}".format(class_name))
    st.write("### score: {}%".format(int(conf_score * 1000) / 10))
