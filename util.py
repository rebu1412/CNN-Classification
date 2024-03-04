import base64

import streamlit as st
from PIL import ImageOps, Image
import numpy as np

def set_background(image_file):
    with open(image_file, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{b64_encoded});
            background-size: cover;
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)    



def classify(image, model, class_names):
    resized_image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
    image_array = np.asarray(resized_image)
    normalized_image_array = image_array.astype(np.float32) / 127.5 - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array


    prediction = model.predict(data)
    # index = np.argmax(prediction)
    index = 0 if prediction[0][0] >= 0.95 else 1
    class_name = class_names[index]
    confidence_score = prediction[0][0]


    return class_name, confidence_score