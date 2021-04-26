from skimage.io import imread
from skimage.transform import resize
import streamlit as st
import numpy as np
import joblib
model = joblib.load('img_recog_model')
st.title('Infinity War Recognizer')
ip = st.text_input('Enter image URL')
flat_data = []
img1 = imread(ip)
img1_resized = resize(img1,(150,150,3))
flat_data.append(img1_resized.flatten())
flat_data = np.array(flat_data)
op = model.predict(flat_data)
if st.button('Predict'):
  st.title(op[0])
