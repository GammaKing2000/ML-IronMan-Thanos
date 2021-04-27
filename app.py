from skimage.io import imread
from skimage.transform import resize
import streamlit as st
import numpy as np
import joblib
model = joblib.load('img_recog_model')
st.title('Infinity War Recognizer')
st.text('Enter image URL of Ironman or Thanos and the model will predict what it is')
st.text('Preferably use pintrest for images')
ip = st.text_input('Enter image URL')
flat_data = []
if st.button('Predict'):
  img1 = imread(ip)
  img1_resized = resize(img1,(150,150,3))
  flat_data.append(img1_resized.flatten())
  flat_data = np.array(flat_data)
  op = model.predict(flat_data)
  st.title(op[0]) 
  st.image(img1)
