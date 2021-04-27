from skimage.io import imread, imshow
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
  imshow(ip)
# numpy==1.19.5
#joblib==1.0.1
#scikit_image==0.16.2
#skimage==0.0
#streamlit==0.80.0

