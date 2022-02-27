import streamlit as st
from PIL import Image
import os
import matplotlib.pyplot as plt

import neural_style_transfer

st.markdown("## Neural Style Transfer - Blend Two Images Perfectly!!")


with st.container():
  col1, col3, col2 = st.columns([3, 0.25, 3])

  with col1:
    st.markdown("#### Content Image")
    # Get content image
    content_image = st.file_uploader("Choose content image", type=['png', 'jpg', 
    'jpeg', 'webp', 'jfif', 'pjpeg', 'pjp'])
    if content_image is not None:
      file_details = {"FileName" : content_image.name,
      "FileType" : content_image.type}
      with open(os.path.join("Inputs", content_image.name), "wb") as f: 
          f.write(content_image.getbuffer())  
      # To read file as bytes:
      bytes_data = content_image.getvalue()
      st.image(bytes_data, caption='Content Image')

  with col2:
    st.markdown("#### Style Image")
    # Get style image
    style_image = st.file_uploader("Choose style image", type=['png', 'jpg', 
    'jpeg', 'webp', 'jfif', 'pjpeg', 'pjp'])
    if style_image is not None:
      file_details = {"FileName" : style_image.name,
      "FileType" : style_image.type}
      with open(os.path.join("Inputs", style_image.name), "wb") as f: 
          f.write(style_image.getbuffer()) 
      # To read file as bytes:
      bytes_data = style_image.getvalue()
      st.image(bytes_data, caption='Style Image')

# Percentage of blending
percent = st.slider('How much blending do you want?', 0, 100, 25)

clicked = st.button('Blend')

if clicked:
  results = neural_style_transfer.stylize(content_image, style_image, percent)
  st.markdown('#### Output Image')
  figure, ax = plt.subplots(figsize=(20,10))
  ax.axis('off')
  print("i m here")
  ax.imshow(results)
  st.pyplot(figure)