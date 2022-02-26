import streamlit as st
from PIL import Image
import os

import neural_style_transfer

st.title("Neural Style Transfer")


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
  figure = neural_style_transfer.stylize(content_image, style_image, percent)
  st.write('### Output Image')
  st.pyplot(figure)