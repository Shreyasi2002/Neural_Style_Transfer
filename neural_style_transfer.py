import torch
from torchvision import models
from torch import optim
from PIL import Image
from torchvision import transforms as T
import numpy as np
import streamlit as st

@st.cache
def stylize(content_image, style_image, percent):
  vgg = models.vgg19(pretrained = True)
  vgg = vgg.features

  for parameters in vgg.parameters():
      parameters.requires_grad_(False)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  vgg.to(device)

  def preprocess(img_path, max_size = 500):
    image = Image.open(img_path).convert('RGB')
    if max(image.size) > max_size:
      size = max_size
    else:
      size = max(image.size)
        
    # Resize the image
    img_transforms = T.Compose([T.Resize(size), 
                                T.ToTensor(), 
                                T.Normalize(mean=[0.485, 0.456, 0.406], 
                                            std=[0.229, 0.224, 0.225])])
    
    image = img_transforms(image)
    image = image.unsqueeze(0)  # (3, 224, 224) -> (1, 3, 224, 224)
    
    return image

  content_pre = preprocess(content_image)
  style_pre = preprocess(style_image)
  content_pre = content_pre.to(device)
  style_pre = style_pre.to(device)

  def deprocess(tensor):
    image = tensor.to('cpu').clone()
    image = image.numpy()
    
    image = image.squeeze(0)  # (1, 3, 224, 224) -> (3, 224, 224)
    image = image.transpose(1,2,0)
    
    # De-normalize image
    image = image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    image.clip(0, 1)
    
    return image

  def get_features(image, model):
    layers = {'0' : 'conv1_1', 
              '5' : 'conv2_1', 
              '10' : 'conv3_1', 
              '19' : 'conv4_1', 
              '21' : 'conv4_2', 
              '28' : 'conv5_1'}
    
    x = image
    features = {}
    
    for name, layer in model._modules.items():
      x = layer(x)
      if name in layers:
        features[layers[name]] = x
    
    return features

  content_f = get_features(content_pre, vgg)
  style_f = get_features(style_pre, vgg)

  def gram_matrix(tensor):
    b, c, h, w = tensor.size()
    tensor = tensor.view(c, h * w)
    
    gram = torch.mm(tensor, tensor.t())
    return gram

  style_grams = {layer : gram_matrix(style_f[layer]) for layer in style_f}

  def content_loss(target_conv4_2, content_conv4_2):
      loss = torch.mean((target_conv4_2 - content_conv4_2) ** 2)
      return loss

  style_weights = {'conv1_1' : 1.0, 
                  'conv2_1' : 0.75, 
                  'conv3_1' : 0.2, 
                  'conv4_1' : 0.2, 
                  'conv5_1' : 0.2}

  def style_loss(style_weights, target_features, style_grams):
    loss = 0
    
    for layer in style_weights:
      target_f = target_features[layer]
      target_gram = gram_matrix(target_f)
      style_gram = style_grams[layer]
      b, c, h, w = target_f.shape
      
      layer_loss = style_weights[layer] * torch.mean((target_gram - style_gram) ** 2)
      loss += layer_loss / (c * h * w)
        
    return loss

  target = content_pre.clone().requires_grad_(True).to(device)
  target_f = get_features(target, vgg)

  optimizer = optim.Adam([target], lr = 0.003)
  alpha = 1
  beta = 1e4

  epochs = 1001

  def total_loss(c_loss, s_loss, alpha, beta):
    loss = alpha * c_loss + beta * s_loss
    return loss

  for i in range(epochs):
    target_f = get_features(target, vgg)
    c_loss = content_loss(target_f['conv4_2'], content_f['conv4_2'])
    s_loss = style_loss(style_weights, target_f, style_grams)
    t_loss = total_loss(c_loss, s_loss, alpha, beta)
    
    optimizer.zero_grad()
    t_loss.backward()
    optimizer.step()
    print(i)
    if i == percent * 10:
      results = deprocess(target.detach())
      break

  print("i am outside")
  print(results.shape)
  print(results)
  return results