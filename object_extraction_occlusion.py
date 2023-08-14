#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 12:01:38 2023

@author: hussain
"""

import torch
from PIL import Image
import torchvision.transforms as transforms
import cv2
import numpy as np
from torchvision.utils import save_image
from skimage.filters import gaussian

def resizeImage(height, width, image):
    resize_transform = transforms.Resize((height, width))
    resized_image = resize_transform(image)
    return resized_image

def occlusion(foreground_path, mask_path, background_path, to_tensor):
    foreground = Image.open(foreground_path)
    foreground = resizeImage(512, 512, foreground)
    foreground = to_tensor(foreground)
    
    mask = Image.open(mask_path)
    mask = to_tensor(mask)
    
    masked_foreground = foreground * mask

    background = Image.open(background_path)
    background = resizeImage(512, 512, background)
    background = to_tensor(background)
    cropped_background = background * mask
    background = background - cropped_background
    final_output = background + masked_foreground
    
    return final_output

if __name__ == '__main__':
    to_tensor = transforms.Compose([transforms.PILToTensor()])
    foreground_path = '1.jpg'
    mask_path = '1.png'
    background_path = "2.jpg"
    bg_less_img = 'output.png'
    img = occlusion(foreground_path, mask_path, background_path, to_tensor)
    
    transform = transforms.ToPILImage()
    pil_img = transform(img)
    pil_img.save(bg_less_img)


    
    
    
    