#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 21:25:06 2023

@author: hussain
"""

import torch
from PIL import Image
import torchvision.transforms as transforms
import argparse
import os
import csv
import random

parser = argparse.ArgumentParser('Object extraction and occlusion with other images', add_help=False)
parser.add_argument('--foreground_image_list_path', default='../1_attention/data/train/Images/', type=str)
parser.add_argument('--foreground_mask_path', default='../1_attention/results/attention_V2/pgt-psa-rw/', type=str)
parser.add_argument('--background_image_list_path', default='../1_attention/data/train/Images/', type=str)
parser.add_argument('--image_size', default=512, type=int)
parser.add_argument('--output_dir', default='./occlusion/', type=str)
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

def resizeImage(height, width, image):
    resize_transform = transforms.Resize((height, width))
    resized_image = resize_transform(image)
    return resized_image
    
def padResize(height, width, image):
    height_diff = (height - image.shape[1])/2
    width_diff = (width - image.shape[2])/2
    if width_diff > 0:
        left = width_diff if (int(width_diff)*2)+image.shape[2] > width else width_diff+0.5
        right = width_diff
    else:
        left = width_diff if (int(width_diff)*2)+image.shape[2] == width else width_diff-0.5
        right = width_diff
    if height_diff > 0:
        top = height_diff if (int(height_diff)*2)+image.shape[1] > height else height_diff+0.5
        bottom = height_diff
    else:
        top = height_diff if (int(height_diff)*2)+image.shape[1] == height else height_diff-0.5
        bottom = height_diff
        
    padding = (int(left), int(top), int(right), int(bottom))
    resized_img = transforms.functional.pad(image, padding)
    return resized_img

def occlusion(foreground_path, mask_path, background_path, to_tensor):
    foreground = Image.open(foreground_path)
    # foreground = resizeImage(args.image_size, args.image_size, foreground)
    foreground = to_tensor(foreground)
    
    mask = Image.open(mask_path)
    mask = resizeImage(foreground.shape[1], foreground.shape[2], mask)
    mask = to_tensor(mask)
    
    mask -= torch.min(mask)
    mask /= torch.max(mask)
    
    masked_foreground = foreground * mask
    
    mask = padResize(args.image_size, args.image_size, mask)
    masked_foreground = padResize(args.image_size, args.image_size, masked_foreground)

    background = Image.open(background_path)
    background = resizeImage(args.image_size, args.image_size, background)
    background = to_tensor(background)
    cropped_background = background * mask
    background = background - cropped_background
    final_output = background + masked_foreground
    
    return final_output


if __name__ == '__main__':
    to_pil = transforms.ToPILImage()
    # to_tensor = transforms.Compose([transforms.PILToTensor()])
    to_tensor = transforms.Compose([transforms.ToTensor()])
    
    foreground_img_list, foreground_mask_list, background_img_list, background_label_list = [], [], [], []
    with open(os.path.join(args.foreground_image_list_path.strip('/Images'), 'labels.csv'), 'r') as file:
        file_rows = csv.reader(file)
        for i, row in enumerate(file_rows):
            if i > 0:
                foreground_img_list.append(os.path.join(args.foreground_image_list_path, row[1], row[0]))
                foreground_mask_list.append(os.path.join(args.foreground_mask_path, row[0].strip('.jpg')+'.png'))
                background_img_list.append(os.path.join(args.background_image_list_path, row[1], row[0]))
                background_label_list.append(row[1])
                
    foreground_indices_list, background_indices_list = [], []
    for i in range(len(foreground_img_list)):
        foreground_indices_list.append(i)
        background_indices_list.append(i)
    random.shuffle(foreground_indices_list)
    random.shuffle(background_indices_list)                
                
    for i in range(len(foreground_img_list)):
        print('iter: ', i)
        foreground_index = foreground_indices_list[i]
        background_index = background_indices_list[i]
        
        foreground_path = foreground_img_list[foreground_index]
        mask_path = foreground_mask_list[foreground_index]
        background_path = background_img_list[background_index]
        
        occluded_image = occlusion(foreground_path, mask_path, background_path, to_tensor)
        
        transform = transforms.ToPILImage()
        pil_img = transform(occluded_image)
        
        img_name = background_path.split('/')
        img_name = img_name[6].strip('.jpg')+'.png'
        label = img_name.split('_')[1]
        pil_img.save(os.path.join(args.output_dir, img_name+'.png'))
