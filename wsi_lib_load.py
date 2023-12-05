# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 16:23:00 2023

@author: Hussain Ahmad Madni
"""

import openslide as ops
import time
import numpy as np
import slideio
import matplotlib.pyplot as plt
# from cucim import CuImage
import cv2
import os

def wsiOpenSlid(wsi_full_path, downsample): 
    time_0 = time.time()
    with ops.OpenSlide(wsi_full_path) as slide:
        # level = slide.get_best_level_for_downsample(downsample)   # Return the best level for displaying the given downsample.
        level = 0 # for high resolution
        level_size = slide.level_dimensions[level]       # A list of (pixels_x, pixels_y) tuples for each Deep Zoom level
        im = slide.read_region((0,0), level, level_size)    # Return an RGBA Image containing the contents of the specified region
    imarray = np.array(im)[..., :3]
    time_1 = time.time()    
    print('openslide shape: ', imarray.shape)
    total_time = time_1 - time_0
    return total_time

def wsiSlideio(wsi_full_path):
    time_0 = time.time()
    # slide = slideio.open_slide(wsi_full_path,driver_id="GDAL")
    slide = slideio.open_slide(wsi_full_path, "GDAL")
    scene = slide.get_scene(0)
    # image = scene.read_block(size=(500,0)) # example to read a block
    image = scene.read_block()   # read whole image
    time_1 = time.time()
    print('slideio shape: ', image.shape)
    total_time = time_1 - time_0    
    # plt.imshow(image)
    return total_time    

def wsiCucim(wsi_full_path):
    time_0 = time.time()
    img = CuImage(wsi_full_path)
    count = img.resolutions['level_count']
    dimensions = img.resolutions['level_dimensions']
    # print('img.metadata')
    # region = img.read_region(location=(0,0), size=dimensions[count-1],level=count-1) # lowest resolution
    region = img.read_region(level=count-1) # same
    time_1 = time.time()
    print('cucim shape: ', region.shape)
    total_time = time_1 - time_0 
    # plt.imshow(region)
    return total_time

def readPatches(patches_path):
    time_0 = time.time()
    images = []
    for filename in os.listdir(patches_path):
        img = cv2.imread(os.path.join(patches_path, filename))
        if img is not None:
            images.append(img)
    time_1 = time.time()
    total_time = time_1 - time_0 
    return total_time

if __name__ == '__main__':
    
    ################## Whole Slide #####################
    wsi_full_path = '2bd704_0.tif'
    downsample = 0
    # openslide
    # time_openSlide = wsiOpenSlid(wsi_full_path, downsample)
    # print('Time taken by openslide: ', time_openSlide)
    
    # slideio
    # time_slideIo = wsiSlideio(wsi_full_path)
    # print('Time taken by slideIo: ', time_slideIo)  
    
    # cucim
    # time_cucim = wsiCucim(wsi_full_path)
    # print('Time taken by cucim: ', time_cucim)
    
    ################# Patches ######################
    patches_path = 'D:/mayo_challenge_3TB/dataset/patches/00c058_0'
    patchReadTime = readPatches(patches_path)
    print('patch reading time: ', patchReadTime)
    
    
    
    
    
    
    
  