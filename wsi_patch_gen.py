# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 16:23:00 2023

@author: Hussain Ahmad Madni
"""

import glob
import os
from multiprocessing import Pool
import openslide
import numpy as np
import cv2
from skimage.filters import threshold_otsu
import PIL.Image as Image
import time
import slideio

def getFolder_name(orig_dir, level, psize):
    tslide = os.path.basename(orig_dir)
    folderName = os.path.dirname(orig_dir)
    subfolder_name = float(psize * level) / 256
    tfolder = os.path.join(folderName, str(subfolder_name*10), tslide)
    return tfolder

def get_roi_bounds(tslide, isDrawContoursOnImages=False, mask_level=5, cls_kernel=50, open_kernal=30):
    dims = tslide.level_dimensions
    scale_factor = 2
    level1_dim = (int(dims[0][0]/scale_factor), int(dims[0][1]/scale_factor))
    subSlide = tslide.read_region((0, 0), mask_level, level1_dim)
    # subSlide = tslide.read_region((0, 0), mask_level, tslide.level_dimensions[mask_level])
    subSlide_np = np.array(subSlide)
    hsv = cv2.cvtColor(subSlide_np, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    try:
        hthresh = threshold_otsu(h)
        sthresh = threshold_otsu(s)
        vthresh = threshold_otsu(v)
    except:
        return np.NaN, np.NaN
    minhsv = np.array([hthresh, sthresh, 70], np.uint8)
    maxhsv = np.array([180, 255, vthresh], np.uint8)
    thresh = [minhsv, maxhsv]
    # extraction the countor for tissue
    mask = cv2.inRange(hsv, thresh[0], thresh[1])
    close_kernel = np.ones((cls_kernel, cls_kernel), dtype=np.uint8)
    image_close_img = Image.fromarray(cv2.morphologyEx(np.array(mask), cv2.MORPH_CLOSE, close_kernel))
    open_kernel = np.ones((open_kernal, open_kernal), dtype=np.uint8)
    image_open_np = cv2.morphologyEx(np.array(image_close_img), cv2.MORPH_OPEN, open_kernel)
    #print('image_open_np', image_open_np.shape, image_open_np.min(), image_open_np.max())
    contours, _ = cv2.findContours(image_open_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #_, contours, _ = cv2.findContours(image_open_np, cv2.RETR_FLOODFILL, cv2.CHAIN_APPROX_SIMPLE)
    #_, contours,_ = cv2.findContours(image_open_np, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE) 
    boundingBox = [cv2.boundingRect(c) for c in contours]
    boundingBox = [sst for sst in boundingBox if sst[2] > 150 and sst[3] > 150]
    #print('boundingBox number ', len(boundingBox))
    if isDrawContoursOnImages:
        line_color = (0, 0, 0)  # blue color code
        contours_rgb_image_np = np.array(subSlide)
        cv2.drawContours(contours_rgb_image_np, contours, -1, line_color, 50)
        contours_rgb_image_np = cv2.resize(contours_rgb_image_np, (0, 0), fx=0.2, fy=0.2)
        countours_rgb_image_img = Image.fromarray(contours_rgb_image_np.astype(np.uint8))
        countours_rgb_image_img.show()
    return image_open_np, boundingBox

def Extract_Patch_From_Slide_STRIDE(tslide:openslide.ImageSlide, tissue_mask, patch_save_folder, patch_level, mask_level, patch_stride, patch_size, threshold, level_list=[1], patch_size_list=[256], patch_surfix='tif'):
    assert patch_level == level_list[0]
    assert patch_size == patch_size_list[0]
    mask_sH, mask_sW = tissue_mask.shape
    #print(f'tissue mask shape {tissue_mask.shape}')
    mask_patch_size = int(patch_size // pow(2, mask_level-patch_level))
    mask_patch_size_square = mask_patch_size ** 2
    mask_stride = int(patch_stride // pow(2, mask_level-patch_level))
    #mag_factor = pow(2, mask_level-patch_level)
    mag_factor = pow(2, mask_level)    # !!
    #print(f'slide level dimensions {tslide.level_dimensions}, mask patch size {mask_patch_size}, mask stride {mask_stride}, mag_factor {mag_factor}')
    tslide_name = os.path.basename(patch_save_folder)
    num_error = 0
    for iw in range(int(mask_sW//mask_stride)):
        for ih in range(int(mask_sH//mask_stride)):
            ww = iw * mask_stride
            hh = ih * mask_stride
            if (ww+mask_patch_size) < mask_sW and (hh+mask_patch_size) < mask_sH:
                tmask = tissue_mask[hh:hh+mask_patch_size, ww:ww+mask_patch_size]
                mRatio = float(np.sum(tmask > 0)) / mask_patch_size_square
                if mRatio > threshold:
                    for sstLevel, tSize in zip(level_list, patch_size_list):
                        try:
                            tsave_folder = getFolder_name(patch_save_folder, sstLevel, tSize)
                            sww = ww * mag_factor
                            shh = hh * mag_factor
                            cW_l0 = sww + (patch_size // 2) * pow(2, patch_level)
                            cH_l0 = shh + (patch_size // 2) * pow(2, patch_level)
                            tlW_l0 = cW_l0 - (tSize // 2) * pow(2, sstLevel)
                            tlH_l0 = cH_l0 - (tSize // 2) * pow(2, sstLevel)
                            #print('save here')
                            tpatch = tslide.read_region((tlW_l0, tlH_l0), sstLevel, (tSize, tSize)) ## (x, y) tuple giving the top left pixel in the level 0 reference frame
                            # if tpatch.mode == 'RGBA':
                            #     tpatch = tpatch.convert('RGB')
                            tname = f'{tslide_name}_{ww * mag_factor}_{hh * mag_factor}_{iw}_{ih}_WW_{mask_sW // mask_stride}_HH_{mask_sH // mask_stride}.{patch_surfix}'
                            tpatch.save(os.path.join(tsave_folder, tname))
                        except:
                            #raise RuntimeError(f'slide error {tslide_name}')
                            num_error += 1
                            print(f'slide {tslide_name} error patch {num_error}')
    if num_error != 0:
        print(f'---------------------In total {num_error} error patch for slide {tslide_name}')
        
# =============================================================================
# def readopenslide(slide_full_path):
#     time_0 = time.time()
#     slide = openslide.open_slide(slide_full_path)
#     time_1 = time.time()
#     total_time = time_1 - time_0
#     print('Time take by openslide: ', total_time)
#     return slide
# 
# def readslideio(slide_full_path):
#     time_0 = time.time()
#     slide = slideio.open_slide(slide_full_path, "GDAL")
#     time_1 = time.time()
#     total_time = time_1 - time_0
#     print('Time take by slideio: ', total_time)
#     return slide
# =============================================================================

def Thread_PatchFromSlides(args):
    normSlidePath, slideName, tsave_slide_dir, patch_level_list, psize_list, mask_dimension_level, patch_dimension_level, stride, psize, tissue_mask_threshold = args
    for tlevel, tsize in zip(patch_level_list, psize_list):
        tsave_dir_level = getFolder_name(tsave_slide_dir, tlevel, tsize)
        if not os.path.exists(tsave_dir_level):
            os.makedirs(tsave_dir_level)
            
    tslide = openslide.open_slide(normSlidePath)
    # tslide = readopenslide(normSlidePath);
    # print('type of slide by openslide: ', type(tslide))
    # tslide = readslideio(normSlidePath);
    
    tissue_mask, boundingBoxes = get_roi_bounds(tslide, isDrawContoursOnImages=False, mask_level=mask_dimension_level)  # mask_level: absolute level
    print('before div: ', tissue_mask.shape, tissue_mask.min(), tissue_mask.max())
    tissue_mask = tissue_mask // 255
    print('After div: ', tissue_mask.shape, tissue_mask.min(), tissue_mask.max())
    Extract_Patch_From_Slide_STRIDE(tslide, tissue_mask, tsave_slide_dir,
                                                  patch_level=patch_dimension_level, mask_level=mask_dimension_level,
                                                  patch_stride=stride, patch_size=psize,
                                                  threshold=tissue_mask_threshold,
                                                  level_list=patch_level_list,
                                                  patch_size_list=psize_list
                                                  )

if __name__ == "__main__":
    patch_level_list = [0] # 0, 1, 2
    psize_list = [256] #[256, 192, 256]
    mask_dimension_level = 0    # 5
    patch_dimension_level = 0    ## 0: 40x, 1: 20x
    stride = 256
    psize = 256
    tissue_mask_threshold = 0.8
    num_thread = 4
    slides_folder_dir = './slides'
    slide_paths = glob.glob(os.path.join(slides_folder_dir, '*.tif'))
    save_folder_dir = './patches'
    pool = Pool(processes=num_thread)
    for tSlidePath in slide_paths:
        arg_list = []
        slideName = os.path.basename(tSlidePath).split('.')[0]    
        tsave_slide_dir = os.path.join(save_folder_dir, slideName)
        arg_list.append([tSlidePath, slideName, tsave_slide_dir, patch_level_list, 
                         psize_list, mask_dimension_level, patch_dimension_level, stride, psize, tissue_mask_threshold])
        print('In process slide: ', slideName)
        pool.map(Thread_PatchFromSlides, arg_list)