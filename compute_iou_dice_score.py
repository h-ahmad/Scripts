#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 15:22:37 2024

@author: hussain
"""

import argparse
import os
import csv
import cv2
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser('Calculating IoU', add_help=False)
parser.add_argument('--data_path', default='../datasets', type=str)
parser.add_argument('--data_file_name', default='metadata_kf5.csv', type=str)
args = parser.parse_args()

def read_data():
    img_names = []
    with open(os.path.join(args.data_path, args.data_file_name), 'r') as file:            
        file_rows = csv.reader(file)
        for i, row in enumerate(file_rows):
            if i > 0 and row[3] == 'test':
                img_path = row[1].split('../')[3] # original path without back slash
                img_names.append(img_path)
    return img_names

def mask_transform(mask):
    # gt_mask = cv2.cvtColor(gt_mask, cv2.COLOR_BGR2GRAY) 
    mask = (mask > 0).astype(np.uint8) # values higher than threshold 128 are converted to 1. Other are 0.
    # cv2.imshow('My Image', mask)
    # cv2.waitKey(0)
    return mask
    
def calculate_iou(gt_mask, pred_mask):
    print('pred_mask: ', pred_mask.shape, 'min: ', pred_mask.min(), ' max: ', pred_mask.max())
    tp = 0
    fp = 0
    fn = 0
    for i in range(len(gt_mask)):
        for j in range(len(gt_mask[0])):
            if gt_mask[i][j] == 1 and pred_mask[i][j] == 1:
                tp += 1
            elif gt_mask[i][j] == 0 and pred_mask[i][j] == 1:
                fp += 1
            elif gt_mask[i][j] == 1 and pred_mask[i][j] == 0:
                fn += 1
    iou = tp / (tp + fp + fn)
    return iou
    
def get_mask_list(gt_mask_paths):
    gt_mask_list, mctformer_mask_list, maskrcnn_mask_list = [], [], []
    for i, gt_mask_path in enumerate(gt_mask_paths):
        name = gt_mask_path.split('/')[-1] # img/mask name with extension
        gt_mask = cv2.imread(os.path.join('../../../', gt_mask_path))
        gt_mask = mask_transform(gt_mask)
        gt_mask_list.append(gt_mask)
        
        mctformer_mask_path = os.path.join('../MCTformer/output_masks', name.split('.')[0]+'.png')
        if os.path.exists(mctformer_mask_path):
            mctformer_mask = cv2.imread(mctformer_mask_path)
            # mctformer_mask = cv2.cvtColor(mctformer_mask, cv2.COLOR_BGR2GRAY) 
            mctformer_mask = (mctformer_mask > 0).astype(np.uint8)
            mctformer_mask_list.append(mctformer_mask)
        else:
            mctformer_mask_list.append('')
            
        maskrcnn_mask_path = os.path.join('../maskrcnn/output_masks', name)
        if os.path.exists(maskrcnn_mask_path):
            maskrcnn_mask = cv2.imread(maskrcnn_mask_path)
            # maskrcnn_mask = cv2.cvtColor(maskrcnn_mask, cv2.COLOR_BGR2GRAY) 
            maskrcnn_mask = (maskrcnn_mask > 0).astype(np.uint8)
            maskrcnn_mask_list.append(maskrcnn_mask)
        else:
            maskrcnn_mask_list.append('')
        # if i == 1:
            # break
    return gt_mask_list, mctformer_mask_list, maskrcnn_mask_list

def compute_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou = np.sum(intersection) / np.sum(union)
    return iou

def compute_iou_between_lists(mask_list1, mask_list2):
    iou_values = []
    for i, mask1 in enumerate(mask_list1):
        mask2 = mask_list2[i]
        if mask2 == '':
            iou = 0.0  
        else:
            # iou = compute_iou(mask1, mask2)
            iou = calculate_iou(mask1, mask2)
        iou_values.append(iou)
    return iou_values

if __name__ == '__main__':
    gt_mask_paths = read_data()
    gt_mask_list, mctformer_mask_list, maskrcnn_mask_list = get_mask_list(gt_mask_paths)
    mctformer_iou = compute_iou_between_lists(gt_mask_list, mctformer_mask_list)
    maskrcnn_iou = compute_iou_between_lists(gt_mask_list, maskrcnn_mask_list)
    mctformer_iou_mean = np.mean(np.array(mctformer_iou))
    maskrcnn_iou_mean = np.mean(np.array(maskrcnn_iou))
    # print('mctformer_iou_mean: ', mctformer_iou_mean, ' maskrcnn_iou_mean: ', maskrcnn_iou_mean)
    data = {'mactformer_iou': mctformer_iou, 'maskrcnn_iou': maskrcnn_iou, 'mctformer_iou_mean': mctformer_iou_mean, 'maskrcnn_iou_mean': maskrcnn_iou_mean}
    df = pd.DataFrame(data)
    # df.to_csv('iou.csv')
    

# =============================================================================
# class IoUCalculator:
#  
#     @staticmethod
#     def main():
#         # Create a ground truth mask and a predicted mask
#         gtMask = [[1, 1, 0], [0, 1, 0], [0, 0, 0]]
#         predMask = [[1, 1, 0], [0, 1, 1], [0, 0, 0]]
#         # Calculate IoU
#         iou = IoUCalculator.calculateIoU(gtMask, predMask)
#         print("IoU: ", iou)
#  
#     @staticmethod
#     def calculateIoU(gtMask, predMask):
#         # Calculate the true positives,
#         # false positives, and false negatives
#         tp = 0
#         fp = 0
#         fn = 0
#  
#         for i in range(len(gtMask)):
#             for j in range(len(gtMask[0])):
#                 if gtMask[i][j] == 1 and predMask[i][j] == 1:
#                     tp += 1
#                 elif gtMask[i][j] == 0 and predMask[i][j] == 1:
#                     fp += 1
#                 elif gtMask[i][j] == 1 and predMask[i][j] == 0:
#                     fn += 1
#  
#         # Calculate IoU
#         iou = tp / (tp + fp + fn)
#         return iou
# 
# if __name__ == '__main__':
#     IoUCalculator().main()
# =============================================================================
