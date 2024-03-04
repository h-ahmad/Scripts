# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 11:20:22 2024

@author: hussain
"""

import argparse
import os
import numpy as np
import pandas as pd
import cv2

parser = argparse.ArgumentParser('Calculating IoU/dice', add_help=False)
parser.add_argument('--gt_mask_path', default='../../../../data/raabindata/Ground Truth/', type=str, help='root path to GT classes/images')
parser.add_argument('--pred_mask_path', default='../MCTformer_plus/output/aml_mll_pbc_bmc/raabin/fold5/output_masks/', type=str, help='root path to predicted classes/images')
parser.add_argument('--output_path', default='../MCTformer_plus/output/aml_mll_pbc_bmc/raabin/fold1/', type=str)
parser.add_argument('--output_filename', default='iou_dice.csv', type=str)
args = parser.parse_args()

class SegmentationEvaluation():
    def __init__(self, gt_mask_path, pred_mask_path):
        self.gt_mask_path = gt_mask_path
        self.pred_mask_path = pred_mask_path
        self.gt_mask_list, self.pred_mask_list = [], []
        self.gt_masks, self.pred_masks = self.read_data()
        self.iou_values, self.dice_values = [], []

    def read_data(self):
        check_list = []
        for pred_cls_img in os.listdir(self.pred_mask_path):
            cls_img_path = os.path.join(self.pred_mask_path, pred_cls_img)
            if os.path.isdir(cls_img_path):
                for pred_mask in os.listdir(cls_img_path):
                    read_pred_mask = cv2.imread(os.path.join(cls_img_path, pred_mask))
                    self.pred_mask_list.append(read_pred_mask)
                    pred_mask_name, pred_extension = os.path.splitext(pred_mask)
                    check_list.append(pred_mask_name)
            else:
                read_pred_mask = cv2.imread(cls_img_path)
                self.pred_mask_list.append(read_pred_mask)
                pred_mask_name, pred_extension = os.path.splitext(pred_cls_img)
                check_list.append(pred_mask_name)                
                           
        for gt_cls_img in os.listdir(self.gt_mask_path):
            cls_img_path = os.path.join(self.gt_mask_path, gt_cls_img)
            if os.path.isdir(cls_img_path):
                for gt_mask in os.listdir(cls_img_path):                    
                    gt_mask_name, gt_extension = os.path.splitext(gt_mask)
                    if gt_mask_name in check_list:
                        read_gt_mask = cv2.imread(os.path.join(cls_img_path, gt_mask))
                        self.gt_mask_list.append(read_gt_mask)
            else:
                gt_mask_name, gt_extension = os.path.splitext(gt_cls_img)
                if gt_mask_name in check_list:
                    read_gt_mask = cv2.imread(cls_img_path)
                    self.gt_mask_list.append(read_gt_mask)                                
        return self.gt_mask_list, self.pred_mask_list       

    def compute_iou(self, mask1, mask2):
        intersection = np.logical_and(mask1, mask2)
        union = np.logical_or(mask1, mask2)
        iou = np.sum(intersection) / np.sum(union)
        return iou

    def dice_coefficient(self, mask1, mask2):
        intersection = np.sum(np.logical_and(mask1, mask2))
        total_pixels_mask1 = np.sum(mask1)
        total_pixels_mask2 = np.sum(mask2)
        dice = (2.0 * intersection) / (total_pixels_mask1 + total_pixels_mask2)
        return dice

    def compute_iou_dice(self):
        for i, mask1 in enumerate(self.gt_masks):
            mask2 = self.pred_masks[i]            
            iou = self.compute_iou(mask1, mask2)
            dice = self.dice_coefficient(mask1, mask2)
            self.iou_values.append(iou)
            self.dice_values.append(dice)
        return self.iou_values, self.dice_values         

if __name__ == '__main__':
    seg_eval = SegmentationEvaluation(args.gt_mask_path, args.pred_mask_path)
    iou, dice = seg_eval.compute_iou_dice()
    mean_iou = np.mean(np.array(iou))
    mean_dice = np.mean(np.array(dice))
    print('mean_iou: ', mean_iou, ', mean_dice: ', mean_dice)
    data = {'mean_iou':mean_iou, 'mean_dice':mean_dice}
    # df = pd.DataFrame(data)    
    # df.to_csv(os.path.join(args.output_path, args.output_filename))
