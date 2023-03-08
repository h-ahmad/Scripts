#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 16:07:42 2023

@author: hussain
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
# =============================================================================
#    
# x = [0, 2, 4, 6, 8, 10, 12, 14, 16]
# y1 = [0, 30, 60, 100, 200, 280, 392, 502, 630]
# y2 = [0, 252, 493, 762, 1004, 1485, 1998, 2278, 2750]
# y3 = [0, 10, 48, 100, 200, 300, 400, 550, 700]
# y4 = [0, 250, 500, 620, 860, 1175, 1325, 1650, 1940]
# plt.plot(x, y1, label = 'CNN-2 without FHE')
# plt.plot(x, y2, label = 'CNN-2 with FHE')
# plt.plot(x, y3, label = 'LeNet-1 without FHE')
# plt.plot(x, y4, label = 'NeNet-1 with FHE')
# plt.legend()
# plt.xlabel('Number of clients (N)', fontsize=12)
# plt.ylabel('Time (s)', fontsize=12)
# plt.title('Model training with and without FHE.', fontsize=12)
# plt.savefig('fhe_time.pdf')
# plt.show()
# =============================================================================

x_axis = []
custom = []
cnn2 = []
lenet1 = []
resnet18 = []
efficient = []

custom_folder = 'results_mnist_simple'
custom_file_path = os.path.join(custom_folder, 'test_acc.csv')
custom_df = pd.read_csv(custom_file_path)
check_value = 0
for i in range(100):
    x_axis.append(i)
    if i < len(custom_df) and not pd.isnull(custom_df.loc[i, 'mnist_m1']):
        check_value = custom_df.loc[i, 'mnist_m1']
    custom.append(check_value)
    
cnn2_folder = 'results_mnist_cnn2'
cnn2_file_path = os.path.join(cnn2_folder, 'test_acc.csv')
cnn2_df = pd.read_csv(cnn2_file_path)
for i in range(len(cnn2_df)):
    if i < len(cnn2_df) and not pd.isnull(cnn2_df.loc[i, 'mnist_m1']):
        check_value = cnn2_df.loc[i, 'mnist_m1']
    cnn2.append(check_value)
    
lenet1_folder = 'results_mnist_lenet1'
lenet1_file_path = os.path.join(lenet1_folder, 'test_acc.csv')
lenet1_df = pd.read_csv(lenet1_file_path)
check_value = 0
for i in range(100):
    if i < len(lenet1_df) and not pd.isnull(lenet1_df.loc[i, 'mnist_m1']):
        check_value = lenet1_df.loc[i, 'mnist_m1']
    lenet1.append(check_value)

resnet18_folder = 'results_mnist_resnet18'
resnet18_file_path = os.path.join(resnet18_folder, 'test_acc.csv')
resnet18_df = pd.read_csv(resnet18_file_path)
check_value = 0
for i in range(100):        
    if i < len(resnet18_df) and not pd.isnull(resnet18_df.loc[i, 'mnist_m1']):
        check_value = resnet18_df.loc[i, 'mnist_m1']
    resnet18.append(check_value)
    
efficient_folder = 'results_mnist_efficienct'
efficient_file_path = os.path.join(efficient_folder, 'test_acc.csv')
efficient_df = pd.read_csv(efficient_file_path)
check_value = 0
for i in range(100):
    if i < len(efficient_df) and not pd.isnull(efficient_df.loc[i, 'mnist_m2']):
        check_value = efficient_df.loc[i, 'mnist_m1']
    efficient.append(check_value)    
    
plt.plot(x_axis, custom, label = 'CustomNet')
plt.plot(x_axis, cnn2, label = 'CNN-2')
plt.plot(x_axis, lenet1, label = 'LeNet-1')
plt.plot(x_axis, resnet18, label = 'ResNet18 (pretrained)')
plt.plot(x_axis, efficient, label = 'Efficient-B5')
plt.legend()
plt.xlabel('Communication rounds', fontsize=12)
plt.ylabel('Test Accuracy (%)', fontsize=12)
plt.title('Test accuracy on USPS global test set.', fontsize=12)
# plt.savefig('mnist.pdf')
plt.show()
