#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 13:07:50 2022

@author: hussain
"""


import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

def class_distribution(data_path, pickle_file_name, classes_name):
    with open(os.path.join(data_path, pickle_file_name), 'rb') as file:
        data_store = pickle.load(file)
        xTrain, yTrain, xTest, yTest = data_store['X_train'], data_store['y_train'], data_store['X_test'], data_store['y_test']
        print('xTrain: ', xTrain.shape)
        print('yTrain: ', yTrain.shape)
        classes, counts = np.unique(yTrain, return_counts=True)        
        bars = plt.barh(classes_name, counts)
#        plt.bar_label(bars, label_type='edge', labels=[f'{x:,}' for x in bars.datavalues])
        plt.bar_label(bars, label_type='center', labels=[f'{x:,}' for x in bars.datavalues])
        plt.title('Class distribution in training set')

def scatter_plot(data_path, pickle_file_name, classes_name):
    with open(os.path.join(data_path, pickle_file_name), 'rb') as file:
        data_store = pickle.load(file)
        xTrain, yTrain, xTest, yTest = data_store['X_train'], data_store['y_train'], data_store['X_test'], data_store['y_test']
        print('xTrain: ', xTrain.shape)
        print('yTrain: ', yTrain.shape)
        classes, counts = np.unique(yTrain, return_counts=True) 
        area = (30 * np.random.rand(len(classes)))**2  # 0 to 15 point radii
        colors = np.random.rand(len(classes))
#        plt.scatter(classes, counts, s=area, c=colors, alpha=0.5)
        
        fig, ax = plt.subplots()
        ax.scatter(classes, counts, s=area, c=colors, alpha=0.5)
        # annotate labels
        for i, txt in enumerate(classes):
            ax.annotate(classes[i], (classes[i], counts[i]))        
        
        plt.show()
    
    
if __name__ == '__main__':
    data_path = './data'
    pickle_file_name = 'mnist_client3.pkl'
#    classes_name = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck'] # cifar10
    classes_name = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] # mnist
    class_distribution(data_path, pickle_file_name, classes_name)    
    scatter_plot(data_path, pickle_file_name, classes_name)