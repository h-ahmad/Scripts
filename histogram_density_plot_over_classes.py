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
import argparse

parser = argparse.ArgumentParser(description = 'Main Script')
parser.add_argument('--data_path', type = str, default = 'data/cifar10_4clients_non_iid_alpha0.5/', help = 'Path to the main directory')
parser.add_argument('--dataset_name', type = str, default = 'cifar10', help = 'cifar10, mnist')
parser.add_argument('--file_name', type = str, default = 'mnist_client3.pkl', help = 'File name with extension')
parser.add_argument('--file_format', type = str, default = 'numpy', help = 'pickle, numpy')
parser.add_argument('--graph_type', type = str, default = 'distribution', help = 'scatter, distribution')
args = parser.parse_args() 


def class_distribution(classes_name):
    
    if args.file_format == 'pickle':
        with open(os.path.join(args.data_path, args.file_name), 'rb') as file:
            data_store = pickle.load(file)
            print('keys: ', data_store.keys())
            X_train, y_train, xTest, yTest = data_store['X_train'], data_store['y_train'], data_store['X_test'], data_store['y_test']
            print('X_train.shape', X_train.shape)
            print('y_train.shape', y_train.shape)
    
    if args.file_format == 'numpy':
        X_train = np.load(os.path.join(args.data_path, 'X_train.npy'), allow_pickle=True)
        y_train = np.load(os.path.join(args.data_path, 'y_train.npy'), allow_pickle=True)
        print('X_train.shape', X_train.shape) # (4, 12500, 3, 32, 32) => 4 clients each having 12500 samples
        print('y_train.shape', y_train.shape) # (4, 12500, 1)
        y_train = y_train[0] # change index for each client
    
    classes, counts = np.unique(y_train, return_counts=True)
    bars = plt.barh(classes_name, counts)
    plt.bar_label(bars, label_type='center', labels=[f'{x:,}' for x in bars.datavalues])
    plt.title('Class distribution in training set')

def scatter_plot(classes_name):
    if args.file_format == 'pickle':
        with open(os.path.join(args.data_path, args.file_name), 'rb') as file:
            data_store = pickle.load(file)
            X_train, y_train, xTest, yTest = data_store['X_train'], data_store['y_train'], data_store['X_test'], data_store['y_test']
            print('X_train: ', X_train.shape)
            print('y_train: ', y_train.shape)
    if args.file_format == 'numpy':
        X_train = np.load(os.path.join(args.data_path, 'X_train.npy'), allow_pickle=True)
        y_train = np.load(os.path.join(args.data_path, 'y_train.npy'), allow_pickle=True)
        print('X_train.shape', X_train.shape) # (4, 12500, 3, 32, 32) => 4 clients each having 12500 samples
        print('y_train.shape', y_train.shape) # (4, 12500, 1)
        y_train = y_train[2] # change index for each client
    
    classes, counts = np.unique(y_train, return_counts=True) 
    area = (30 * np.random.rand(len(classes)))**2  # 0 to 15 point radii
    colors = np.random.rand(len(classes)) 
    fig, ax = plt.subplots()
    ax.scatter(classes, counts, s=area, c=colors, alpha=0.5)
    # annotate labels
    for i, txt in enumerate(classes):
        ax.annotate(classes[i], (classes[i], counts[i]))        
    
    plt.title('Class distribution in training set')
    plt.show()
    
    
if __name__ == '__main__':
    if args.dataset_name == 'cifar10':
        classes_name = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
    if args.dataset_name == 'mnist':
        classes_name = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    if args.graph_type == 'distribution':
        class_distribution(classes_name)
    if args.graph_type == 'scatter':
        scatter_plot(classes_name)
