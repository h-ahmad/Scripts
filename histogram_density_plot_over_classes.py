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
parser.add_argument('--data_path', type = str, default = './data/mnist_m/', help = 'Path to the main directory')
parser.add_argument('--dataset_name', type = str, default = 'mnist-m', help = 'cifar10, mnist, mnist-m, svhn, usps')
parser.add_argument('--set_type', type=str, default='train', choices=['train', 'test'])
parser.add_argument('--file_name', type = str, default = 'mnist_m_train_test.pkl', help = 'File name with extension')
parser.add_argument('--file_format', type = str, default = 'pickle', help = 'pickle, numpy')
parser.add_argument('--graph_type', type = str, default = 'distribution', help = 'scatter, distribution')
parser.add_argument('--client_index', type=int, default = 2, help = 'Client index if data is stacked for participating clients.')
args = parser.parse_args() 


def class_distribution(classes_name):
    
    if args.file_format == 'pickle':
        with open(os.path.join(args.data_path, args.file_name), 'rb') as file:
            data_store = pickle.load(file)
            print('keys: ', data_store.keys())
            X_train, y_train, X_test, y_test = data_store['X_train'], data_store['y_train'], data_store['X_test'], data_store['y_test']
            print('X_train.shape', X_train.shape)
            print('y_train.shape', y_train.shape)
    
    if args.file_format == 'numpy':
        X_train = np.load(os.path.join(args.data_path, 'X_train.npy'), allow_pickle=True)
        y_train = np.load(os.path.join(args.data_path, 'y_train.npy'), allow_pickle=True)
        X_test = np.load(os.path.join(args.data_path, 'X_test.npy'), allow_pickle=True)
        y_test = np.load(os.path.join(args.data_path, 'y_test.npy'), allow_pickle=True)
        print('X_train.shape', X_train.shape) # (4, 12500, 3, 32, 32) => 4 clients each having 12500 samples
        print('y_train.shape', y_train.shape) # (4, 12500, 1)
    y_train = y_train[args.client_index] # change index for each client
    y_test = y_test[args.client_index]
    
    if args.set_type in 'train':
        classes, counts = np.unique(y_train, return_counts=True)
    else:
        classes, counts = np.unique(y_test, return_counts=True)
    plt.figure(figsize=(6, 5))
    counts = counts.tolist()
    import pandas as pd
    score_series = pd.Series(counts)
    fig = score_series.plot(kind='bar')
    fig.set_xticklabels(classes)
    fig.bar_label(fig.containers[0], label_type='edge')
    plt.title('Class distribution in '+ args.dataset_name+' '+args.set_type+' set.')
    plt.savefig(os.path.join(args.data_path, args.dataset_name+"_"+args.set_type+"_distribution.pdf"), format="pdf", bbox_inches="tight")
    
# =============================================================================
#     bars = plt.barh(classes, counts)
#     plt.yticks(classes)
#     plt.bar_label(bars, label_type='center', color='white', labels=[f'{x:,}' for x in bars.datavalues])
#     plt.title('Class distribution in '+ args.dataset_name+' '+args.set_type+' set.')
#     plt.savefig(os.path.join(args.data_path, args.dataset_name+"_"+args.set_type+"_distribution.pdf"), format="pdf", bbox_inches="tight")
# =============================================================================

def scatter_plot(classes_name):
    if args.file_format == 'pickle':
        with open(os.path.join(args.data_path, args.file_name), 'rb') as file:
            data_store = pickle.load(file)
            X_train, y_train, X_test, y_test = data_store['X_train'], data_store['y_train'], data_store['X_test'], data_store['y_test']
            print('X_train: ', X_train.shape)
            print('y_train: ', y_train.shape)
    if args.file_format == 'numpy':
        X_train = np.load(os.path.join(args.data_path, 'X_train.npy'), allow_pickle=True)
        y_train = np.load(os.path.join(args.data_path, 'y_train.npy'), allow_pickle=True)
        X_test = np.load(os.path.join(args.data_path, 'X_test.npy'), allow_pickle=True)
        y_test = np.load(os.path.join(args.data_path, 'y_test.npy'), allow_pickle=True)
        print('X_train.shape', X_train.shape) # (4, 12500, 3, 32, 32) => 4 clients each having 12500 samples
        print('y_train.shape', y_train.shape) # (4, 12500, 1)
    y_train = y_train[args.client_index] # change index for each client
    y_test = y_test[args.client_index]
    
    if args.set_type in 'train':
        classes, counts = np.unique(y_train, return_counts=True)
    else:
        classes, counts = np.unique(y_test, return_counts=True)
    colors = np.random.rand(len(classes)) 
    fig, ax = plt.subplots()
    sizes = np.array(counts/10)
    ax.scatter(classes, counts, s = sizes,  c=colors, alpha=0.5)
    plt.xticks(classes)
    # annotate labels
    for i, txt in enumerate(classes):
        ax.annotate(classes[i], (classes[i], counts[i]))           
    plt.title('Class distribution in '+args.dataset_name+' '+args.set_type+' set.')
    plt.savefig(os.path.join(args.data_path, args.dataset_name+"_"+args.set_type+"_scatter.pdf"), format="pdf", bbox_inches="tight")
    plt.show()
    
    
if __name__ == '__main__':
    if args.dataset_name == 'cifar10':
        classes_name = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
    if args.dataset_name == 'mnist':
        classes_name = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    if args.dataset_name == 'mnist-m':
        classes_name = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    if args.dataset_name == 'svhn':
        classes_name = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    if args.dataset_name == 'usps':
        classes_name = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    if args.graph_type == 'distribution':
        class_distribution(classes_name)
    if args.graph_type == 'scatter':
        scatter_plot(classes_name)
