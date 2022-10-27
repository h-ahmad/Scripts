#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 14:40:46 2022

@author: hussain
"""


import argparse
import pickle
import os
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
from skimage import io
import numpy as np

parser = argparse.ArgumentParser(description = 'Main Script')
parser.add_argument('--data_path', type = str, default = './data', help = 'Path to the main directory')
parser.add_argument('--dataset_name', type = str, default = 'mnist', help = 'cifar10, mnist')
parser.add_argument('--client_data_folder', type = str, default = 'client3', help = 'client1, client2, client3')
parser.add_argument('--client_label_file', type = str, default = '3.csv', help = '1.csv, 2.csv, 3.csv')
args = parser.parse_args() 

def load_data(data_path, client_label_file, client_data_folder):    
    label_path = os.path.join(data_path, client_label_file)
    label_path = pd.read_csv(label_path)
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    # train data
    for index, row in label_path.iterrows():
        image_name = row[0]
        image_label = row[1]
        image_path = os.path.join(data_path, client_data_folder)
        image_name = os.path.join(image_path, image_name)
        image = io.imread(image_name) 
        X_train.append(image)
        y_train.append(image_label)
    # test data
    test_label = os.path.join(data_path, 'test.csv')
    test_label = pd.read_csv(test_label)
    for index, row in test_label.iterrows():
        image_name = row[0]
        image_label = row[1]
        image_path = os.path.join(data_path, 'test')
        image_name = os.path.join(image_path, image_name)
        image = io.imread(image_name) 
        X_test.append(image)
        y_test.append(image_label)
    print('X_train: ', np.array(X_train).shape)
    print('y_train: ', np.array(y_train).shape)
    print('X_test: ', np.array(X_test).shape)
    print('y_test: ', np.array(y_test).shape)
    return (np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test))


def data_to_pickle(pickle_file_name, X_train, y_train, X_test, y_test):
        with open(pickle_file_name, 'wb') as file:  
            data_store = {'X_train': X_train, 'y_train': y_train, 'X_test': X_test, 'y_test': y_test}
            pickle.dump(data_store, file)
    # show data from pickle
        with open(pickle_file_name, 'rb') as file:
            data_store = pickle.load(file)
            print(data_store.keys())     
            print('X_train: ', data_store['X_train'].shape)
            print('y_train: ', data_store['y_train'].shape)
            print('X_test: ', data_store['X_test'].shape)
            print('y_test: ', data_store['y_test'].shape)   
            
if __name__ == '__main__':    
    X_train, y_train, X_test, y_test = load_data(args.data_path, args.client_label_file, args.client_data_folder) 
    
    pickle_file_name = os.path.join(args.data_path, args.dataset_name+'_'+args.client_data_folder+'.pkl')
    data_to_pickle(pickle_file_name, X_train, y_train, X_test, y_test)        
