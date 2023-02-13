#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 11:01:10 2022
@author: hussain
"""

import os
import pickle
import cv2
import csv
import argparse
import torch
from torchvision.utils import save_image

parser = argparse.ArgumentParser(description = 'Main Script')
parser.add_argument('--data_path', type = str, default = './data', help = 'Main path to the dataset')
parser.add_argument('--dataset_name', type = str, default = 'usps', help = 'cifar10, mnist, cifar100, mnist_m, svhn, usps')
parser.add_argument('--data_file_name', type = str, default = 'usps_train_test.pkl', help = 'Pickle file name')
parser.add_argument('--number_of_clients', type = int, default = 4, help = 'Number of client to which data is divided')
parser.add_argument('--is_data_distributed', type = bool, default = True, help = 'True if data is already iid or non-iid distributed over clients at first index and you want to split individually, False otherwise')
parser.add_argument('--this_client_number', type = int, default = 3, help = 'If is_data_distributed flag is True, then set current client number from 0 to any positive integer')
parser.add_argument('--output_type', type = str, default = 'separate', help = 'pickle, separate')
args = parser.parse_args() 

def load_data():
    with open(os.path.join(args.data_path, args.data_file_name), 'rb') as file:
        data_store = pickle.load(file)
        print(data_store.keys())     
        if args.is_data_distributed == True:
            data_store['X_train'] = data_store['X_train'][args.this_client_number]
            data_store['y_train'] = data_store['y_train'][args.this_client_number].reshape(-1)
            data_store['y_test'] = data_store['y_test'].reshape(-1)
        # convert numpy arrays to tensors
        data_store['X_train'] = torch.from_numpy(data_store['X_train'])
        data_store['X_test'] = torch.from_numpy(data_store['X_test'])
        
        print('X_train: ', data_store['X_train'].shape)
        print('y_train: ', data_store['y_train'].shape)
        print('X_test: ', data_store['X_test'].shape)
        print('y_test: ', data_store['y_test'].shape)
        
        total_data_count = data_store['y_train'].shape[0]
        if args.is_data_distributed == True:
            client_data = total_data_count
        else:
            client_data = int(total_data_count/args.number_of_clients)
    return (data_store['X_train'], data_store['y_train'], data_store['X_test'], data_store['y_test'], client_data)

def folder_images_csv_labels():
    X_train, y_train, X_test, y_test, client_data = load_data()    
    # save multiple train data for multiple clients given
    j = 0
    for i in range(y_train.shape[0]):
        if i % client_data == 0:
            j = j + 1
            if args.is_data_distributed == True:
                os.makedirs(os.path.join(args.data_path, 'client'+str(args.this_client_number)+'/'), exist_ok = True)
                data_store_path = os.path.join(args.data_path, 'client'+str(args.this_client_number)+'/')   
                csv_file = open(os.path.join(args.data_path, str(args.this_client_number)+'.csv'), 'w', newline='')
            else:
                os.makedirs(os.path.join(args.data_path, 'client'+str(j)+'/'), exist_ok = True)
                data_store_path = os.path.join(args.data_path, 'client'+str(j)+'/')   
                csv_file = open(os.path.join(args.data_path, str(j)+'.csv'), 'w', newline='')                 
            writer = csv.writer(csv_file)            
        save_image(X_train[i], os.path.join(data_store_path, str(i)+'.png'))
        # cv2.imwrite(os.path.join(data_store_path, str(i)+'.png'), image)                
        writer.writerow([str(i)+'.png', y_train[i]])
    csv_file.close()
    # save a single test data for all clients
    csv_file = open(os.path.join(args.data_path, 'test.csv'), 'w', newline='')
    writer = csv.writer(csv_file)     
    os.makedirs(os.path.join(args.data_path, 'test'), exist_ok = True)
    data_store_path = os.path.join(args.data_path, 'test')
    csv_file = open(os.path.join(args.data_path, 'test.csv'), 'w', newline='')
    writer = csv.writer(csv_file)    
    for k in range(y_test.shape[0]):
        save_image(X_test[k], os.path.join(data_store_path, str(k)+'.png'))
        # cv2.imwrite(os.path.join(data_store_path, str(k)+'.png'), image)
        writer.writerow([str(k)+'.png', y_test[k]])
    csv_file.close()

def data_to_pickle_file():
    X_train, y_train, X_test, y_test, client_data = load_data()
    X_train = X_train.numpy()
    y_train = y_train.squeeze(2)
    X_test = X_test.numpy()    
    y_test = y_test.squeeze(1)
    
    # save multiple train data for multiple clients given
    if args.is_data_distributed == True:        
        with open(os.path.join(args.data_path, 'client_'+str(args.this_client_number)+'_train_test.pkl'), 'wb') as file:  
            data_store = {'X_train': X_train, 'y_train': y_train, 'X_test': X_test, 'y_test': y_test}
            pickle.dump(data_store, file)
    else:
        for i in range(X_train.shape[0]):            
            with open(os.path.join(args.data_path, 'client_'+str(i)+'_train_test.pkl'), 'wb') as file:
                data_store = {'X_train': X_train[i], 'y_train': y_train[i], 'X_test': X_test, 'y_test': y_test}
                pickle.dump(data_store, file)            

if __name__ == '__main__':
    if args.output_type == 'pickle':
        data_to_pickle_file()
    else:
        folder_images_csv_labels()    
