#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 16:36:59 2022

@author: hussain
"""


import os
import pickle
import cv2
import csv

def main():
    data_path = 'data/'
    dataset_name = 'cifar10'  # (cifar10, mnist) pickle file name
    number_of_clients = 3
    with open(os.path.join(data_path, dataset_name+'.pkl'), 'rb') as file:
        data_store = pickle.load(file)
        print(data_store.keys())     
        print('X_train: ', data_store['X_train'].shape)
        print('y_train: ', data_store['y_train'].shape)
        print('X_test: ', data_store['X_test'].shape)
        print('y_test: ', data_store['y_test'].shape)        
    total_data_count = data_store['y_train'].shape[0]
    client_data = int(total_data_count/number_of_clients)
    # save multiple train data for multiple clients given
    j = 0
    for i in range(total_data_count):
        if i % client_data == 0:
            j = j + 1
            os.makedirs(os.path.join(data_path, 'client'+str(j)+'/'), exist_ok = True)
            data_store_path = os.path.join(data_path, 'client'+str(j)+'/')   
            csv_file = open(os.path.join(data_path, str(j)+'.csv'), 'w', newline='')
            writer = csv.writer(csv_file)      
        image = data_store['X_train'][i]  # for numpy . i.e CIFAR10
#        image = data_store['X_train'][i].numpy()  # for tensor . i.e MNIST
        cv2.imwrite(os.path.join(data_store_path, str(i)+'.png'), image)                
        writer.writerow([str(i), data_store['y_train'][i]])
    csv_file.close()
    # save a single test data for all clients
    csv_file = open(os.path.join(data_path, 'test.csv'), 'w', newline='')
    writer = csv.writer(csv_file)     
    os.makedirs(os.path.join(data_path, 'test'), exist_ok = True)
    data_store_path = os.path.join(data_path, 'test')
    csv_file = open(os.path.join(data_path, 'test.csv'), 'w', newline='')
    writer = csv.writer(csv_file)    
    for k in range(data_store['y_test'].shape[0]):
        image = data_store['X_test'][k]  # for numpy . i.e CIFAR10
#        image = data_store['X_test'][k].numpy()  # for tensor . i.e MNIST
        cv2.imwrite(os.path.join(data_store_path, str(k)+'.png'), image)
        writer.writerow([str(k), data_store['y_test'][k]])
    csv_file.close()

if __name__ == '__main__':
  main()