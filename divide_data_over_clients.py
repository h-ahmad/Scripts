#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 12:05:21 2022

@author: hussain
"""


import os
import pickle
import cv2
import csv

def main():
    data_path = 'data/'
    dataset_name = 'cifar10'
    number_of_clients = 3
    with open(os.path.join(data_path, dataset_name+'.pkl'), 'rb') as file:
        data_store = pickle.load(file)
        print(data_store.keys())     
        print('data: ', data_store['data'].shape)
        print('label: ', data_store['label'].shape)
    total_data_count = data_store['label'].shape[0]
    client_data = int(total_data_count/number_of_clients)
    j = 0
    for i in range(total_data_count):
        if i % client_data == 0:
            j = j + 1
            os.makedirs(os.path.join(data_path, 'client'+str(j)+'/'), exist_ok = True)
            data_store_path = os.path.join(data_path, 'client'+str(j)+'/')   
            csv_file = file = open(os.path.join(data_path, str(j)+'.csv'), 'w', newline='')
            writer = csv.writer(csv_file)            
        cv2.imwrite(os.path.join(data_store_path, str(i)+'.png'), data_store['data'][i])                
        writer.writerow([str(i), data_store['label'][i]])
    csv_file.close()

if __name__ == '__main__':
  main()