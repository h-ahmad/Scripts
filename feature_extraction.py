#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 09:22:04 2022

@author: hussain
"""

import argparse
from torchvision.models.resnet import resnet18
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor
import torch
import os
import pickle
import numpy as np

parser = argparse.ArgumentParser(description = 'Main Script')
parser.add_argument('--data_path', type = str, default = './data', help = 'Main path to the dataset')
parser.add_argument('--data_file_name', type = str, default = 'cifar10_train_test.pkl', help = 'Pickle file name')
parser.add_argument('--batch_size', type = int, default = 50, help = 'Batch size. i.e 1 to any number')
args = parser.parse_args() 

def load_data():
    with open(os.path.join(args.data_path, args.data_file_name), 'rb') as file:
        data_store = pickle.load(file)
        print(data_store.keys())    
        X_train, y_train, X_test, y_test = data_store['X_train'], data_store['y_train'], data_store['X_test'], data_store['y_test']
    
    # if 4 dimensions (4-th dimension is for a client)
    node_number = 3
    X_train = X_train[node_number]
    y_train = y_train[node_number].squeeze(1)
    y_test = y_test.squeeze(1)
    
    # data is numpy array, convert to tensor
    X_train = torch.from_numpy(X_train)
    y_train = torch.from_numpy(y_train)
    
    return (X_train, y_train, X_test, y_test)

def feature_extractor(test_input, device):
    model = resnet18(pretrained=True).to(device)
    model.eval()
    nodes, eval_nodes = get_graph_node_names(model)
    # print('model nodes:', nodes)
    features_ext = create_feature_extractor(model, return_nodes=['avgpool']) # avgpool, flatten...
    output_feature = features_ext(test_input.to(device))['avgpool']
    return output_feature

def data_to_pickle(pickle_file_name, X_train, y_train, X_test, y_test):    
    with open(os.path.join(args.data_path, pickle_file_name), 'wb') as file:  
        data_store = {'X_train': X_train, 'y_train': y_train, 'X_test': X_test, 'y_test': y_test}
        pickle.dump(data_store, file)

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X_train, y_train, X_test, y_test = load_data()    
    
    trainset = torch.utils.data.TensorDataset(X_train, y_train)    
    trainload = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers=0) 
    
    trainX = []
    trainY = []
    for batchIndex, (data, target) in enumerate(trainload):
        output_feature = feature_extractor(data, device)
        
        output_feature = output_feature.squeeze(2).squeeze(2)
        output_feature = output_feature.detach().cpu().numpy()
        trainX.append(output_feature) 
        trainY.append(target.numpy()) 
    trainX = np.array(trainX, dtype=np.float32)
    trainY = np.array(trainY, dtype=np.int_)
    data_to_pickle('features_client4_'+args.data_file_name, trainX, trainY, X_test, y_test)
