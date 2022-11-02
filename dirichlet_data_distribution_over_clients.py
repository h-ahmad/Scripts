#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 13:47:56 2022

@author: hussain
"""

import argparse
import torchvision.transforms as transforms
import torch
import torchvision
import numpy as np
import os

parser = argparse.ArgumentParser(description = 'Main Script')
parser.add_argument('--data_path', type = str, default = './data', help = 'Path to the main directory')
parser.add_argument('--dataset_name', type = str, default = 'mnist', help = 'cifar10, mnist')
parser.add_argument('--number_of_classes', type = int, default = 10, help = 'Number of classes in the given dataset')
parser.add_argument('--image_height', type = int, default = 28, help = 'Height of a single image in dataset')
parser.add_argument('--image_width', type = int, default = 28, help = 'Width of a single image in dataset')
parser.add_argument('--image_channel', type = int, default = 1, help = 'Channel of a single image in dataset')
parser.add_argument('--number_of_clients', type = int, default = 4, help = 'Total nodes to which dataset is divided')
parser.add_argument('--distribution_method', type = str, default = 'non_iid', help = 'iid, non_iid')
parser.add_argument('--dirichlet_alpha', type = float, default = 0.5, help = 'Value of alpha for dirichlet distribution')
parser.add_argument('--imbalance_sigma', type = int, default = 0, help = '0 or otherwise')
args = parser.parse_args() 

def cifar10():
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])])
    trainset = torchvision.datasets.CIFAR10(root=args.data_path,
                                          train=True , download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root=args.data_path,
                                          train=False, download=True, transform=transform)    
    trainload = torch.utils.data.DataLoader(trainset, batch_size=50000, shuffle=False, num_workers=1)
    testload = torch.utils.data.DataLoader(testset, batch_size=10000, shuffle=False, num_workers=1)
    return trainload, testload
def mnist():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    trainset = torchvision.datasets.MNIST(root=args.data_path, 
                                        train=True , download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root=args.data_path, 
                                        train=False, download=True, transform=transform)    
    trainload = torch.utils.data.DataLoader(trainset, batch_size=60000, shuffle=False, num_workers=1)
    testload = torch.utils.data.DataLoader(testset, batch_size=10000, shuffle=False, num_workers=1)
    return trainload, testload

def imbalance(samples_per_client, y_train):
    if args.imbalance_sigma != 0:
        client_data_list = (np.random.lognormal(mean=np.log(samples_per_client), sigma=args.imbalance_sigma, size=args.number_of_clients))
        client_data_list = (client_data_list/np.sum(client_data_list)*len(y_train)).astype(int)
        diff = np.sum(client_data_list) - len(y_train)
        # Add/Subtract the excess number starting from first client
        if diff!= 0:
            for client_i in range(args.number_of_clients):
                if client_data_list[client_i] > diff:
                    client_data_list[client_i] -= diff
                    break
    else:
        client_data_list = (np.ones(args.number_of_clients) * samples_per_client).astype(int)
    return client_data_list

def dirichlet_distribution(client_data_list, X_train, y_train):
    class_priors   = np.random.dirichlet(alpha=[args.dirichlet_alpha]*args.number_of_classes,size=args.number_of_clients) # <class 'numpy.ndarray'>  (4, 10)
    prior_cumsum = np.cumsum(class_priors, axis=1) # <class 'numpy.ndarray'>  (4, 10)
    idx_list = [np.where(y_train==i)[0] for i in range(args.number_of_classes)] # <class 'list'>
    class_amount = [len(idx_list[i]) for i in range(args.number_of_classes)] # <class 'list'>  0=>50000
    client_x = [ np.zeros((client_data_list[clnt__], args.image_channel, args.image_height, args.image_width)).astype(np.float32) for clnt__ in range(args.number_of_clients) ] # <class 'list'>                
    client_y = [ np.zeros((client_data_list[clnt__], 1)).astype(np.int64) for clnt__ in range(args.number_of_clients) ] # <class 'list'>                
    while(np.sum(client_data_list)!=0):
        current_client = np.random.randint(args.number_of_clients)
        # If current node is full resample a client
        # print('Remaining Data: %d' %np.sum(client_data_list))
        if client_data_list[current_client] <= 0:
            continue
        client_data_list[current_client] -= 1
        curr_prior = prior_cumsum[current_client]
        while True:
            cls_label = np.argmax(np.random.uniform() <= curr_prior)
            # Redraw class label if trn_y is out of that class
            if class_amount[cls_label] <= 0:
                continue
            class_amount[cls_label] -= 1
            client_x[current_client][client_data_list[current_client]] = X_train[idx_list[cls_label][class_amount[cls_label]]]
            client_y[current_client][client_data_list[current_client]] = y_train[idx_list[cls_label][class_amount[cls_label]]]
            break                
    client_x = np.asarray(client_x)  # (4, 12500, 1)
    client_y = np.asarray(client_y)  #(4, 12500, 1)          
    cls_means = np.zeros((args.number_of_clients, args.number_of_classes))
    for clnt in range(args.number_of_clients):
        for cls in range(args.number_of_classes):
            cls_means[clnt,cls] = np.mean(client_y[clnt]==cls)
    prior_real_diff = np.abs(cls_means-class_priors)
    print('--- Max deviation from prior: %.4f' %np.max(prior_real_diff))
    print('--- Min deviation from prior: %.4f' %np.min(prior_real_diff))
    return (client_x, client_y)

def independent_identical_data(client_data_list, X_train, y_train):
    client_x = [ np.zeros((client_data_list[clnt__], args.image_channel, args.image_height, args.image_width)).astype(np.float32) for clnt__ in range(args.number_of_clients) ]
    client_y = [ np.zeros((client_data_list[clnt__], 1)).astype(np.int64) for clnt__ in range(args.number_of_clients) ]    
    clnt_data_list_cum_sum = np.concatenate(([0], np.cumsum(client_data_list)))
    for clnt_idx_ in range(args.number_of_clients):
        client_x[clnt_idx_] = X_train[clnt_data_list_cum_sum[clnt_idx_]:clnt_data_list_cum_sum[clnt_idx_+1]]
        client_y[clnt_idx_] = y_train[clnt_data_list_cum_sum[clnt_idx_]:clnt_data_list_cum_sum[clnt_idx_+1]]        
    client_x = np.asarray(client_x)
    client_y = np.asarray(client_y)
    return (client_x, client_y)

def data_distribution():
    if args.dataset_name == 'cifar10':
        trainload, testload = cifar10()
    if args.dataset_name == 'mnist':
        trainload, testload = mnist()
        
    # iterate over whole data
    train_iteration = trainload.__iter__(); 
    test_iteration = testload.__iter__() 
    X_train, y_train = train_iteration.__next__()  # <class 'torch.Tensor'>
    X_test, y_test = test_iteration.__next__()
    
    # convert tensor to numpy array and reshape
    X_train = X_train.numpy();   # <class 'numpy.ndarray'>
    y_train = y_train.numpy().reshape(-1,1)
    X_test = X_test.numpy(); 
    y_test = y_test.numpy().reshape(-1,1)
    
    # shuffle data
    random_permutation = np.random.permutation(len(y_train))
    X_train = X_train[random_permutation]  # <class 'numpy.ndarray'>
    y_train = y_train[random_permutation]
    
    # count samples per client
    samples_per_client = int((len(y_train)) / args.number_of_clients)
    
    # imbalance if set
    client_data_list = imbalance(samples_per_client, y_train)
        
    if args.distribution_method == 'non_iid':
        X_train, y_train = dirichlet_distribution(client_data_list, X_train, y_train)
    elif args.distribution_method == 'iid':  
        X_train, y_train = independent_identical_data(client_data_list, X_train, y_train)
        
    # Save data in the same directory with a name specified by attributes
    file_path = args.dataset_name+'_'+str(args.number_of_clients)+'clients_'+args.distribution_method+'_alpha'+str(args.dirichlet_alpha)+'/'
    os.makedirs(os.path.join(args.data_path, file_path), exist_ok = True)
    np.save(os.path.join(args.data_path, os.path.join(file_path, 'X_train.npy')), X_train)
    np.save(os.path.join(args.data_path, os.path.join(file_path, 'y_train.npy')), y_train)
    np.save(os.path.join(args.data_path, os.path.join(file_path, 'X_test.npy')), X_test)
    np.save(os.path.join(args.data_path, os.path.join(file_path, 'y_test.npy')), y_test)
    print('Data saved on the location: ', os.path.join(args.data_path, file_path))
            
        
if __name__ == '__main__':    
    data_distribution()
