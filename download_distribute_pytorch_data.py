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
from skimage import io
import pickle
import csv
from torchvision.utils import save_image

parser = argparse.ArgumentParser(description = 'Main Script')
parser.add_argument('--data_path', type = str, default = './data', help = 'Path to the main directory')
parser.add_argument('--dataset_name', type = str, default = 'usps', choices = ['pacs', 'cifar10', 'cifar100', 'mnist', 'svhn', 'mnist_m', 'usps', 'imagenet'], help = 'Name of dataset')
parser.add_argument('--number_of_classes', type = int, default = 10, choices = ['2', '10', '100', '1000'], help = 'Number of classes in dataset')
parser.add_argument('--image_height', type = int, default = 16, choices = ['227', '28', '32', '16', '224'], help = 'Height of each image in dataset')
parser.add_argument('--image_width', type = int, default = 16, choices = ['227', '28', '32', '16', '224'], help = 'Width of each image in dataset')
parser.add_argument('--image_channel', type = int, default = 1, help = 'Channel of a single image in dataset, i.e., 1, 3')
parser.add_argument('--transform', type=bool, default = False, help = 'True, False')
parser.add_argument('--number_of_clients', type = int, default = 1, help = 'Total nodes to which dataset is divided')
parser.add_argument('--distribution_method', type = str, default = 'iid', choices = ['iid, non_iid'], help = 'Type of data distribution')
parser.add_argument('--dirichlet_alpha', type = float, default = 0.5, help = 'Value of alpha for dirichlet distribution')
parser.add_argument('--imbalance_sigma', type = int, default = 0, help = '0 or otherwise')
parser.add_argument('--num_workers', type=int, default = 1, help='1, 4, 8, ...')
parser.add_argument('--download_type', type=str, default='class_wise_folders', choices=['images', 'pickle', 'class_wise_folders'])
args = parser.parse_args() 

def cifar10(transform):
    if transform is not None:
        transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    else:
        transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.CIFAR10(root=args.data_path,train=True , download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root=args.data_path,train=False, download=True, transform=transform)    
    train_batch = 50000
    test_batch = 10000
    return trainset, testset, train_batch, test_batch

def cifar100(transform):
    if transform is not None:
        transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762])])
    else:
        transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.CIFAR100(root=args.data_path,train=True , download=True, transform=transform)
    testset = torchvision.datasets.CIFAR100(root=args.data_path,train=False, download=True, transform=transform)    
    train_batch = 50000
    test_batch = 10000
    return trainset, testset, train_batch, test_batch

def mnist(transform):
    if transform is not None:
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    else:
        transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.MNIST(root=args.data_path, train=True , download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root=args.data_path, train=False, download=True, transform=transform)    
    train_batch = 60000
    test_batch = 10000
    return trainset, testset, train_batch, test_batch

def mnist_m(transform):
    if transform is not None:
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    else:
        transform = transforms.Compose([transforms.ToTensor()])
    data_path = os.path.join(args.data_path, 'mnist_m')
    train_data = os.path.join(data_path, 'mnist_m_train')
    test_data = os.path.join(data_path, 'mnist_m_test')
    train_labels = os.path.join(data_path, 'mnist_m_train_labels.txt')
    train_labels = open(train_labels,'r')
    train_labels = train_labels.readlines()
    train_x = []
    train_y = []
    print('Processing train images...')
    for i in range(len(train_labels)):
        image_label = train_labels[i].split()
        image_name = image_label[0]
        label = image_label[1]
        image = io.imread(os.path.join(train_data, image_name)) 
        train_x.append(transform(image))
        train_y.append(torch.tensor(int(label)))
    test_labels = os.path.join(data_path, 'mnist_m_test_labels.txt')
    test_labels = open(test_labels, 'r')
    test_labels = test_labels.readlines()
    test_x = []
    test_y = []
    print('Training images saved! \nNow processing test images...')
    for i in range(len(test_labels)):
        image_label = test_labels[i].split()
        image_name = image_label[0]
        label = image_label[1]
        image = io.imread(os.path.join(test_data, image_name)) 
        test_x.append(transform(image))
        test_y.append(torch.tensor(int(label)))
    
    train_x = torch.stack(train_x, dim=0)
    train_y = torch.stack(train_y, dim=0)
    test_x = torch.stack(test_x, dim=0)
    test_y = torch.stack(test_y, dim=0)
    
    print('train_x: ', type(train_x), train_x.shape)
    
    trainset = torch.utils.data.TensorDataset(train_x,train_y)
    testset = torch.utils.data.TensorDataset(test_x,test_y)
    train_batch = 59001
    test_batch = 9001
    return trainset, testset, train_batch, test_batch

def usps(transform):
    if transform is not None:
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1564,), (0.2566,))])
    else:
        transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.USPS(root=args.data_path, train=True , download=True, transform=transform)
    testset = torchvision.datasets.USPS(root=args.data_path, train=False, download=True, transform=transform)
    train_batch = 7291
    test_batch = 2007
    return trainset, testset, train_batch, test_batch

def svhn(transform):
    if transform is not None:
        transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.4377, 0.4438, 0.4728], std=[0.198 , 0.201 , 0.197])])
    else:
        transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.SVHN(root=args.data_path, split='train', download=True, transform=transform) # split = train, test, extra
    testset = torchvision.datasets.SVHN(root=args.data_path, split='test', download=True, transform=transform) # split = train, test, extra
    train_batch = 73257
    test_batch = 26032
    return trainset, testset, train_batch, test_batch

def imagenet(transform):
    if transform is not None:
        transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    trainset = torchvision.datasets.ImageNet(root=args.data_path, split='train', download=True, transform=transform) # split = train, test, extra
    testset = torchvision.datasets.SVHN(root=args.data_path, split='val', download=True, transform=transform) # split = train, test, extra
    train_batch = 73257
    test_batch = 26032
    return trainset, testset, train_batch, test_batch

def pacs(transform):
    transform = transforms.Compose([transforms.ToTensor()])
    raw_data = os.path.join(os.path.join(args.data_path, args.dataset_name), 'Raw images/kfold/')
    data_dir = os.path.join(os.path.join(args.data_path, args.dataset_name), 'train_val_split')
    train_labels = os.path.join(data_dir, 'photo_train_kfold.txt')
    train_labels = open(train_labels,'r')
    train_labels = train_labels.readlines()
    train_x = []
    train_y = []
    for i in range(len(train_labels)):
        image_label = train_labels[i].split()
        image_name = image_label[0]
        label = image_label[1]
        image = io.imread(os.path.join(raw_data, image_name)) 
        train_x.append(transform(image))
        train_y.append(torch.tensor(int(label)))
        
    test_labels = os.path.join(data_dir, 'cartoon_test_kfold.txt')
    test_labels = open(test_labels,'r')
    test_labels = test_labels.readlines()
    test_x = []
    test_y = []
    for i in range(len(test_labels)):
        image_label = test_labels[i].split()
        image_name = image_label[0]
        label = image_label[1]
        image = io.imread(os.path.join(raw_data, image_name)) 
        test_x.append(transform(image))
        test_y.append(torch.tensor(int(label)))
    train_x = torch.stack(train_x, dim=0)
    train_y = torch.stack(train_y, dim=0)
    test_x = torch.stack(test_x, dim=0)
    test_y = torch.stack(test_y, dim=0)
    trainset = torch.utils.data.TensorDataset(train_x,train_y)
    testset = torch.utils.data.TensorDataset(test_x,test_y)
    train_batch = 1499
    test_batch = 2344
    return trainset, testset, train_batch, test_batch

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

def data_download():
    transform = None
    if args.transform == True:
        transform = True
    if args.dataset_name == 'cifar10':
        trainset, testset, train_batch, test_batch = cifar10(transform)
    if args.dataset_name == 'cifar100':
        trainset, testset, train_batch, test_batch = cifar100(transform)        
    if args.dataset_name == 'mnist':
        trainset, testset, train_batch, test_batch = mnist(transform)
    if args.dataset_name == 'mnist_m':
        trainset, testset, train_batch, test_batch = mnist_m(transform)
    if args.dataset_name == 'usps':
        trainset, testset, train_batch, test_batch = usps(transform)
    if args.dataset_name == 'svhn':
        trainset, testset, train_batch, test_batch = svhn(transform)
    if args.dataset_name == 'imagenet':
        trainset, testset, train_batch, test_batch = imagenet(transform)
    if args.dataset_name == 'pacs':
        trainset, testset, train_batch, test_batch = pacs(transform)
    
    trainload = torch.utils.data.DataLoader(trainset, batch_size=train_batch, shuffle=False, num_workers=args.num_workers)
    testload = torch.utils.data.DataLoader(testset, batch_size=test_batch, shuffle=False, num_workers=args.num_workers)       
    print('<============= Data loaded, and distribution process started! ================>')
    # iterate over whole data
    train_iteration = trainload.__iter__(); 
    test_iteration = testload.__iter__() 
    X_train, y_train = train_iteration.__next__()  # <class 'torch.Tensor'>
    X_test, y_test = test_iteration.__next__()
    if args.download_type in 'pickle':
        data_to_clients_pickle(X_train, y_train, X_test, y_test)
    if args.download_type in 'class_wise_folders':
        class_wise_distribution(X_train, y_train)
    else:
        folder_images_csv_labels(X_train, y_train, X_test, y_test)

def folder_images_csv_labels(X_train, y_train, X_test, y_test):
    # training data 
    os.makedirs(os.path.join(args.data_path, args.dataset_name+'_train'), exist_ok = True)
    data_store_path = os.path.join(args.data_path, args.dataset_name+'_train')  
    csv_file = open(os.path.join(args.data_path, args.dataset_name+'_train.csv'), 'w', newline='')  
    writer = csv.writer(csv_file)
    for i in range(y_train.shape[0]):
        save_image(X_train[i], os.path.join(data_store_path, str(i)+'.png'))
        writer.writerow([str(i)+'.png', y_train[i].item()])
    csv_file.close()
    # test data
    os.makedirs(os.path.join(args.data_path, args.dataset_name+'_test'), exist_ok = True)
    data_store_path = os.path.join(args.data_path, args.dataset_name+'_test')  
    csv_file = open(os.path.join(args.data_path, args.dataset_name+'_test.csv'), 'w', newline='')  
    writer = csv.writer(csv_file)
    for i in range(y_test.shape[0]):        
        save_image(X_test[i], os.path.join(data_store_path, str(i)+'.png'))
        writer.writerow([str(i)+'.png', y_test[i].item()])
    csv_file.close()
    print('Data download at: ', args.data_path)
    
def class_wise_distribution(X_train, y_train):
    os.makedirs(os.path.join(args.data_path, args.dataset_name+'_train'), exist_ok = True)
    data_store_path = os.path.join(args.data_path, args.dataset_name+'_train')  
    for i in range(y_train.shape[0]):
        os.makedirs(os.path.join(data_store_path, str(y_train[i].item())), exist_ok = True)
        image_path = os.path.join(data_store_path, str(y_train[i].item()))
        save_image(X_train[i], os.path.join(image_path, str(i)+'.png'))

def data_to_clients_pickle(X_train, y_train, X_test, y_test):
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
    
    # if you want to save a single pickle file
    with open(os.path.join(args.data_path, os.path.join(file_path, args.dataset_name+'_train_test.pkl')), 'wb') as file:  
            data_store = {'X_train': X_train, 'y_train': y_train, 'X_test': X_test, 'y_test': y_test}
            pickle.dump(data_store, file)
    print('Data saved on the location: ', os.path.join(args.data_path, file_path))    
            
        
if __name__ == '__main__':    
    data_download()  
