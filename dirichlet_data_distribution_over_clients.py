#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 15:15:27 2022

@author: hussain
"""


import numpy as np
import torchvision.transforms as transforms
#from datasets import CIFAR10_truncated, MNIST_truncated
import sys
import pickle
import os
import torch.utils.data as data
import torchvision
from torchvision.datasets import CIFAR10, MNIST
#import numpy as np

class CIFAR10_truncated(data.Dataset):
    def __init__(
        self,
        root,
        dataidxs=None,
        train=True,
        transform=None,
        target_transform=None,
        download=False,
    ):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        cifar_dataobj = CIFAR10(self.root, self.train, self.transform, self.target_transform, self.download)
        if torchvision.__version__ == "0.2.1":
            if self.train:
                data, target = cifar_dataobj.train_data, np.array(cifar_dataobj.train_labels)
            else:
                data, target = cifar_dataobj.test_data, np.array(cifar_dataobj.test_labels)
        else:
            data = cifar_dataobj.data
            target = np.array(cifar_dataobj.targets)
        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]
        return data, target

    def truncate_channel(self, index):
        for i in range(index.shape[0]):
            gs_index = index[i]
            self.data[gs_index, :, :, 1] = 0.0
            self.data[gs_index, :, :, 2] = 0.0

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]
        if self.transform is not None:
            img1 = self.transform(img)
            img2 = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img1, img2, target, index

    def __len__(self):
        return len(self.data)

class MNIST_truncated():
    def __init__(self, data_path, dataidxs=None, train=True, transform=None, target_transform = None, download=False):
        self.data_path = data_path
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.data, self.target = self.__truncated_dataset__()
    def __len__(self):
        return len(self.data)
    def __truncated_dataset__(self):
        mnist_dataobj = MNIST(self.data_path, self.train, self.transform, self.target_transform, self.download)
        data = mnist_dataobj.data
        target = np.array(mnist_dataobj.targets)
        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]
        return data, target
    def __getitem__(self, index):
        img, target = self.data[index], self.target[index]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target, index

def dirichlet_data_distribution(dataset_name, data_path, number_of_nodes, partition_type, beta=0.5):
    if dataset_name == "mnist":
        X_train, y_train, X_test, y_test = load_mnist(data_path)
    elif dataset_name == "cifar10":
        X_train, y_train, X_test, y_test = load_cifar10(data_path)
    
    n_train = y_train.shape[0]
    if partition_type == "iid":
        idxs = np.random.permutation(n_train)
        batch_idxs = np.array_split(idxs, number_of_nodes)
        net_dataidx_map = {i: batch_idxs[i] for i in range(number_of_nodes)}
    elif partition_type == "noniid":
        min_size = 0
        min_require_size = 10
        K = 10
        N = y_train.shape[0]
        net_dataidx_map = {}

        while min_size < min_require_size:
            idx_batch = [[] for _ in range(number_of_nodes)]
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(beta, number_of_nodes))
                proportions = np.array([p * (len(idx_j) < N / number_of_nodes) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]                
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(number_of_nodes):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]
        
    traindata_cls_counts = get_net_data_stats(y_train, net_dataidx_map)
    return (X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts)
    
def get_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = {}
    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp

    data_list = []
    for net_id, data in net_cls_counts.items():
        n_total = 0
        for class_id, n_data in data.items():
            n_total += n_data
        data_list.append(n_total)
    print("mean:", np.mean(data_list))
    print("std:", np.std(data_list))
    print("Data statistics: %s" % str(net_cls_counts))
    return net_cls_counts    

def load_cifar10(data_path):
    transform = transforms.Compose([transforms.ToTensor()])
    cifar10_train_ds = CIFAR10_truncated(data_path, train=True, transform=transform, download=True)
    cifar10_test_ds = CIFAR10_truncated(data_path, train=False, transform=transform, download=True)
    X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.target
    X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.target
    return (X_train, y_train, X_test, y_test)

def load_mnist(data_path):
    transform = transforms.Compose([transforms.ToTensor()])
    cifar10_train_ds = MNIST_truncated(data_path, train=True, transform=transform, download=True)
    cifar10_test_ds = MNIST_truncated(data_path, train=False, transform=transform, download=True)
    X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.target
    X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.target
    return (X_train, y_train, X_test, y_test)

def data_to_pickle(data_path, dataset_name, X_train, y_train):
    # store data with pickle
    with open(os.path.join(data_path, dataset_name+'.pkl'), 'wb') as file:  
        data_store = {'data': X_train, 'label': y_train}
        pickle.dump(data_store, file)
    # show data from pickle
    with open(os.path.join(data_path, dataset_name+'.pkl'), 'rb') as file:
        data_store = pickle.load(file)
        print(data_store.keys())     
        print('data: ', data_store['data'].shape)
        print('label: ', data_store['label'].shape)
    
    

if __name__ == '__main__':
        
    sys.setrecursionlimit(100000)
#    print(sys.getrecursionlimit())    
    
    dataset_name = 'cifar10' # cifar10, mnist
    data_path = 'data/'
    number_of_nodes = 3
    partition_type = 'noniid' # iid, noniid
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = dirichlet_data_distribution(dataset_name, data_path, number_of_nodes, partition_type, beta=0.4)
    # save data obtained from Dirichlet distribution
    data_to_pickle(data_path, dataset_name, X_train, y_train)
    

    