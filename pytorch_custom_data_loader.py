# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 11:33:19 2022

@author: hussain
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from skimage import io, transform
import numpy as np
import torchvision.transforms as transforms
from sklearn.preprocessing import OneHotEncoder

#### Loader for pickle file ###########
class FLDataset(Dataset):
    def __init__(self, args, client, phase, client_index = None):
        super(FLDataset, self).__init__()
        self.phase = phase

        if self.phase == 'train':
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop((args.resize, args.resize), scale=(0.05, 1.0)),
                # transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((args.resize, args.resize)),
                # transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
            
        
        data_file = os.path.join(os.path.join(args.data_path, client), client+'_train_test.pkl')
        with open(data_file, 'rb') as file:
                data_store = pickle.load(file)
        train_x, train_y, test_x, test_y = data_store['X_train'], data_store['y_train'], data_store['X_test'], data_store['y_test'] 
        train_x, train_y, test_x, test_y = map(torch.tensor, (train_x.astype(np.float32), train_y.astype(np.int_), 
                                                          test_x.astype(np.float32), test_y.astype(np.int_))) 
        
        if self.phase == 'train':
            self.label = train_y.type(torch.LongTensor)
            if client_index is not None:
                self.data = train_x[client_index]
                self.label = self.label[client_index]
            if (phase == 'train' and self.data.shape[1] == 1):
                self.data = torch.cat((self.data, self.data, self.data), dim=1)
        else:
            self.label = test_y.type(torch.LongTensor)
            if test_x.shape[1] == 1:
                self.data = torch.cat((test_x, test_x, test_x), dim=1)
            else:
                self.data = test_x
        # img, target = self.data, self.label
    def __getitem__(self, index):
        img, target = self.data[index], self.label[index]
        if self.transform is not None:
            img = self.transform(img)        
        return img,  target
    def __len__(self):
        return len(self.data)

class Cifar10Loader(Dataset):
    def __init__(self, csv_path, dataset_path, transform=None):
        self.image_names = pd.read_csv(csv_path)
        self.data_path = dataset_path
        self.transform = transform
    def __len__(self):
        return len(self.image_names)
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        img_name = os.path.join(self.data_path, self.image_names.iloc[index, 0])
        img = io.imread(img_name) 
        label = self.image_names.iloc[index, 1]
        label = torch.tensor(label)
        if self.transform:
            img = self.transform(img)
        return img, label 

class Cifar10LoaderTrain(Dataset):
    def __init__(self, csv_path, dataset_path, transform=None):
        self.image_names = pd.read_csv(csv_path)
        self.data_path = dataset_path
        self.transform = transform
    def __len__(self):
        return len(self.image_names)
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        img_name = os.path.join(self.data_path, self.image_names.iloc[index, 0])
        img = io.imread(img_name)   
        label = self.image_names.iloc[index, 1]
        if label == 0: label = torch.tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        elif label == 1: label = torch.tensor([0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        elif label == 2: label = torch.tensor([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
        elif label == 3: label = torch.tensor([0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
        elif label == 4: label = torch.tensor([0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
        elif label == 5: label = torch.tensor([0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
        elif label == 6: label = torch.tensor([0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
        elif label == 7: label = torch.tensor([0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
        elif label == 8: label = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
        elif label == 9: label = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0,1])
        if self.transform:
            img = self.transform(img)
        return img, label
   

if __name__ == '__main__':
  directory = r'C:\Users\hussain\.spyder-py3\data\cifar1'
  csv_path = os.path.join(directory, 'labels1.csv')
  data_path = os.path.join(directory, 'train1')  
  transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
  trainset = Cifar10Loader(csv_path, data_path, transform)
  # trainset = Cifar10LoaderTrain(csv_path, data_path, transform)
  trainLoader = torch.utils.data.DataLoader(trainset,batch_size=4)  
  for index, (data, target) in enumerate(trainLoader):
      print('This is input: ', data.shape)
      print('this is label: ', target.shape)
      break
        
