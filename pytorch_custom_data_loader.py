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
  import torchvision.transforms as transforms
  transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
  trainset = Cifar10Loader(csv_path, data_path, transform)
  trainLoader = torch.utils.data.DataLoader(trainset,batch_size=4)
  from sklearn.preprocessing import OneHotEncoder
  for index, (data, target) in enumerate(trainLoader):
      print('This is input: ', data.shape)
      print('this is label: ', target.shape)
      break
        
