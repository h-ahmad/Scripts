#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 16:26:40 2022

@author: hussain
"""

from Pyfhel import Pyfhel, PyCtxt
import numpy as np
import os
import torch 
import torch.nn as nn
import torch.optim as optim
from torchvision.models.resnet import resnet18
import pickle
import argparse
import sys

parser = argparse.ArgumentParser(description = 'Main Script')
parser.add_argument('--data_path', type = str, default = './data', help = 'Main path to the dataset')
parser.add_argument('--data_file_name', type = str, default = 'cifar10_data.pkl', help = 'Dataset file')
parser.add_argument('--model_name', type = str, default = 'cnn2', help = 'cnn2, resnet18')
parser.add_argument('--epochs', type = int, default = 5, help = 'Number of epochs for each local model training')
parser.add_argument('--batch_size', type = int, default = 128, help = 'Batch size for each local data and model')
parser.add_argument('--client_model_path', type = str, default = 'client_models', help = 'Folder to save client individual models by their number/index')
args = parser.parse_args() 

class Net(nn.Module): # CNN2
    def __init__(self):
        super(Net, self).__init__()
        self.activation = nn.ReLU(True)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(5, 5), padding=1, stride=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32 * 2, kernel_size=(5, 5), padding=1, stride=1, bias=False)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=(32 * 2) * (8 * 8), out_features=512, bias=False)
        self.fc2 = nn.Linear(in_features=512, out_features=10, bias=False)

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.maxpool1(x)
        x = self.activation(self.conv2(x))
        x = self.maxpool2(x)
        x = self.flatten(x)
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x

def load_data():
    with open(os.path.join(args.data_path, args.data_file_name), 'rb') as file:
            data_store = pickle.load(file)
    xTrain, yTrain, xTest, yTest = data_store['X_train'], data_store['y_train'], data_store['X_test'], data_store['y_test']    
    xTrain, yTrain, xTest, yTest = map(torch.tensor, (xTrain.astype(np.float32), yTrain.astype(np.int_), 
                                                      xTest.astype(np.float32), yTest.astype(np.int_))) 
    # print('xTrain: ', xTrain.shape)
    # print('yTrain: ', yTrain.shape)
    # print('xTest: ', xTest.shape)
    # print('yTest: ', yTest.shape)
    yTrain = yTrain.type(torch.LongTensor)
    yTest = yTest.type(torch.LongTensor)
    trainDs = torch.utils.data.TensorDataset(xTrain,yTrain)
    testDs = torch.utils.data.TensorDataset(xTest,yTest)
    trainLoader = torch.utils.data.DataLoader(trainDs,batch_size=args.batch_size)
    testLoader = torch.utils.data.DataLoader(testDs,batch_size=args.batch_size)
    return trainLoader, testLoader 

def train(train_loader, optimizer, model, loss_fn):
    training_loss = 0.0
    model.train()
    test = []
    for index, (data, target) in enumerate(train_loader):
        test.append(index)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        training_loss += loss.item()
        # if index % 100 == 99:    # print every 100 mini-batches
        #     print(f'batch loss: {training_loss / 100:.3f}')
        #     training_loss = 0.0
    return model

def test(test_loader, loss_fn):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0) # add batches count
            correct += (predicted == target).sum().item() # integer value of correct count            
    return (100 * correct // total)

def train_clients(num_client):
    train_loader, test_loader = load_data()
    for epoch in range(1, args.epochs + 1):
        client_model = train(train_loader, optimizer, model, loss_fn)
    os.makedirs(args.client_model_path, exist_ok=True)
    saved_model_path = os.path.join(args.client_model_path, str(num_client+1)+'.pt')
    torch.save(client_model, saved_model_path)        
        
def aggregate_models(num_of_clients):
    model_list = []    
    main_model_state = {}
    for num_client in range(num_of_clients):
        client_model_path = os.path.join(args.client_model_path, str(num_client+1)+'.pt')
        client_model_state = torch.load(client_model_path).state_dict()
        model_list.append(client_model_state)
    
    for layer in model_list[0]: # Get layers of any model
        weight_avg = 0
        for model_state in model_list:
            weight_avg += model_state[layer]            
        main_model_state[layer] = weight_avg/num_of_clients
    return main_model_state

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.model_name == 'cnn2':
        model = Net().to(device)
    if args.model_name == 'resnet18':
        model = resnet18(pretrained = True).to(device)    
    train_loader, test_loader = load_data()
    optimizer = optim.Adam(model.parameters())
    loss_fn = torch.nn.CrossEntropyLoss()  
    # Federated learning...........
    num_of_clients = 3
    for num_client in range(num_of_clients):        
        train_clients(num_client)
    main_model_dict = aggregate_models(num_of_clients)
