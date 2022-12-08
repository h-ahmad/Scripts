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

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.model_name == 'cnn2':
        model = Net().to(device)
    if args.model_name == 'resnet18':
        model = resnet18(pretrained = True).to(device)    
    train_loader, test_loader = load_data()
    optimizer = optim.Adam(model.parameters())
    loss_fn = torch.nn.CrossEntropyLoss()
# =============================================================================
#     print('<============ Training started! =============>')
#     for epoch in range(1, args.epochs + 1):
#         model = train(train_loader, optimizer, model, loss_fn)
#         accuracy = test(test_loader, loss_fn)
#         print(f'Epoch: {epoch}, Accuracy: {accuracy}')
# =============================================================================        
        
    # federated learning 
    
    def gen_pk():
        HE = Pyfhel()  
        HE.contextGen(scheme='bfv', t=214047853, n=2**14, t_bits=20)
        HE.keyGen()        
        keys ={}
        keys['HE'] = HE
        keys['con'] = HE.to_bytes_context()
        keys['pk'] = HE.to_bytes_public_key()        
        filename =  "publickey.pickle"
        with open(filename, 'wb') as handle:
            pickle.dump(keys, handle, protocol=pickle.HIGHEST_PROTOCOL)    
        return HE
    
    def get_pk():
        filename =  "publickey.pickle"
        with open(filename, 'rb') as handle:
            key = pickle.load(handle)    
        HE = key['HE']
        HE.from_bytes_context(key['con'])
        HE.from_bytes_public_key(key['pk'])
        return HE
        
    def train_clients(num_clients):
        for i in range(num_clients):   
            train_loader, test_loader = load_data()
            for epoch in range(1, args.epochs + 1):
                client_model = train(train_loader, optimizer, model, loss_fn)
            os.makedirs('client_models', exist_ok=True)
            saved_model_path = 'client_models', str(i)+'.pt'
            # torch.save(client_model, saved_model_path)
            encrypt_gradients(client_model, i)
            
    def encrypt_gradients(client_model, client_number):
        # HE = get_pk()
        HE = Pyfhel()  
        HE.contextGen(scheme='bfv', t=214047853, n=2**14, t_bits=20)
        HE.keyGen() 
        encrypted_weights={}
        layers_list = [module for module in client_model.modules() if not isinstance(module, nn.Sequential)]
        for i in range(len(layers_list)):
            if hasattr(layers_list[i], 'weight') and type(layers_list[i]) != torch.nn.modules.batchnorm.BatchNorm2d:                
                weights = layers_list[i].weight
                for j in range(len(weights)):
                    weight = np.asarray(weights[j].detach().cpu())
                    shape = weight.shape
                    weight = weight.flatten()
                    array= np.empty(len(weight),dtype=PyCtxt)
                    enc_array = [None] * len(weight)
                    for k in np.arange(len(weight)):
                        # weight = np.asarray([weight[k]], dtype = np.double)
                        array[k] = HE.encrypt(weight[k])
                    enc_array = array.reshape(shape)
                    encrypted_weights['c_'+str(i)+'_'+str(j)]=enc_array        
        filename =  "client_" + str(client_number+1)+ ".pickle"
        dic = {}
        dic['key'] = HE
        dic['val'] = encrypted_weights
        with open(filename, 'wb') as handle:
            pickle.dump(dic, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
    def aggregate_encrypted_weights(num_client):
        dct_weights ={}
        denom = float(1/num_client)
        for i in range(num_client):
            enc_weights={}
            filename =  "client_" + str(i+1)+ ".pickle"
            with open(filename, 'rb') as handle:
                dct = pickle.load(handle)        
            cweights=dct['val']
            HE2 = dct['key']
            enc_weights={}  
            for key in cweights:
                arr = cweights[key]
                shape = arr.shape
                weight = arr.flatten()        
                for l in np.arange(len(weight)):
                    weight[l]._pyfhel = HE2        
                enc_array = weight.reshape(shape)
                enc_weights[key] = enc_array
            for key in enc_weights:
                if i == 0:
                    arr = enc_weights[key]
                    dct_weights[key] = np.zeros_like(arr,dtype=PyCtxt)
                dct_weights[key] = enc_weights[key] + dct_weights[key]     
        for key in dct_weights:
            dct_weights[key]= dct_weights[key]*denom #c_denom
        return dct_weights
            
                
    num_of_client_list = [1]
    gen_pk() # Generate public key
    for j, num_client in enumerate(num_of_client_list):
        # Train clients and save encrypted weights
        train_clients(num_client)        
        # print("Aggregate Weights")
        # main_model_dict = aggregate_encrypted_weights(num_client)
        # filename="weights/aggregated.pickle"
        # print("Export Aggregate Weights")
        
        # HE = get_pk()
        # dic = {}
        # dic['key']=HE
        # dic['val']=encrypted_weights
        # with open(filename, 'wb') as handle:
        #     pickle.dump(dic, handle, protocol=pickle.HIGHEST_PROTOCOL)
                
        # agg_model = decrypt_import_weights(filename)
