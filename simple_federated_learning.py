#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 16:41:25 2023

@author: hussain
"""

import argparse
import os
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from copy import deepcopy
import torch
import torchvision.models as torch_models
import torchvision.transforms as transforms
import pandas as pd
from skimage import io

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default='./data/', help="Location of dataset")
    parser.add_argument("--datasets", type=list, default = ['mnist', 'mnist_m', 'svhn', 'usps'], choices=['mnist', 'mnist_m', 'svhn', 'usps'], help="List of datasets")
    parser.add_argument('--num_classes', type = int, default = 10, choices = [2, 10], help = 'Number of classes in dataset')
    parser.add_argument('--model', type=str, default = 'simple', choices=['cnn2', 'lenet1', 'resnet18', 'simple'], help='Choose model')
    parser.add_argument('--Pretrained', action='store_true', default=True, help="Whether use pretrained or not")
    parser.add_argument("--optimizer_type", default="sgd",choices=["sgd", "adamw"], type=str, help="Type of optimizer")
    parser.add_argument("--num_workers", default=4, type=int, help="num_workers")
    parser.add_argument("--learning_rate", default=0.01, type=float,  help="The initial learning rate for SGD. Set to [3e-3] for ViT-CWT")
    parser.add_argument("--weight_decay", default=0, choices=[0.05, 0], type=float, help="Weight deay if we apply. E.g., 0 for SGD and 0.05 for AdamW")
    parser.add_argument("--batch_size", default=128, type=int,  help="Local batch size for training")
    parser.add_argument("--epoch", default=50, type=int, help="Local training epochs in FL")
    parser.add_argument("--communication_rounds", default=100, type=int,  help="Total communication rounds")
    parser.add_argument("--gpu_ids", type=str, default='0', help="gpu ids: e.g. 0,1,2")
    parser.add_argument('--output_path', type=str, default='', help='path to store evaluation metrics')
    args = parser.parse_args()
    
    args.device = torch.device("cuda:{gpu_id}".format(gpu_id = args.gpu_ids) if torch.cuda.is_available() else "cpu")
    if args.model in 'cnn2':
        model = CNN2Model()
    if args.model in 'lenet1':
        model = LeNetModel()
    if args.model == 'simple':
        model = Net()
    if args.model == 'resnet18':
        model = torch_models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.weight.shape[1], args.num_classes)
    model.to(args.device)
    
    model_avg = deepcopy(model).cpu()
    loss_fn = torch.nn.CrossEntropyLoss()
    
    model_all = {}
    # optimizer_all = {}
    best_global_accuracy = {}
    best_local_accuracy = {}
    best_global_loss = {}
    best_local_loss = {}
    for index, client in enumerate(args.datasets):
        model_all[index] = deepcopy(model).cpu()
        # optimizer_all[index] = optimizer
        best_global_accuracy[client] = 0
        best_local_accuracy[client] = 0
        best_global_loss[client] = 99
        best_local_loss[client] = 99
    with open(os.path.join(args.output_path, "training.log"), "w") as logger:
        for comm_round in range(args.communication_rounds):
            for index, client in enumerate(args.datasets):
                print('Training client ', (index+1), ' having data ', client,' for communication round ', (comm_round+1))    
                logger.write(f"Communication round: {comm_round}            Client: {index}: {client}"+"\n \n")
                train_loader = load_data(args, client, phase = 'train')
                model = model_all[index].to(args.device)
                if args.model in 'resnet':
                    model.eval()
                else:
                    model.train()
                
                if args.optimizer_type == 'sgd':
                    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.5, weight_decay=args.weight_decay)
                elif args.optimizer_type == 'adamw':
                    optimizer = torch.optim.AdamW(model.parameters(), eps=1e-8, betas=(0.9, 0.999), lr=args.learning_rate, weight_decay=0.05)
                else:
                    optimizer = torch.optim.AdamW(model.parameters(), eps=1e-8, betas=(0.9, 0.999), lr=args.learning_rate, weight_decay=0.05)
                
                for epoch in range(args.epoch):
                    correct = 0
                    avg_loss = 0
                    for step, batch in enumerate(train_loader):
                        batch = tuple(t.to(args.device) for t in batch)
                        x, y = batch
                        optimizer.zero_grad()
                        prediction = model(x)
                        pred = prediction.argmax(dim=1, keepdim=True)
                        correct += pred.eq(y.view_as(pred)).sum().item()
                        loss = loss_fn(prediction.view(-1, args.num_classes), y.view(-1))
                        loss.backward()
                        optimizer.step()
                        avg_loss += loss.item()                
                    avg_loss = avg_loss/len(train_loader.dataset)
                    accuracy = 100*(correct/len(train_loader.dataset))
                # evaluate local model
                val_loss, val_accuracy = test(args, model_all, loss_fn, local_model = model)
                for idx, dataset in enumerate(args.datasets):
                    if best_local_accuracy[dataset] < val_accuracy[dataset]:
                        best_local_accuracy[dataset] = val_accuracy[dataset]
                        best_local_loss[dataset] = val_loss[dataset]
                        log_string = f"{client} model for {dataset}:  Best loss: {best_local_loss[dataset]:.4f},  Best accuracy: {best_local_accuracy[dataset]:.4f}"
                        logger.write(log_string+ "\n")
                    
                model.to('cpu')
                
            # average model
            averaging(args, model_avg, model_all) 
            
            print('----------- Validation of communication round ', comm_round, '--------------')
            
            test_loss, test_accuracy = test(args, model_all, loss_fn)  
            for idx, client in enumerate(args.datasets):
                if best_global_accuracy[client] < test_accuracy[client]:
                    best_global_accuracy[client] = test_accuracy[client]
                    best_global_loss[client] = test_loss[client]
                    log_string = f"\n \n Global model for {client}:  Best loss: {best_global_loss[client]:.4f},  Best accuracy: {best_global_accuracy[client]:.4f}"
                    logger.write(log_string+ "\n")
        print('<===== End Training/Testing!======>')
        
                    

class CNN2Model(nn.Module):
    def __init__(self):
        super(CNN2Model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=4, kernel_size=(5, 5), padding=1, stride=1, bias=False)
        self.activation = nn.ReLU(True)
        # self.pool = nn.MaxPool2d(kernel_size=(2, 2), padding=1)
        self.pool = nn.AdaptiveAvgPool2d((7,7))   # adaptive pool
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=12, kernel_size=(5, 5), padding=1, stride=1, bias=False)
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(in_features=(12) * (7 * 7), out_features=10, bias=False)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.pool(x)
        x = self.flatten(x)
        # print(x.view(-1, self.num_flat_features(x)).shape[1])
        x = self.dense1(x)
        return x
    
class LeNetModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, kernel_size=5)
        self.activation = nn.Tanh()
        # self.pool = nn.AvgPool2d(kernel_size=2)
        self.pool = nn.AdaptiveAvgPool2d((7,7))
        self.conv2 = nn.Conv2d(4, 12, kernel_size=5)
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear( (12) * (7 * 7), 10)
    
    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.pool(x)
        x = self.activation(self.conv2(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = self.dense1(x)
        return x        
    
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        # x = nn.functional.relu(nn.functional.max_pool2d(self.conv1(x), 2))
        x = nn.functional.relu(nn.functional.adaptive_max_pool2d(self.conv1(x), 12))
        x = nn.functional.relu(nn.functional.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return nn.functional.log_softmax(x, dim=1)     
    
# ============================Perfect for MNIST=================================================
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
#         self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
#         self.fc1 = nn.Linear(320, 50)
#         self.fc2 = nn.Linear(50, 10)
# 
#     def forward(self, x):
#         x = nn.functional.relu(nn.functional.max_pool2d(self.conv1(x), 2))
#         x = nn.functional.relu(nn.functional.max_pool2d(self.conv2(x), 2))
#         x = x.view(-1, 320)
#         x = nn.functional.relu(self.fc1(x))
#         x = self.fc2(x)
#         return nn.functional.log_softmax(x, dim=1)  
# =============================================================================

class CustomLoader(Dataset):
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
        #sample = {'img':img, 'label':label}
        return img, label 

def averaging(args, model_avg, model_all):
    print('Model averaging...')
    model_avg.cpu()
    params = dict(model_avg.named_parameters())
    for name, param in params.items():
        for index, client in enumerate(args.datasets):
            if index == 0:
                tmp_param_data = dict(model_all[index].named_parameters())[name].data * (1/len(args.datasets))
            else:
                tmp_param_data = tmp_param_data + dict(model_all[index].named_parameters())[name].data * (1/len(args.datasets))
        params[name].data.copy_(tmp_param_data)
    print('Updating clients...')
    for index, client in enumerate(args.datasets):
        tmp_params = dict(model_all[index].named_parameters())
        for name, param in params.items():
            tmp_params[name].data.copy_(param.data)

def test(args, model_all, loss_fn, local_model = None):
    client_accuracy = {}
    client_loss = {}
    for index, client in enumerate(args.datasets):
        data_loader = load_data(args, client, phase = 'test')
        if local_model is not None:
            model = local_model
        else:
            model = model_all[index]
        model.to(args.device)
        model.eval()
        correct = 0
        loss = 0
        with torch.no_grad():
            for step, batch in enumerate(data_loader):
                batch = tuple(t.to(args.device) for t in batch)
                x, y = batch                    
                logits = model(x)
                loss += loss_fn(logits.view(-1, args.num_classes), y.view(-1))
                pred = logits.argmax(dim=1, keepdim=True)
                correct += pred.eq(y.view_as(pred)).sum().item()               
        avg_loss = loss/len(data_loader.dataset)
        accuracy = 100*(correct / len(data_loader.dataset))
        model.train()
        model.cpu()
        client_accuracy[client] = accuracy
        client_loss[client] = avg_loss.item()
    return client_loss, client_accuracy
        
def load_data(args, client, phase):
    if phase in 'train':
        csv_path = os.path.join(os.path.join(args.data_path, client), client+'_train.csv')
        dataset_path = os.path.join(os.path.join(args.data_path, client), client+'_train')
        transform = transforms.Compose([transforms.ToTensor()])
        dataset = CustomLoader(csv_path, dataset_path, transform)
        data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    if phase in 'test':
        csv_path = os.path.join(os.path.join(args.data_path, client), client+'_test.csv')
        dataset_path = os.path.join(os.path.join(args.data_path, client), client+'_test')
        transform = transforms.Compose([transforms.ToTensor()])
        dataset = CustomLoader(csv_path, dataset_path, transform)
        data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    return data_loader     

if __name__ == "__main__":
    main()
