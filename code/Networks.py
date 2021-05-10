import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Reminder
# Conv2d(in, out, kernelsize)

class MNIST_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3) 
        self.conv2 = nn.Conv2d(32, 32, 3) 
        self.conv3 = nn.Conv2d(32, 64, 3) 
        self.conv4 = nn.Conv2d(64, 64, 3) 
        
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.5)
        
        self.fc1 = nn.Linear(1024, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CIFAR_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3) 
        self.conv2 = nn.Conv2d(64, 64, 3) 
        self.conv3 = nn.Conv2d(64, 128, 3) 
        self.conv4 = nn.Conv2d(128, 128, 3) 
        
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.5)
        
        self.fc1 = nn.Linear(3200, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = MNIST_Net()

import torchvision
from torchvision import datasets
mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
x_train = mnist_trainset.data/255
x_train = x_train.unsqueeze(1)

print(net(x_train[:11]).shape)


net = CIFAR_Net()
cifar_trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=None)
x_train = cifar_trainset.data/255
x_train = np.moveaxis(x_train,3,1)
x_train = torch.Tensor(x_train)         


print(net(x_train[:11]).shape)