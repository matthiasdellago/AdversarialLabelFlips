import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
from torchvision.datasets import MNIST
from torchvision import transforms
from livelossplot import PlotLosses
from Networks import MNIST_Net, train_model




model = MNIST_Net()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

criterion = nn.CrossEntropyLoss()

transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))
    ])

trainset = MNIST(root='./data', train=True, 
                 download=True, transform=transform,
)
testset = MNIST(root='./data', train=False, 
                transform=transform,
)

train_loader = torch.utils.data.DataLoader(trainset, batch_size=128)
test_loader = torch.utils.data.DataLoader(testset, batch_size=128)

dataloaders = {
    "train": train_loader,
    "validation": test_loader
}

'''
Training specification taken from https://arxiv.org/pdf/1608.04644.pdf
'''
epochs = 50
optimizer = torch.optim.SGD(model.parameters(), 
                            lr=0.1, 
                            momentum=0.9,
                            nesterov=True
)
        
train_model(model, criterion, optimizer, dataloaders, device, epochs)