import os
import numpy as np

# Torch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD
from torchvision.datasets import MNIST
from torchvision import transforms

# Custom modules
from Networks import MNIST_Net
from utils import train_model

"""
Trains and saves a neural net for the MNIST dataset.
"""
if __name__ == "__main__":
    # Save model if validation accuracy increases.
    save_path = "models" + os.sep + "MNIST" + os.sep
    
    model = MNIST_Net()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    criterion = nn.CrossEntropyLoss()
    
    # Preparing dataloaders
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=1)])
        
    trainset = MNIST(root='./data', train=True, download=True, transform=transform,)
    testset = MNIST(root='./data', train=False, transform=transform,)
    
    kwargs = {
        "batch_size": 128, "shuffle": True, "pin_memory": True, 
        "num_workers": 4, "persistent_workers": True
    }

    dataloaders = {
        "train": DataLoader(trainset, **kwargs),
        "validation": DataLoader(testset, **kwargs),
    }
    
    '''
    Training specification taken from https://arxiv.org/pdf/1608.04644.pdf
    '''
    epochs = 50
    optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True,
                    weight_decay=1e-6)
            
    train_model(model, criterion, optimizer, dataloaders, device, epochs,
                save_path = save_path)