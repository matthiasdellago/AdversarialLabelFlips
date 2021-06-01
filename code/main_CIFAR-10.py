import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR 
from torchvision.datasets import CIFAR10
from torchvision import transforms

# Custom modules
from Networks import CIFAR_Net
from train import train_model

"""
The neural net sometimes diverges in the first epoch. 
In that case restart the program.


"""

if __name__ == "__main__":
    # Save model if validation accuracy increases.
    save_path = "models" + os.sep + "CIFAR-10" + os.sep
    
    model = CIFAR_Net()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    criterion = nn.CrossEntropyLoss()
    
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=1)])
    
    trainset = CIFAR10(root='./data', train=True, download=True,
                       transform=transform,)
    testset = CIFAR10(root='./data', train=False, transform=transform,)
    
    
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
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True,
                    weight_decay=1e-6)
    scheduler = StepLR(optimizer, 10, 0.5) # Decay lr by 50% every 10 epochs
            
    train_model(model, criterion, optimizer, dataloaders, device, epochs,
                save_path = save_path, scheduler=scheduler)