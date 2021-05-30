import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
import os
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from livelossplot import PlotLosses
from Networks import MNIST_Net, train_model

"""
The neural net sometimes diverges in the first epoch. 
In that case restart the program.


"""

if __name__ == "__main__":
    # Save model if validation accuracy increases.
    save_path = "models" + os.sep + "MNIST" + os.sep
    
    model = MNIST_Net()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    criterion = nn.CrossEntropyLoss()
    
    
    trainset = MNIST(root='./data', train=True,
                     download=True, transform=transforms.ToTensor(),
    )
    testset = MNIST(root='./data', train=False, 
                    transform=transforms.ToTensor(),
    )
    
    
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
    optimizer = torch.optim.SGD(model.parameters(), 
                                lr=0.1, 
                                momentum=0.9,
                                nesterov=True
    )
            
    train_model(model, criterion, optimizer, dataloaders, device, epochs,
                save_path = save_path)