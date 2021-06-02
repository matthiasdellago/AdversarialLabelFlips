import torch
import torch.nn as nn
import os
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR 
from torchvision.datasets import CIFAR10
from torchvision import transforms

# Custom modules
from Networks import CIFAR_Net
from train import train_model

# Foolbox
import foolbox as fb
from foolbox import PyTorchModel, accuracy, samples
from foolbox.attacks import LinfPGD, FGSM, L0BrendelBethgeAttack, L2CarliniWagnerAttack


if __name__ == "__main__":
    save_path = "models" + os.sep + "CIFAR-10" + os.sep
    model = torch.load(save_path + 'reference_model_val_acc=0.8023.pt')
    model.eval()
    
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device) # Moves and/or casts the parameters and buffers to device.
    
    fmodel = PyTorchModel(model, bounds=(-0.5,0.5)) 
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=1)])
    
    
    testset = CIFAR10(root='./data', train=False, transform=transform,)
    
    kwargs = {
        "batch_size": 2**10, "shuffle": True, "pin_memory": True, 
        "num_workers": 4, "persistent_workers": True
    }
    
    dataloader = DataLoader(testset, **kwargs)
    
    
    attack = LinfPGD()
    eps = 0.05
    
    #criterion = nn.CrossEntropyLoss()
    
    
    running_loss = 0.0 # keep track of loss
    running_corrects = 0 # count of carrectly classified images
    
    
    confusion_matrix = np.zeros((10,10), dtype=int)
    
    
    
    for images, labels in dataloader:
        images = images.to(device) # Perform Tensor device conversion
        labels = labels.to(device)
    
        raw_advs, clipped_advs, success = attack(fmodel, images, labels, epsilons=eps)
        
        predicted_labels = model(clipped_advs).argmax(dim=1)
        
        running_corrects += torch.sum(predicted_labels == labels.data)
    
        source_labels = labels.data.cpu().numpy()
        predicted_labels = predicted_labels.cpu().numpy()
        
        # outputs = model(images) # forward pass through network
        # loss = criterion(outputs, labels) # Calculate loss
    
        # running_loss += loss.detach() * images.size(0) # multiply mean loss by the number of elements
        for i, j in zip(source_labels, predicted_labels):
            confusion_matrix[i,j] += 1
        
    names = [str(k) for k in np.arange(10)]
    df = pd.DataFrame(data=confusion_matrix, index=names, columns=names)
    print(df)