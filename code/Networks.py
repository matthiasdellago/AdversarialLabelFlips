import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import datasets
from livelossplot import PlotLosses
# Reminder
# Conv2d(in, out, kernelsize)

# https://arxiv.org/pdf/1608.04644.pdf

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

#net = MNIST_Net()

# net = CIFAR_Net()
# cifar_trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=None)
# x_train = cifar_trainset.data/255
# x_train = np.moveaxis(x_train,3,1)
# x_train = torch.Tensor(x_train)         


# print(net(x_train[:11]).shape)


def train_model(model, criterion, optimizer, dataloaders, device, num_epochs):
    liveloss = PlotLosses() # Live training plot generic API
    model = model.to(device) # Moves and/or casts the parameters and buffers to device.
    
    for epoch in range(num_epochs): # Number of passes through the entire training & validation datasets
        logs = {}
        for phase in ['train', 'validation']: # First train, then validate
            if phase == 'train':
                model.train() # Set the module in training mode
            else:
                model.eval() # Set the module in evaluation mode
    
            running_loss = 0.0 # keep track of loss
            running_corrects = 0 # count of carrectly classified inputs
    
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device) # Perform Tensor device conversion
                labels = labels.to(device)
    
                outputs = model(inputs) # forward pass through network
                loss = criterion(outputs, labels) # Calculate loss
    
                if phase == 'train':
                    optimizer.zero_grad() # Set all previously calculated gradients to 0
                    loss.backward() # Calculate gradients
                    optimizer.step() # Step on the weights using those gradient w -=  gradient(w) * lr
    
                _, preds = torch.max(outputs, 1) # Get model's predictions
                running_loss += loss.detach() * inputs.size(0) # multiply mean loss by the number of elements
                running_corrects += torch.sum(preds == labels.data) # add number of correct predictions to total
    
            epoch_loss = running_loss / len(dataloaders[phase].dataset) # get the "mean" loss for the epoch
            epoch_acc = running_corrects.float() / len(dataloaders[phase].dataset) # Get proportion of correct predictions
            
            # Logging
            prefix = ''
            if phase == 'validation':
                prefix = 'val_'
    
            logs[prefix + 'log loss'] = epoch_loss.item()
            logs[prefix + 'accuracy'] = epoch_acc.item()
        
        liveloss.update(logs) # Update logs
        liveloss.send() # draw, display stuff