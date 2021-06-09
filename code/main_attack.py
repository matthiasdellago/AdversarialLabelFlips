import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Torch
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10
from torchvision import transforms

# Foolbox
import foolbox as fb
from foolbox.attacks import LinfPGD, FGSM, L0BrendelBethgeAttack, L2CarliniWagnerAttack

# Custom modules
from utils import compute_confusion_matrix

def execute_attack(dataset: str, model_filename: str, 
           attack, attack_kwargs, attack_name: str):
    
    ##########################################################################
    # Loading model
    ##########################################################################
    model_path = "models" + os.sep + dataset + os.sep + model_filename
    
    # Move everything to the GPU if possible
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = torch.load(model_path, map_location=device)
    model.eval()
    
    ##########################################################################
    # Loading test data + data loader
    ##########################################################################
    # Make input suitable for our models
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=1)])
    
    dataset_kwargs = {"root":'./data', "download":True, 
                      "train":False, "transform":transform} 
    
    # Choose the correct dataset 
    if dataset == "MNIST":
        testset = MNIST(**dataset_kwargs)
        labels = [str(k) for k in np.arange(10)]
    if dataset == "CIFAR-10":
        testset = CIFAR10(**dataset_kwargs)
        labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    dataloader = DataLoader(testset, batch_size=1000, num_workers=4)
    
    ##########################################################################
    # Generate confusion matrix
    ##########################################################################
    confusion_matrix = compute_confusion_matrix(
        model, attack, attack_kwargs, dataloader, device)
    
    ##########################################################################
    # Print as Pandas dataframe and visualize with matplotlib
    ##########################################################################
    df = pd.DataFrame(data=confusion_matrix, index=labels, columns=labels)
    
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    print(df)
    
    plt.imshow(confusion_matrix)
    plt.xticks(range(len(labels)), labels, size='small')
    plt.yticks(range(len(labels)), labels, size='small')
    
    plt.xticks(rotation=90)
    ##########################################################################
    # Save confusion_matrix
    ##########################################################################
    #df.save("")
    
"""    
All configs are done here
"""  
if __name__ == "__main__":
    ##########################################################################
    # Choose dataset and model
    ##########################################################################
    dataset = "MNIST"
    model_filename = 'reference_model_val_acc=0.9948.pt'
    
    # dataset = "CIFAR-10"
    # model_filename = 'reference_model_val_acc=0.8023.pt'
    
    ##########################################################################
    # Config attack
    ##########################################################################
    # attack = L0BrendelBethgeAttack()
    # attack_kwargs = {"epsilons": None}
    # attack_name = "L0BrendelBethgeAttack"
    
    # attack = L1BrendelBethgeAttack()
    # attack_kwargs = {"epsilons": None}
    # attack_name = "L1BrendelBethgeAttack"
    
    # attack = L2CarliniWagnerAttack()
    # attack_kwargs = {"epsilons": None}
    # attack_name = "L2CarliniWagnerAttack"
    
    attack = LinfPGD()
    attack_kwargs = {"epsilons": 0.2}
    attack_name = "LinfPGD"

    ##########################################################################
    # ATTACK
    ##########################################################################

    execute_attack(dataset, model_filename, attack, attack_kwargs, attack_name)    