import os
import numpy as np
import pandas as pd

# Torch
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms

# Foolbox
import foolbox as fb
from foolbox import PyTorchModel
from foolbox.attacks import LinfPGD, FGSM, L0BrendelBethgeAttack, L2CarliniWagnerAttack

# Custom modules
from utils import compute_confusion_matrix

if __name__ == "__main__":
    ### CONFIG
    save_path = "models" + os.sep + "CIFAR-10" + os.sep
    model = torch.load(save_path + 'reference_model_val_acc=0.8023.pt')
    model.eval()
    
    # Move everything to GPU if possible
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device) 
    
    attack = LinfPGD()
    attack_kwargs = {"epsilons": 1}
    
    # Tell FoolBox this is a PyTorchModel
    fmodel = PyTorchModel(model, bounds=(-0.5,0.5)) 
    
    # Preparing dataloader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=1)])
    testset = CIFAR10(root='./data', train=False, transform=transform,)
    kwargs = {
        "batch_size": 2**10, "shuffle": True, "pin_memory": True, 
        "num_workers": 4, "persistent_workers": True
    }
    dataloader = DataLoader(testset, **kwargs)
    
    # Generate confusion matrix
    confusion_matrix = compute_confusion_matrix(
        fmodel, attack, attack_kwargs, dataloader, device)
    
    # For MNIST
    names = [str(k) for k in np.arange(10)]
    
    # For CIFAR-10
    names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    df = pd.DataFrame(data=confusion_matrix, index=names, columns=names)
    
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    print(df)
    