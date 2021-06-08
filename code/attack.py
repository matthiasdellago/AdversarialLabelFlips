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
from foolbox.attacks import LinfPGD, FGSM, L0BrendelBethgeAttack, L2CarliniWagnerAttack

# Custom modules
from utils import compute_confusion_matrix

if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    ### CONFIG
    save_path = "models" + os.sep + "CIFAR-10" + os.sep
    
    #attack, attack_kwargs = L0BrendelBethgeAttack(), {"epsilons": None}
    #attack, attack_kwargs = L1BrendelBethgeAttack(), {"epsilons": None}
    #attack, attack_kwargs = L2CarliniWagnerAttack(), {"epsilons": None}
    attack, attack_kwargs = FGSM(), {"epsilons": 0.01}
    #attack, attack_kwargs = LinfPGD(), {"epsilons": 0.01}
    print(attack_kwargs)    
    
    # Load model and move everything to GPU if possible
    model = torch.load(save_path + 'reference_model_val_acc=0.8023.pt', 
                       map_location=device)
    model.eval()
    
    
    # Preparing dataloader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=0.5, std=1)])
    testset = CIFAR10(root='./data', download=True, train=False, transform=transform,)
    kwargs = {
        "batch_size": 2**7, "shuffle": True, "pin_memory": True, 
        "num_workers": 4, "persistent_workers": True
    }
    dataloader = DataLoader(testset, **kwargs)
    
    # Generate confusion matrix
    confusion_matrix = compute_confusion_matrix(
        model, attack, attack_kwargs, dataloader, device)
    
    # For MNIST
    names = [str(k) for k in np.arange(10)]
    
    # For CIFAR-10
    names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    df = pd.DataFrame(data=confusion_matrix, index=names, columns=names)
    
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    print(df)
    