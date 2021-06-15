import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def create_plots(dataset, attack_name, epsilons=None, save=False):
    print("#############################################################")
    print(dataset, attack_name, epsilons)
    print("#############################################################")
    result_path = "results" + os.sep + dataset + os.sep 
    figure_path = result_path + "figures" + os.sep
    
    if epsilons is None:
        df = pd.read_csv(result_path + f"{attack_name}.csv")
    if epsilons is not None:
        df = pd.read_csv(result_path + f"{attack_name}, epsilon={epsilons}.csv")
    
    confusion_matrix = df[df.columns[1:]].to_numpy()
    
    if dataset == "MNIST": 
        labels = [str(k) for k in np.arange(10)] 
        
    if dataset == "FashionMNIST": 
        labels = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 
                  'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        
    if dataset == "CIFAR-10":
        labels = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 
                  'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
    
    plt.imshow(confusion_matrix)
    plt.xticks(range(len(labels)), labels, size='small')
    plt.yticks(range(len(labels)), labels, size='small')
    
    plt.xticks(rotation=90)
    plt.tight_layout()
    
    if not save:
        plt.show()
        return 
    
    if epsilons is None:
        plt.savefig(figure_path + f"{attack_name}.png")
        plt.savefig(figure_path + f"{attack_name}.pdf")
    
    if epsilons is not None:
        plt.savefig(figure_path + f"{attack_name}, epsilon={epsilons}.png")
        plt.savefig(figure_path + f"{attack_name}, epsilon={epsilons}.pdf")
        
    plt.show()
    
        
if __name__ == "__main__":
    save = True
    
    datasets = ["MNIST", "FashionMNIST", "CIFAR-10"]
    attack_names = ["L0BrendelBethgeAttack",
                    "L1BrendelBethgeAttack",
                    "L2CarliniWagnerAttack",
                    "LinfPGD"]
    
    for dataset in datasets:
        for attack_name in attack_names:
            if attack_name == "LinfPGD":
                for eps in [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1]:
                    create_plots(dataset, attack_name, epsilons=eps, save=save)
            else:
                create_plots(dataset, attack_name, epsilons=None, save=save)
 