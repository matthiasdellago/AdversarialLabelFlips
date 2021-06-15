import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def measure_symmetry(confusion_matrix):
    trans = np.transpose(confusion_matrix)
    #symmetric component
    sym = (confusion_matrix + trans)/2
    #remove the diagonal values because, that just means "attack unsuccessful" and shouldn't count towards a higher "symmetry"
    np.fill_diagonal(sym,0)
    #print(sym)
    #antisymmetric component
    skew = (confusion_matrix - trans)/2
    #print(skew)
    # difference of the 1-norms of the two matrices, normalised by their sum
    # something like the relative difference of the two ¯\_(ツ)_/¯
    order = 1
    norm_skew = np.linalg.norm(skew, ord = order)
    norm_sym = np.linalg.norm(sym, ord = order)
    symmetryness =  (norm_sym - norm_skew) / (norm_sym + norm_skew)
    return symmetryness

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

    print("Symmetry:", measure_symmetry(confusion_matrix),"\n")
        
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
 