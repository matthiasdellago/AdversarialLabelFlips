# Adversarial Label Flips

![](./Seminar thesis/titlefigure.eps)

Welcome to our repository that delves into the realm of adversarial examples and their effect on neural network classifiers. Our primary aim is to identify the class an adversarial example falls into post an untargeted evasion attack. For this, we've run several experiments using simple neural network classifiers trained on industry-standard datasets and tested against state-of-the-art attacks. Intriguingly, our preliminary findings indicate that semantically similar classes are more likely to be confused with one another.

## Introduction

The phenomenon of adversarial attacks on deep neural networks was brought to light by [Szegedy et al.](https://arxiv.org/abs/1312.6199) in 2013. They found that adversarial examples, which are formed by applying minor perturbations to benign inputs, could lead a neural network astray.

This repository mainly deals with image classifiers, more specifically, Convolutional Neural Networks (CNNs). The attacks here involve minimal changes to the input images, with the intention to force the CNN to misclassify them.

We focus on two primary types of attacks:

1. **Targeted Attacks**: Aimed at causing a misclassification into a specific class.
2. **Untargeted Attacks**: Designed to evade correct classification without having a specific class target.

The objective of our experiments is to identify the class that an adversarial image falls into after an untargeted attack.

## Background and Related Work

Since their discovery by [Szegedy et al.](https://arxiv.org/abs/1312.6199), adversarial examples have been extensively studied. We focus on the following established attacks:

- Fast Gradient Sign Method (FGSM) by [Goodfellow et al.](https://arxiv.org/abs/1412.6572)
- Projected Gradient Descent (PGD) as introduced by [Madry et al.](https://arxiv.org/abs/1706.06083)
- Carlini-Wagner Attack by [Carlini and Wagner](https://arxiv.org/abs/1608.04644)
- Brendel-Bethge Attack by [Brendel and Bethge](https://arxiv.org/abs/1910.09338)

We leveraged the Foolbox framework for generating adversarial examples. Our neural networks, trained on the MNIST, Fashion-MNIST, and CIFAR-10 datasets, were subjected to the aforementioned adversarial attacks.

## Experimental Setup

All experiments were conducted using Python 3.8.5 and PyTorch 1.8.1 on a Windows machine. Adversarial attacks were computed using Foolbox 3.3.1.

## Adversarial Attacks - Results and Analysis

We present the confusion matrices for the **Carlini-Wagner Attack:**.

### CIFAR-10

![](./code/results/CIFAR-10/figures/L2CarliniWagnerAttack.png)


### FashionMNIST

![](./code/results/FashionMNIST/figures/L2CarliniWagnerAttack.png)


### MNIST

**Carlini-Wagner Attack:**
![](./code/results/MNIST/figures/L2CarliniWagnerAttack.png)


### Symmetry

One of the standout features of our results is the high degree of symmetry in the CIFAR-10 confusion matrices. This implies that if an adversarial example of class i is mistaken for class j, the reverse is also likely. A discernible pattern across all datasets reveals that the pairs "Automobile"-"Truck", and "Dog"-"Cat" are most commonly confused, which aligns with human perspective. Misclassifications between animals and vehicles are significantly less common.

However, this hypothesis seems less applicable for the MNIST and FashionMNIST datasets.

### Catch-all Classes

Adversarial examples created with large perturbation budgets (`𝜖`) are most often misclassified as "frog", and "8" for MNIST and CIFAR-10 respectively. For FashionMNIST, there are multiple high probability classes. To better understand this phenomenon, we generated and classified 10,000 white noise images sampled from a uniform distribution on the input domain.

![White Noise Classification](./code/results/barplot.png)

The results indicate that the tested neural networks tend to default to one or multiple outputs for low probability images. This behaviour significantly impacts adversarial examples computed with large perturbation budgets.

## Conclusion

In conclusion, our studies have uncovered two intriguing patterns:

- Adversarial images with small perturbation sizes often lead to surprisingly symmetric confusion matrices, suggesting the classifier's understanding of the relationship between certain classes.
- Attacks that employ a larger `𝜖` typically cluster into one or multiple specific classes. This can be attributed to the CNN's tendency to use these classes as a catch-all for images it struggles to classify correctly.

We hope our research offers valuable insights into adversarial attacks and inspires further investigations into this fascinating area.
