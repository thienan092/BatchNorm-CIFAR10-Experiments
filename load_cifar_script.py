# %% [code]
# Source: https://www.kaggle.com/code/mgorinova/load-cifar-script
import os
import requests

import numpy as np

import torch
from torch.utils.data import DataLoader, Subset, random_split

import torchvision
from torchvision import transforms
from torchvision.models import resnet18
from sklearn.model_selection import train_test_split


def get_cifar10_data(rng = 42):
    """Returns a dictionary containing all CIFAR10 data loaders
    required for the unlearn task.
    """
    # manual random seed is used for dataset partitioning
    # to ensure reproducible results across runs
    # download and pre-process CIFAR10
    normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    train_set = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=normalize)
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True)

    # we split held out data into test and validation set
    held_out = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=normalize)
    torch.manual_seed(rng)
    test_set, val_set = random_split(held_out, [len(held_out)-len(held_out)//2, len(held_out)//2])
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False)
    val_loader = DataLoader(val_set, batch_size=128, shuffle=False)
    
    return {
        "training": train_loader,
        "testing": test_loader,
        "validation": val_loader
    }


def get_cifar10_pretrained_models(device = None):
    """Returns a dictionary of the original ResNet18 model, pretrained on 
    the whole of the training set, and the retrained ResNet18 model, which
    was retrained only on the retain set."""
    
    # Original model
    model = resnet18(pretrained=False, num_classes=10)
    model.load_state_dict(torch.load("weights_resnet18_cifar10.pth"))
    model.to(device)
    model.eval();
    
    # load model with pre-trained weights
    net = resnet18(pretrained=False, num_classes=10)
    net.load_state_dict(torch.load("weights_resnet18.pth"))
    net.to(device)
    net.eval();
    
    return {
        "original": model,
        "backend": net
    }   