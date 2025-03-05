import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import random
def get_transform():
    transform=transforms.Compose([
       transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    return transform
def get_dataset(transform):
    dataset=datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    y=np.array(dataset.targets)
    splitter=StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, valid_idx = next(splitter.split(np.zeros((len(y), 1)), y))
    train_dataset=Subset(dataset, train_idx)
    valid_dataset=Subset(dataset, valid_idx)
    test_dataset=datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    return train_dataset, valid_dataset, test_dataset

def get_dataloader(train_dataset, valid_dataset, test_dataset, batch_size):
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, valid_loader, test_loader