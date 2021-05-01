#!/bin/python

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, random_split
import matplotlib.pyplot as plt
import numpy as np

def LoadCifar10DatasetTrain(save, transform=None):
    trainset = torchvision.datasets.CIFAR10(root=save, train=True,
                                        download=True, transform=transform)
    return trainset

def LoadCifar10DatasetTest(save, transform):
    return torchvision.datasets.CIFAR10(root=save, train=False,
                                       download=False, transform=transform)

def GetCustTransform():
    transform_train = transforms.Compose([
        transforms.RandomRotation(20),
        transforms.RandomCrop(32, (2, 2), pad_if_needed=False, padding_mode='constant'),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    return transform_train

def Dataloader_train_valid(save, batch_size):

    # See utils/dataloader.py for data augmentations
    transform_train_valid = GetCustTransform()

    # Get Cifar 10 Datasets
    trainset = LoadCifar10DatasetTrain(save, transform_train_valid)
    train_val_abs = int(len(trainset) * 0.8)
    train_subset, val_subset = random_split(trainset, [train_val_abs, len(trainset) - train_val_abs])

    # Get Cifar 10 Dataloaders
    trainloader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size,
                                              shuffle=True, num_workers=4)

    valloader = torch.utils.data.DataLoader(val_subset, batch_size=batch_size,
                                            shuffle=True, num_workers=4)
    return trainloader, valloader

def Dataloader_train(save, batch_size):

    # See utils/dataloader.py for data augmentations
    transform_train = GetCustTransform()

    # Get Cifar 10 Datasets
    trainset = LoadCifar10DatasetTrain(save, transform_train)
    # Get Cifar 10 Dataloaders
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=4)

    return trainloader

def Dataloader_test(save, batch_size):

    # transformation test set
    transform_test = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # initialize test dataset and dataloader
    testset = LoadCifar10DatasetTest(save, transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                             shuffle=False, num_workers=4)

    return testloader

def imshow(im):
    image = im.cpu().clone().detach().numpy()
    image = image.transpose(1, 2, 0)
    image = image * np.array((0.5, 0.5, 0.5)) + np.array((0.5, 0.5, 0.5)) # unnormalize
    plt.imshow(image)
    plt.show()

def imretrun(im):
    image = im.cpu().clone().detach().numpy()
    image = image.transpose(1, 2, 0)
    image = image * np.array((0.5, 0.5, 0.5)) + np.array((0.5, 0.5, 0.5)) # unnormalize
    return image