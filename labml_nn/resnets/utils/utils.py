#!/bin/python

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import KFold
from torch.utils.data.sampler import SubsetRandomSampler



# Plot the loss of multiple runs together
def PlotLosses(losses, titles, save=None):
    fig = plt.figure()
    fig.set_size_inches(14, 22)
    # Plot results on 3 subgraphs
    # subplot integers:
    #       nrows
    #       ncols
    #       index
    sublplot_str_start = "" + str(len(losses)) + "1"

    for i in range(len(losses)):
        subplot = sublplot_str_start + str(i+1)
        loss = losses[i]
        title = titles[i]

        ax = plt.subplot(int(subplot))
        ax.plot(range(len(loss)), loss)
        ax.set_xlabel("Epoch")
        ax.set_title(title)
        ax.set_ylabel("Loss")

    # Save Figure
    if save:
    	plt.savefig(save)
    else:
    	plt.show()



def ClassSpecificTestCifar10(net, testdata, device=None):
    classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in testdata:
            if device:
                images, labels = data[0].to(device), data[1].to(device)
            else:
                images, labels = data

            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    # Print out
    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))



def GetActivation(name="relu"):
    if name == "relu":
        return nn.ReLU()
    elif name == "leakyrelu":
        return nn.LeakyReLU()
    elif name == "Sigmoid":
        return nn.Sigmoid()
    elif name == "Tanh":
        return nn.Tanh()
    elif name == "Identity":
        return nn.Identity()
    else:
        return nn.ReLU()