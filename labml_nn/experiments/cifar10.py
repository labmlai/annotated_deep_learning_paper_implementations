"""
---
title: CIFAR10 Experiment
summary: >
  This is a reusable trainer for CIFAR10 dataset
---

# CIFAR10 Experiment
"""
from typing import List

import torch.nn as nn

from labml_helpers.datasets.cifar10 import CIFAR10Configs as CIFAR10DatasetConfigs
from labml_helpers.module import Module
from labml_nn.experiments.mnist import MNISTConfigs


class CIFAR10Configs(CIFAR10DatasetConfigs, MNISTConfigs):
    dataset_name: str = 'CIFAR10'


class CIFAR10VGGModel(Module):
    """
    ### VGG model for CIFAR-10 classification
    """

    def conv_block(self, in_channels, out_channels) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def __init__(self, blocks: List[List[int]]):
        super().__init__()

        # 5 $2 \times 2$ pooling layers will produce a output of size $1 \ times 1$.
        # CIFAR 10 image size is $32 \times 32$
        assert len(blocks) == 5
        layers = []
        # RGB channels
        in_channels = 3
        # Number of channels in each layer in each block
        for block in blocks:
            # Convolution, Normalization and Activation layers
            for channels in block:
                layers += self.conv_block(in_channels, channels)
                in_channels = channels
            # Max pooling at end of each block
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

        # Create a sequential model with the layers
        self.layers = nn.Sequential(*layers)
        # Final logits layer
        self.fc = nn.Linear(in_channels, 10)

    def __call__(self, x):
        # The VGG layers
        x = self.layers(x)
        # Reshape for classification layer
        x = x.view(x.shape[0], -1)
        # Final linear layer
        return self.fc(x)
