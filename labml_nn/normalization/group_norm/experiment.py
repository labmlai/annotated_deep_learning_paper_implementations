"""
---
title: CIFAR10 Experiment to try Group Normalization
summary: >
  This trains is a simple convolutional neural network that uses group normalization
  to classify CIFAR10 images.
---

# CIFAR10 Experiment for Group Normalization
"""

import torch.nn as nn

from labml import experiment
from labml.configs import option
from labml_helpers.module import Module
from labml_nn.experiments.cifar10 import CIFAR10Configs
from labml_nn.normalization.group_norm import GroupNorm


class Model(Module):
    """
    ### VGG model for CIFAR-10 classification
    """

    def __init__(self, groups: int = 32):
        super().__init__()
        layers = []
        # RGB channels
        in_channels = 3
        # Number of channels in each layer in each block
        for block in [[64, 64], [128, 128], [256, 256, 256], [512, 512, 512], [512, 512, 512]]:
            # Convolution, Normalization and Activation layers
            for channels in block:
                layers += [nn.Conv2d(in_channels, channels, kernel_size=3, padding=1),
                           GroupNorm(groups, channels),
                           nn.ReLU(inplace=True)]
                in_channels = channels
            # Max pooling at end of each block
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

        # Create a sequential model with the layers
        self.layers = nn.Sequential(*layers)
        # Final logits layer
        self.fc = nn.Linear(512, 10)

    def __call__(self, x):
        # The VGG layers
        x = self.layers(x)
        # Reshape for classification layer
        x = x.view(x.shape[0], -1)
        # Final linear layer
        return self.fc(x)


class Configs(CIFAR10Configs):
    # Number of groups
    groups: int = 16


@option(Configs.model)
def model(c: Configs):
    """
    ### Create model
    """
    return Model(c.groups).to(c.device)


def main():
    # Create experiment
    experiment.create(name='cifar10', comment='group norm')
    # Create configurations
    conf = Configs()
    # Load configurations
    experiment.configs(conf, {
        'optimizer.optimizer': 'Adam',
        'optimizer.learning_rate': 2.5e-4,
    })
    # Start the experiment and run the training loop
    with experiment.start():
        conf.run()


#
if __name__ == '__main__':
    main()
