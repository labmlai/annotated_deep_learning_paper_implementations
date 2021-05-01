"""
---
title: CIFAR10 Experiment to try Weight Standardization and Batch-Channel Normalization
summary: >
  This trains is a VGG net that uses weight standardization  and batch-channel normalization
  to classify CIFAR10 images.
---

# CIFAR10 Experiment to try Weight Standardization and Batch-Channel Normalization
"""

import torch.nn as nn

from labml import experiment
from labml.configs import option
from labml_helpers.module import Module
from labml_nn.experiments.cifar10 import CIFAR10Configs
from labml_nn.normalization.batch_channel_norm import BatchChannelNorm
from labml_nn.normalization.weight_standardization.conv2d import Conv2d


class Model(Module):
    """
    ### Model

    A VGG model that use [Weight Standardization](./index.html) and
     [Batch-Channel Normalization](../batch_channel_norm/index.html).
    """
    def __init__(self):
        super().__init__()
        layers = []
        in_channels = 3
        for block in [[64, 64], [128, 128], [256, 256, 256], [512, 512, 512], [512, 512, 512]]:
            for channels in block:
                layers += [Conv2d(in_channels, channels, kernel_size=3, padding=1),
                           BatchChannelNorm(channels, 32),
                           nn.ReLU(inplace=True)]
                in_channels = channels
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        self.layers = nn.Sequential(*layers)
        self.fc = nn.Linear(512, 10)

    def __call__(self, x):
        x = self.layers(x)
        x = x.view(x.shape[0], -1)
        return self.fc(x)


@option(CIFAR10Configs.model)
def model(c: CIFAR10Configs):
    """
    ### Create model
    """
    return Model().to(c.device)


def main():
    # Create experiment
    experiment.create(name='cifar10', comment='weight standardization')
    # Create configurations
    conf = CIFAR10Configs()
    # Load configurations
    experiment.configs(conf, {
        'optimizer.optimizer': 'Adam',
        'optimizer.learning_rate': 2.5e-4,
        'train_batch_size': 64,
    })
    # Start the experiment and run the training loop
    with experiment.start():
        conf.run()


#
if __name__ == '__main__':
    main()
