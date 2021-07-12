"""
---
title: Train a resnet on CIFAR 10
summary: >
  Train a resnet on CIFAR 10
---

#  Train a resnet on CIFAR 10
"""
import torch
from torch import nn

from labml import experiment
from labml.configs import option
from labml_helpers.module import Module
from labml_nn.experiments.cifar10 import CIFAR10Configs
from labml_nn.resnet import ResNetBase


class Configs(CIFAR10Configs):
    """
    ## Configurations

    We use [`CIFAR10Configs`](../experiments/cifar10.html) which defines all the
    dataset related configurations, optimizer, and a training loop.
    """
    pass


class ResnetModel(Module):
    def __init__(self):
        super().__init__()

        # self.resnet = ResNetBase([3, 3, 3], [16, 32, 64], img_channels=3, first_kernel_size=7)
        self.resnet = ResNetBase([9, 9, 9], [16, 32, 64], img_channels=3, first_kernel_size=7)
        self.linear = nn.Linear(64, 10)

    def __call__(self, x: torch.Tensor):
        x = self.resnet(x)
        return self.linear(x)


@option(Configs.model)
def _resnet(c: Configs):
    """
    ### Create model
    """
    return ResnetModel().to(c.device)


def main():
    # Create experiment
    experiment.create(name='resnet', comment='cifar10')
    # Create configurations
    conf = Configs()
    # Load configurations
    experiment.configs(conf, {
        'optimizer.optimizer': 'Adam',
        'optimizer.learning_rate': 2.5e-4,
        'epochs': 100,
    })
    # Set model for saving/loading
    experiment.add_pytorch_models({'model': conf.model})
    # Start the experiment and run the training loop
    with experiment.start():
        conf.run()


#
if __name__ == '__main__':
    main()
