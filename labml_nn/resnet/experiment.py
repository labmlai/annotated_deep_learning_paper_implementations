"""
---
title: Train a resnet on CIFAR 10
summary: >
  Train a resnet on CIFAR 10
---

#  Train a resnet on CIFAR 10
"""
from typing import List, Optional

from torch import nn

from labml import experiment
from labml.configs import option
from labml_nn.experiments.cifar10 import CIFAR10Configs
from labml_nn.resnet import ResNetBase


class Configs(CIFAR10Configs):
    """
    ## Configurations

    We use [`CIFAR10Configs`](../experiments/cifar10.html) which defines all the
    dataset related configurations, optimizer, and a training loop.
    """

    n_blocks: List[int] = [3, 3, 3]
    n_channels: List[int] = [16, 32, 64]
    bottlenecks: Optional[List[int]] = None
    first_kernel_size: int = 7


@option(Configs.model)
def _resnet(c: Configs):
    """
    ### Create model
    """
    base = ResNetBase(c.n_blocks, c.n_channels, c.bottlenecks, img_channels=3, first_kernel_size=c.first_kernel_size)
    classification = nn.Linear(c.n_channels[-1], 10)

    model = nn.Sequential(base, classification)
    return model.to(c.device)


def main():
    # Create experiment
    experiment.create(name='resnet', comment='cifar10')
    # Create configurations
    conf = Configs()
    # Load configurations
    experiment.configs(conf, {
        'bottlenecks': [8, 16, 16],

        'optimizer.optimizer': 'Adam',
        'optimizer.learning_rate': 2.5e-4,
        # 'optimizer.weight_decay': 1e-4,
        'epochs': 100,
        'train_batch_size': 256,

        'train_dataset': 'cifar10_train_augmented',
        'valid_dataset': 'cifar10_valid_no_augment',
    })
    # Set model for saving/loading
    experiment.add_pytorch_models({'model': conf.model})
    # Start the experiment and run the training loop
    with experiment.start():
        conf.run()


#
if __name__ == '__main__':
    main()
