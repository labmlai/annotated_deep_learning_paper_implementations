"""
---
title: Train a large model on CIFAR 10
summary: >
  Train a large model on CIFAR 10 for distillation.
---

#  Train a large model on CIFAR 10

This trains a large model on CIFAR 10 for [distillation](index.html).

[![View Run](https://img.shields.io/badge/labml-experiment-brightgreen)](https://app.labml.ai/run/d46cd53edaec11eb93c38d6538aee7d6)
"""

import torch.nn as nn

from labml import experiment, logger
from labml.configs import option
from labml_nn.experiments.cifar10 import CIFAR10Configs, CIFAR10VGGModel
from labml_nn.normalization.batch_norm import BatchNorm


class Configs(CIFAR10Configs):
    """
    ## Configurations

    We use [`CIFAR10Configs`](../experiments/cifar10.html) which defines all the
    dataset related configurations, optimizer, and a training loop.
    """
    pass


class LargeModel(CIFAR10VGGModel):
    """
    ### VGG style model for CIFAR-10 classification

    This derives from the [generic VGG style architecture](../experiments/cifar10.html).
    """

    def conv_block(self, in_channels, out_channels) -> nn.Module:
        """
        Create a convolution layer and the activations
        """
        return nn.Sequential(
            # Dropout
            nn.Dropout(0.1),
            # Convolution layer
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            # Batch normalization
            BatchNorm(out_channels, track_running_stats=False),
            # ReLU activation
            nn.ReLU(inplace=True),
        )

    def __init__(self):
        # Create a model with given convolution sizes (channels)
        super().__init__([[64, 64], [128, 128], [256, 256, 256], [512, 512, 512], [512, 512, 512]])


@option(Configs.model)
def _large_model(c: Configs):
    """
    ### Create model
    """
    return LargeModel().to(c.device)


def main():
    # Create experiment
    experiment.create(name='cifar10', comment='large model')
    # Create configurations
    conf = Configs()
    # Load configurations
    experiment.configs(conf, {
        'optimizer.optimizer': 'Adam',
        'optimizer.learning_rate': 2.5e-4,
        'is_save_models': True,
        'epochs': 20,
    })
    # Set model for saving/loading
    experiment.add_pytorch_models({'model': conf.model})
    # Print number of parameters in the model
    logger.inspect(params=(sum(p.numel() for p in conf.model.parameters() if p.requires_grad)))
    # Start the experiment and run the training loop
    with experiment.start():
        conf.run()


#
if __name__ == '__main__':
    main()
