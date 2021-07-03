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
from labml_nn.experiments.cifar10 import CIFAR10Configs, CIFAR10VGGModel
from labml_nn.normalization.batch_channel_norm import BatchChannelNorm
from labml_nn.normalization.weight_standardization.conv2d import Conv2d


class Model(CIFAR10VGGModel):
    """
    ### VGG model for CIFAR-10 classification

    This derives from the [generic VGG style architecture](../../experiments/cifar10.html).
    """

    def conv_block(self, in_channels, out_channels) -> nn.Module:
        return nn.Sequential(
            Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            BatchChannelNorm(out_channels, 32),
            nn.ReLU(inplace=True),
        )

    def __init__(self):
        super().__init__([[64, 64], [128, 128], [256, 256, 256], [512, 512, 512], [512, 512, 512]])


@option(CIFAR10Configs.model)
def _model(c: CIFAR10Configs):
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
