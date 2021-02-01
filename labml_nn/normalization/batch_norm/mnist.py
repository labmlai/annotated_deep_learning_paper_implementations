"""
---
title: MNIST Experiment to try Batch Normalization
summary: >
  This trains is a simple convolutional neural network that uses batch normalization
  to classify MNIST digits.
---

# MNIST Experiment for Batch Normalization
"""

import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

from labml import experiment
from labml.configs import option
from labml_helpers.module import Module
from labml_nn.experiments.mnist import MNISTConfigs
from labml_nn.normalization.batch_norm import BatchNorm


class Model(Module):
    """
    ### Model definition
    """

    def __init__(self):
        super().__init__()
        # Note that we omit the bias parameter
        self.conv1 = nn.Conv2d(1, 20, 5, 1, bias=False)
        # Batch normalization with 20 channels (output of convolution layer).
        # The input to this layer will have shape `[batch_size, 20, height(24), width(24)]`
        self.bn1 = BatchNorm(20)
        #
        self.conv2 = nn.Conv2d(20, 50, 5, 1, bias=False)
        # Batch normalization with 50 channels.
        # The input to this layer will have shape `[batch_size, 50, height(8), width(8)]`
        self.bn2 = BatchNorm(50)
        #
        self.fc1 = nn.Linear(4 * 4 * 50, 500, bias=False)
        # Batch normalization with 500 channels (output of fully connected layer).
        # The input to this layer will have shape `[batch_size, 500]`
        self.bn3 = BatchNorm(500)
        #
        self.fc2 = nn.Linear(500, 10)

    def __call__(self, x: torch.Tensor):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.bn3(self.fc1(x)))
        return self.fc2(x)


@option(MNISTConfigs.model)
def model(c: MNISTConfigs):
    """
    ### Create model

    We use [`MNISTConfigs`](../../experiments/mnist.html#MNISTConfigs) configurations
    and set a new function to calculate the model.
    """
    return Model().to(c.device)


def main():
    # Create experiment
    experiment.create(name='mnist_batch_norm')
    # Create configurations
    conf = MNISTConfigs()
    # Load configurations
    experiment.configs(conf, {'optimizer.optimizer': 'Adam'})
    # Start the experiment and run the training loop
    with experiment.start():
        conf.run()


#
if __name__ == '__main__':
    main()
