"""
---
title: Train ConvMixer on CIFAR 10
summary: >
  Train ConvMixer on CIFAR 10
---

#  Train a [ConvMixer](index.html) on CIFAR 10

[![View Run](https://img.shields.io/badge/labml-experiment-brightgreen)](https://app.labml.ai/run/8b531d9ce3dc11eb84fc87df6756eb8f)
"""

from labml import experiment
from labml.configs import option
from labml_nn.experiments.cifar10 import CIFAR10Configs


class Configs(CIFAR10Configs):
    """
    ## Configurations

    We use [`CIFAR10Configs`](../experiments/cifar10.html) which defines all the
    dataset related configurations, optimizer, and a training loop.
    """

    # Size of a patch
    patch_size: int = 2
    d_model: int = 256
    n_layers: int = 8
    kernel_size: int = 7
    # Number of classes in the task
    n_classes: int = 10


@option(Configs.model)
def _conv_mixer(c: Configs):
    """
    ### Create model
    """
    from labml_nn.conv_mixer import ConvMixerLayer, ConvMixer, ClassificationHead, PatchEmbeddings

    # Create ConvMixer
    return ConvMixer(ConvMixerLayer(c.d_model, c.kernel_size), c.n_layers,
                     PatchEmbeddings(c.d_model, c.patch_size, 3),
                     ClassificationHead(c.d_model, c.n_classes)).to(c.device)


def main():
    # Create experiment
    experiment.create(name='ConvMixer', comment='cifar10')
    # Create configurations
    conf = Configs()
    # Load configurations
    experiment.configs(conf, {
        # Optimizer
        'optimizer.optimizer': 'Adam',
        'optimizer.learning_rate': 2.5e-4,

        # Training epochs and batch size
        'epochs': 150,
        'train_batch_size': 64,

        # Augment CIFAR 10 images for training
        'train_dataset': 'cifar10_train_augmented',
        # Do not augment CIFAR 10 images for validation
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
