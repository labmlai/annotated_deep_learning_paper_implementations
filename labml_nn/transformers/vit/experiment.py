"""
---
title: Train a ViT on CIFAR 10
summary: >
  Train a ViT on CIFAR 10
---

#  Train a ViT on CIFAR 10
"""

from labml import experiment
from labml.configs import option
from labml_nn.experiments.cifar10 import CIFAR10Configs
from labml_nn.transformers import TransformerConfigs


class Configs(CIFAR10Configs):
    """
    ## Configurations

    We use [`CIFAR10Configs`](../experiments/cifar10.html) which defines all the
    dataset related configurations, optimizer, and a training loop.
    """

    transformer: TransformerConfigs

    patch_size: int = 4
    n_hidden: int = 2048
    n_classes: int = 10


@option(Configs.transformer)
def _transformer(c: Configs):
    return TransformerConfigs()


@option(Configs.model)
def _vit(c: Configs):
    """
    ### Create model
    """
    from labml_nn.transformers.vit import VisionTransformer, LearnedPositionalEmbeddings, ClassificationHead, \
        PatchEmbeddings

    d_model = c.transformer.d_model
    return VisionTransformer(c.transformer.encoder_layer, c.transformer.n_layers,
                             PatchEmbeddings(d_model, c.patch_size, 3),
                             LearnedPositionalEmbeddings(d_model),
                             ClassificationHead(d_model, c.n_hidden, c.n_classes)).to(c.device)


def main():
    # Create experiment
    experiment.create(name='ViT', comment='cifar10')
    # Create configurations
    conf = Configs()
    # Load configurations
    experiment.configs(conf, {
        'device.cuda_device': 0,

        # 'optimizer.optimizer': 'Noam',
        # 'optimizer.learning_rate': 1.,
        'optimizer.optimizer': 'Adam',
        'optimizer.learning_rate': 2.5e-4,
        'optimizer.d_model': 512,

        'transformer.d_model': 512,

        'epochs': 1000,
        'train_batch_size': 64,

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
