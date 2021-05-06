"""
---
title: Deep Convolutional Generative Adversarial Networks (DCGAN)
summary: A simple PyTorch implementation/tutorial of Deep Convolutional Generative Adversarial Networks (DCGAN).
---

# Deep Convolutional Generative Adversarial Networks (DCGAN)

This is a [PyTorch](https://pytorch.org) implementation of paper
[Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434).

This implementation is based on the [PyTorch DCGAN Tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html).
"""

import torch.nn as nn

from labml import experiment
from labml.configs import calculate
from labml_helpers.module import Module
from labml_nn.gan.original.experiment import Configs


class Generator(Module):
    """
    ### Convolutional Generator Network

    This is similar to the de-convolutional network used for CelebA faces,
    but modified for MNIST images.

    <img src="https://pytorch.org/tutorials/_images/dcgan_generator.png" style="max-width:90%" />
    """

    def __init__(self):
        super().__init__()
        # The input is $1 \times 1$ with 100 channels
        self.layers = nn.Sequential(
            # This gives $3 \times 3$ output
            nn.ConvTranspose2d(100, 1024, 3, 1, 0, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            # This gives $7 \times 7$
            nn.ConvTranspose2d(1024, 512, 3, 2, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # This gives $14 \times 14$
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # This gives $28 \times 28$
            nn.ConvTranspose2d(256, 1, 4, 2, 1, bias=False),
            nn.Tanh()
        )

        self.apply(_weights_init)

    def __call__(self, x):
        # Change from shape `[batch_size, 100]` to `[batch_size, 100, 1, 1]`
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = self.layers(x)
        return x


class Discriminator(Module):
    """
    ### Convolutional Discriminator Network
    """

    def __init__(self):
        super().__init__()
        # The input is $28 \times 28$ with one channel
        self.layers = nn.Sequential(
            # This gives $14 \times 14$
            nn.Conv2d(1, 256, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # This gives $7 \times 7$
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # This gives $3 \times 3$
            nn.Conv2d(512, 1024, 3, 2, 0, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            # This gives $1 \times 1$
            nn.Conv2d(1024, 1, 3, 1, 0, bias=False),
        )
        self.apply(_weights_init)

    def forward(self, x):
        x = self.layers(x)
        return x.view(x.shape[0], -1)


def _weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# We import the [simple gan experiment]((simple_mnist_experiment.html) and change the
# generator and discriminator networks
calculate(Configs.generator, 'cnn', lambda c: Generator().to(c.device))
calculate(Configs.discriminator, 'cnn', lambda c: Discriminator().to(c.device))


def main():
    conf = Configs()
    experiment.create(name='mnist_dcgan')
    experiment.configs(conf,
                       {'discriminator': 'cnn',
                        'generator': 'cnn',
                        'label_smoothing': 0.01})
    with experiment.start():
        conf.run()


if __name__ == '__main__':
    main()
