"""
---
title: Generative Adversarial Networks experiment with MNIST
summary: This experiment generates MNIST images using multi-layer perceptron.
---

# Generative Adversarial Networks experiment with MNIST
"""

from typing import Any

import torch
import torch.nn as nn
import torch.utils.data
from torchvision import transforms

from labml import tracker, monit, experiment
from labml.configs import option
from labml_helpers.datasets.mnist import MNISTConfigs
from labml_helpers.device import DeviceConfigs
from labml_helpers.module import Module
from labml_helpers.optimizer import OptimizerConfigs
from labml_helpers.train_valid import TrainValidConfigs, hook_model_outputs, BatchIndex
from labml_nn.gan import DiscriminatorLogitsLoss, GeneratorLogitsLoss


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(Module):
    """
    ### Simple MLP Generator

    This has three linear layers of increasing size with `LeakyReLU` activations.
    The final layer has a $tanh$ activation.
    """

    def __init__(self):
        super().__init__()
        layer_sizes = [256, 512, 1024]
        layers = []
        d_prev = 100
        for size in layer_sizes:
            layers = layers + [nn.Linear(d_prev, size), nn.LeakyReLU(0.2)]
            d_prev = size

        self.layers = nn.Sequential(*layers, nn.Linear(d_prev, 28 * 28), nn.Tanh())

        self.apply(weights_init)

    def forward(self, x):
        return self.layers(x).view(x.shape[0], 1, 28, 28)


class Discriminator(Module):
    """
    ### Simple MLP Discriminator

    This has three  linear layers of decreasing size with `LeakyReLU` activations.
    The final layer has a single output that gives the logit of whether input
    is real or fake. You can get the probability by calculating the sigmoid of it.
    """

    def __init__(self):
        super().__init__()
        layer_sizes = [1024, 512, 256]
        layers = []
        d_prev = 28 * 28
        for size in layer_sizes:
            layers = layers + [nn.Linear(d_prev, size), nn.LeakyReLU(0.2)]
            d_prev = size

        self.layers = nn.Sequential(*layers, nn.Linear(d_prev, 1))
        self.apply(weights_init)

    def forward(self, x):
        return self.layers(x.view(x.shape[0], -1))


class Configs(MNISTConfigs, TrainValidConfigs):
    device: torch.device = DeviceConfigs()
    epochs: int = 10

    is_save_models = True
    discriminator: Module
    generator: Module
    generator_optimizer: torch.optim.Adam
    discriminator_optimizer: torch.optim.Adam
    generator_loss: GeneratorLogitsLoss
    discriminator_loss: DiscriminatorLogitsLoss
    label_smoothing: float = 0.2
    discriminator_k: int = 1

    def init(self):
        self.state_modules = []
        self.generator = Generator().to(self.device)
        self.discriminator = Discriminator().to(self.device)
        self.generator_loss = GeneratorLogitsLoss(self.label_smoothing).to(self.device)
        self.discriminator_loss = DiscriminatorLogitsLoss(self.label_smoothing).to(self.device)

        hook_model_outputs(self.mode, self.generator, 'generator')
        hook_model_outputs(self.mode, self.discriminator, 'discriminator')
        tracker.set_scalar("loss.generator.*", True)
        tracker.set_scalar("loss.discriminator.*", True)
        tracker.set_image("generated", True, 1 / 100)

    def step(self, batch: Any, batch_idx: BatchIndex):
        self.generator.train(self.mode.is_train)
        self.discriminator.train(self.mode.is_train)

        data, target = batch[0].to(self.device), batch[1].to(self.device)

        # Increment step in training mode
        if self.mode.is_train:
            tracker.add_global_step(len(data))

        # Train the discriminator
        with monit.section("discriminator"):
            for _ in range(self.discriminator_k):
                latent = torch.randn(data.shape[0], 100, device=self.device)
                logits_true = self.discriminator(data)
                logits_false = self.discriminator(self.generator(latent).detach())
                loss_true, loss_false = self.discriminator_loss(logits_true, logits_false)
                loss = loss_true + loss_false

                # Log stuff
                tracker.add("loss.discriminator.true.", loss_true)
                tracker.add("loss.discriminator.false.", loss_false)
                tracker.add("loss.discriminator.", loss)

                # Train
                if self.mode.is_train:
                    self.discriminator_optimizer.zero_grad()
                    loss.backward()
                    if batch_idx.is_last:
                        tracker.add('discriminator', self.discriminator)
                    self.discriminator_optimizer.step()

        # Train the generator
        with monit.section("generator"):
            latent = torch.randn(data.shape[0], 100, device=self.device)
            generated_images = self.generator(latent)
            logits = self.discriminator(generated_images)
            loss = self.generator_loss(logits)

            # Log stuff
            tracker.add('generated', generated_images[0:5])
            tracker.add("loss.generator.", loss)

            # Train
            if self.mode.is_train:
                self.generator_optimizer.zero_grad()
                loss.backward()
                if batch_idx.is_last:
                    tracker.add('generator', self.generator)
                self.generator_optimizer.step()

        tracker.save()


@option(Configs.dataset_transforms)
def mnist_transforms():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])


@option(Configs.discriminator_optimizer)
def _discriminator_optimizer(c: Configs):
    opt_conf = OptimizerConfigs()
    opt_conf.optimizer = 'Adam'
    opt_conf.parameters = c.discriminator.parameters()
    opt_conf.learning_rate = 2.5e-4
    # Setting exponent decay rate for first moment of gradient,
    # $\beta_`$ to `0.5` is important.
    # Default of `0.9` fails.
    opt_conf.betas = (0.5, 0.999)
    return opt_conf


@option(Configs.generator_optimizer)
def _generator_optimizer(c: Configs):
    opt_conf = OptimizerConfigs()
    opt_conf.optimizer = 'Adam'
    opt_conf.parameters = c.generator.parameters()
    opt_conf.learning_rate = 2.5e-4
    # Setting exponent decay rate for first moment of gradient,
    # $\beta_`$ to `0.5` is important.
    # Default of `0.9` fails.
    opt_conf.betas = (0.5, 0.999)
    return opt_conf


def main():
    conf = Configs()
    experiment.create(name='mnist_gan', comment='test')
    experiment.configs(conf,
                       {'label_smoothing': 0.01})
    with experiment.start():
        conf.run()


if __name__ == '__main__':
    main()
