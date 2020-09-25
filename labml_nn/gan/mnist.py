from typing import Optional

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.utils.data

from labml import tracker, monit, experiment
from labml.configs import option, calculate
from labml_helpers.datasets.mnist import MNISTConfigs
from labml_helpers.device import DeviceConfigs
from labml_helpers.module import Module
from labml_helpers.optimizer import OptimizerConfigs
from labml_helpers.train_valid import MODE_STATE, BatchStepProtocol, TrainValidConfigs
from labml_nn.gan import DiscriminatorLogitsLoss, GeneratorLogitsLoss

plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        layer_sizes = [256, 512, 1024]
        layers = []
        d_prev = 100
        for size in layer_sizes:
            layers = layers + [nn.Linear(d_prev, size), nn.LeakyReLU(0.2)]
            d_prev = size

        self.layers = nn.Sequential(*layers, nn.Linear(d_prev, 28 * 28), nn.Tanh())

    def forward(self, x):
        x = self.layers(x)
        x = x.view(x.shape[0], 1, 28, 28)

        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        layer_sizes = [512, 256]
        layers = []
        d_prev = 28 * 28
        for size in layer_sizes:
            layers = layers + [nn.Linear(d_prev, size), nn.LeakyReLU(0.2)]
            d_prev = size

        self.layers = nn.Sequential(*layers, nn.Linear(d_prev, 1))

    def forward(self, x):
        return self.layers(x.view(x.shape[0], -1))


class GANBatchStep(BatchStepProtocol):
    def __init__(self, *,
                 discriminator: Module,
                 generator: Module,
                 discriminator_optimizer: Optional[torch.optim.Adam],
                 generator_optimizer: Optional[torch.optim.Adam],
                 discriminator_loss: DiscriminatorLogitsLoss,
                 generator_loss: GeneratorLogitsLoss):

        self.generator = generator
        self.discriminator = discriminator
        self.generator_loss = generator_loss
        self.discriminator_loss = discriminator_loss
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        tracker.set_scalar("loss.generator.*", True)
        tracker.set_scalar("loss.discriminator.*", True)

    def process(self, batch: any, state: any):
        device = self.discriminator.device
        data, target = batch
        data, target = data.to(device), target.to(device)

        with monit.section("generator"):
            latent = torch.normal(0, 1, (data.shape[0], 100), device=device)
            if MODE_STATE.is_train:
                self.generator_optimizer.zero_grad()
            logits = self.discriminator(self.generator(latent))
            loss = self.generator_loss(logits)
            tracker.add("loss.generator.", loss)
            if MODE_STATE.is_train:
                loss.backward()
                self.generator_optimizer.step()

        with monit.section("discriminator"):
            latent = torch.normal(0, 1, (data.shape[0], 100), device=device)
            if MODE_STATE.is_train:
                self.discriminator_optimizer.zero_grad()
            logits_false = self.discriminator(self.generator(latent).detach())
            logits_true = self.discriminator(data)
            loss = self.discriminator_loss(logits_true, logits_false)
            tracker.add("loss.generator.", loss)
            if MODE_STATE.is_train:
                loss.backward()
                self.discriminator_optimizer.step()

        return {}, None


class Configs(MNISTConfigs, TrainValidConfigs):
    device: torch.device = DeviceConfigs()
    epochs: int = 10

    is_save_models = True
    discriminator: Module
    generator: Module
    generator_optimizer: torch.optim.Adam
    discriminator_optimizer: torch.optim.Adam
    discriminator_loss = DiscriminatorLogitsLoss()
    generator_loss = GeneratorLogitsLoss()
    batch_step = 'gan_batch_step'


@option(Configs.batch_step)
def gan_batch_step(c: Configs):
    return GANBatchStep(discriminator=c.discriminator,
                        generator=c.generator,
                        discriminator_optimizer=c.discriminator_optimizer,
                        generator_optimizer=c.generator_optimizer,
                        discriminator_loss=c.discriminator_loss,
                        generator_loss=c.generator_loss)


calculate(Configs.generator, lambda c: Generator().to(c.device))
calculate(Configs.discriminator, lambda c: Discriminator().to(c.device))


@option(Configs.discriminator_optimizer)
def _discriminator_optimizer(c: Configs):
    opt_conf = OptimizerConfigs()
    opt_conf.parameters = c.discriminator.parameters()
    return opt_conf


@option(Configs.generator_optimizer)
def _generator_optimizer(c: Configs):
    opt_conf = OptimizerConfigs()
    opt_conf.parameters = c.generator.parameters()
    return opt_conf


def main():
    conf = Configs()
    experiment.create(name='mnist_gan', comment='test')
    experiment.configs(conf,
                       {'generator_optimizer.learning_rate': 2.5e-4,
                        'generator_optimizer.optimizer': 'Adam',
                        'discriminator_optimizer.learning_rate': 2.5e-4,
                        'discriminator_optimizer.optimizer': 'Adam'},
                       ['set_seed', 'main'])
    with experiment.start():
        conf.run()


if __name__ == '__main__':
    main()
