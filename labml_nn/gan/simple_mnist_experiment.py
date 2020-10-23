"""
# Generative Adversarial Networks experiment with MNIST
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.utils.data
from torchvision import transforms

import labml.utils.pytorch as pytorch_utils
from labml import tracker, monit, experiment
from labml.configs import option, calculate
from labml_helpers.datasets.mnist import MNISTConfigs
from labml_helpers.device import DeviceConfigs
from labml_helpers.module import Module
from labml_helpers.optimizer import OptimizerConfigs
from labml_helpers.train_valid import MODE_STATE, BatchStepProtocol, TrainValidConfigs, hook_model_outputs
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


class GANBatchStep(BatchStepProtocol):
    def __init__(self, *,
                 discriminator: Module,
                 generator: Module,
                 discriminator_optimizer: Optional[torch.optim.Adam],
                 generator_optimizer: Optional[torch.optim.Adam],
                 discriminator_loss: DiscriminatorLogitsLoss,
                 generator_loss: GeneratorLogitsLoss,
                 discriminator_k: int):

        self.discriminator_k = discriminator_k
        self.generator = generator
        self.discriminator = discriminator
        self.generator_loss = generator_loss
        self.discriminator_loss = discriminator_loss
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer

        hook_model_outputs(self.generator, 'generator')
        hook_model_outputs(self.discriminator, 'discriminator')
        tracker.set_scalar("loss.generator.*", True)
        tracker.set_scalar("loss.discriminator.*", True)
        tracker.set_image("generated", True, 1 / 100)

    def prepare_for_iteration(self):
        if MODE_STATE.is_train:
            self.generator.train()
            self.discriminator.train()
        else:
            self.generator.eval()
            self.discriminator.eval()

    def process(self, batch: any, state: any):
        device = self.discriminator.device
        data, target = batch
        data, target = data.to(device), target.to(device)

        # Train the discriminator
        with monit.section("discriminator"):
            for _ in range(self.discriminator_k):
                latent = torch.randn(data.shape[0], 100, device=device)
                if MODE_STATE.is_train:
                    self.discriminator_optimizer.zero_grad()
                logits_true = self.discriminator(data)
                logits_false = self.discriminator(self.generator(latent).detach())
                loss_true, loss_false = self.discriminator_loss(logits_true, logits_false)
                loss = loss_true + loss_false

                # Log stuff
                tracker.add("loss.discriminator.true.", loss_true)
                tracker.add("loss.discriminator.false.", loss_false)
                tracker.add("loss.discriminator.", loss)

                # Train
                if MODE_STATE.is_train:
                    loss.backward()
                    if MODE_STATE.is_log_parameters:
                        pytorch_utils.store_model_indicators(self.discriminator, 'discriminator')
                    self.discriminator_optimizer.step()

        # Train the generator
        with monit.section("generator"):
            latent = torch.randn(data.shape[0], 100, device=device)
            if MODE_STATE.is_train:
                self.generator_optimizer.zero_grad()
            generated_images = self.generator(latent)
            logits = self.discriminator(generated_images)
            loss = self.generator_loss(logits)

            # Log stuff
            tracker.add('generated', generated_images[0:5])
            tracker.add("loss.generator.", loss)

            # Train
            if MODE_STATE.is_train:
                loss.backward()
                if MODE_STATE.is_log_parameters:
                    pytorch_utils.store_model_indicators(self.generator, 'generator')
                self.generator_optimizer.step()

        return {'samples': len(data)}, None


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
    batch_step = 'gan_batch_step'
    label_smoothing: float = 0.2
    discriminator_k: int = 1


@option(Configs.dataset_transforms)
def mnist_transforms():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])


@option(Configs.batch_step)
def gan_batch_step(c: Configs):
    return GANBatchStep(discriminator=c.discriminator,
                        generator=c.generator,
                        discriminator_optimizer=c.discriminator_optimizer,
                        generator_optimizer=c.generator_optimizer,
                        discriminator_loss=c.discriminator_loss,
                        generator_loss=c.generator_loss,
                        discriminator_k=c.discriminator_k)


calculate(Configs.generator, 'mlp', lambda c: Generator().to(c.device))
calculate(Configs.discriminator, 'mlp', lambda c: Discriminator().to(c.device))
calculate(Configs.generator_loss, lambda c: GeneratorLogitsLoss(c.label_smoothing).to(c.device))
calculate(Configs.discriminator_loss, lambda c: DiscriminatorLogitsLoss(c.label_smoothing).to(c.device))


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
                       {'label_smoothing': 0.01},
                       'run')
    with experiment.start():
        conf.run()


if __name__ == '__main__':
    main()
