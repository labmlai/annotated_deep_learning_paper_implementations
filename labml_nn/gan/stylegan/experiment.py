"""
---
title: Style GAN 2 Model Training
summary: >
 An annotated PyTorch implementation of StyleGAN2 model training code.
---

# [Style GAN 2](index.html) Model Training

Here's the training code for [Style GAN 2](index.html) model.
"""

import math
from pathlib import Path
from typing import Any, Iterator, Tuple

import torch
import torch.utils.data
import torchvision
from PIL import Image

from labml import tracker, lab, monit, experiment
from labml.configs import BaseConfigs
from labml_helpers.device import DeviceConfigs
from labml_helpers.train_valid import ModeState, hook_model_outputs
from labml_nn.gan.stylegan import Discriminator, Generator, MappingNetwork, GradientPenalty, PathLengthPenalty
from labml_nn.gan.wasserstein import DiscriminatorLoss, GeneratorLoss
from labml_nn.utils import cycle_dataloader


class Dataset(torch.utils.data.Dataset):
    """
    ## Dataset

    This loads the training dataset and resize it to the give image size.

    We trained this on [CelebA-HQ dataset](https://github.com/tkarras/progressive_growing_of_gans).
    You can find the download instruction in this
    [discussion on fast.ai](https://forums.fast.ai/t/download-celeba-hq-dataset/45873/3).
    Save the images inside `data/stylegan` folder.
    """
    def __init__(self, image_size: int):
        """
        * `image_size` size of the image
        """
        super().__init__()

        # Place the images in `data/stylegan2`.
        # `get_data_path` returns the data path set in `.labml.yaml` which defaults to `./data`.
        self.folder = lab.get_data_path() / 'stylegan2'
        # Get the paths of all `jpg` files
        self.paths = [p for p in Path(f'{self.folder}').glob(f'**/*.jpg')]

        # Transformation
        self.transform = torchvision.transforms.Compose([
            # Resize the image
            torchvision.transforms.Resize(image_size),
            # Convert to PyTorch tensor
            torchvision.transforms.ToTensor(),
        ])

    def __len__(self):
        """Number of images"""
        return len(self.paths)

    def __getitem__(self, index):
        """Get the the `index`-th image"""
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)


class Configs(BaseConfigs):
    device: torch.device = DeviceConfigs()
    data_loader: Any

    discriminator: Discriminator
    generator: Generator

    discriminator_loss: DiscriminatorLoss
    generator_loss: GeneratorLoss

    generator_optimizer: torch.optim.Adam
    discriminator_optimizer: torch.optim.Adam
    mapping_network_optimizer: torch.optim.Adam

    mapping_network: MappingNetwork

    gradient_penalty_coefficient: float = 10.
    gradient_penalty = GradientPenalty()

    batch_size: int = 32
    d_latent: int = 512
    image_size: int = 32
    n_gen_blocks: int
    mapping_network_layers: int = 8
    learning_rate: float = 1e-3
    mapping_network_learning_rate: float = 1e-5
    gradient_accumulate_steps: int = 1
    betas: Tuple[float, float] = (0.0, 0.99)
    style_mixing_prob: float = 0.9

    path_length_penalty: PathLengthPenalty

    dataset: Dataset
    loader: Iterator

    lazy_gradient_penalty_interval: int = 4
    lazy_path_penalty_interval: int = 32
    lazy_path_penalty_after: int = 5_000

    log_generated_interval: int = 500
    save_checkpoint_interval: int = 2_000

    mode: ModeState

    log_layer_outputs: bool = False

    def init(self):
        self.dataset = Dataset(self.image_size)
        dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size,
                                                 num_workers=32,
                                                 shuffle=True, drop_last=True, pin_memory=True)
        self.loader = cycle_dataloader(dataloader)
        log_resolution = int(math.log2(self.image_size))

        self.discriminator = Discriminator(log_resolution).to(self.device)
        self.generator = Generator(log_resolution, self.d_latent).to(self.device)
        self.n_gen_blocks = self.generator.n_blocks
        self.mapping_network = MappingNetwork(self.d_latent, self.mapping_network_layers).to(self.device)
        self.path_length_penalty = PathLengthPenalty(0.99).to(self.device)

        if self.log_layer_outputs:
            hook_model_outputs(self.mode, self.discriminator, 'discriminator')
            hook_model_outputs(self.mode, self.generator, 'generator')
            hook_model_outputs(self.mode, self.mapping_network, 'mapping_network')

        self.discriminator_loss = DiscriminatorLoss().to(self.device)
        self.generator_loss = GeneratorLoss().to(self.device)

        self.discriminator_optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.learning_rate, betas=self.betas
        )
        self.generator_optimizer = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.learning_rate, betas=self.betas
        )
        self.mapping_network_optimizer = torch.optim.Adam(
            self.mapping_network.parameters(),
            lr=self.mapping_network_learning_rate, betas=self.betas
        )

        tracker.set_image("generated", True)

    def get_w(self, batch_size: int):
        if torch.rand(()).item() < self.style_mixing_prob:
            cross_over_point = int(torch.rand(()).item() * self.n_gen_blocks)
            z1 = torch.randn(batch_size, self.d_latent).to(self.device)
            w1 = self.mapping_network(z1)
            z2 = torch.randn(batch_size, self.d_latent).to(self.device)
            w2 = self.mapping_network(z2)
            w1 = w1[None, :, :].expand(cross_over_point, -1, -1)
            w2 = w2[None, :, :].expand(self.n_gen_blocks - cross_over_point, -1, -1)
            return torch.cat((w1, w2), dim=0)
        else:
            z = torch.randn(batch_size, self.d_latent).to(self.device)
            w = self.mapping_network(z)
            return w[None, :, :].expand(self.n_gen_blocks, -1, -1)

    def image_noise(self, batch_size):
        noise = []
        r = 4
        for i in range(self.n_gen_blocks):
            if i == 0:
                n1 = None
            else:
                n1 = torch.randn(batch_size, 1, r, r, device=self.device)
            n2 = torch.randn(batch_size, 1, r, r, device=self.device)

            noise.append((n1, n2))
            r *= 2

        return noise

    def generate_images(self, batch_size: int):
        w_style = self.get_w(batch_size)
        noise = self.image_noise(batch_size)

        generated_images = self.generator(w_style, noise)

        return generated_images, w_style

    def step(self, idx):
        with monit.section('Discriminator'):
            self.discriminator_optimizer.zero_grad()

            for i in range(self.gradient_accumulate_steps):
                with self.mode.update(is_log_activations=(idx + 1) % self.log_generated_interval == 0):
                    generated_images, _ = self.generate_images(self.batch_size)
                    fake_output = self.discriminator(generated_images.detach())

                    real_images = next(self.loader).to(self.device)
                    if (idx + 1) % self.lazy_gradient_penalty_interval == 0:
                        real_images.requires_grad_()
                    real_output = self.discriminator(real_images)

                    real_loss, fake_loss = self.discriminator_loss(real_output, fake_output)
                    disc_loss = real_loss + fake_loss

                    if (idx + 1) % self.lazy_gradient_penalty_interval == 0:
                        gp = self.gradient_penalty(real_images, real_output)
                        tracker.add('loss.gp', gp)
                        disc_loss = disc_loss + 0.5 * self.gradient_penalty_coefficient * gp * self.lazy_gradient_penalty_interval

                    disc_loss.backward()

                    tracker.add('loss.discriminator', disc_loss)

            if (idx + 1) % self.log_generated_interval == 0:
                tracker.add('discriminator', self.discriminator)
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
            self.discriminator_optimizer.step()

        with monit.section('Generator'):
            self.generator_optimizer.zero_grad()
            self.mapping_network_optimizer.zero_grad()

            for i in range(self.gradient_accumulate_steps):
                generated_images, w_style = self.generate_images(self.batch_size)
                fake_output = self.discriminator(generated_images)

                gen_loss = self.generator_loss(fake_output)

                if idx > self.lazy_path_penalty_after and (idx + 1) % self.lazy_path_penalty_interval == 0:
                    ppl = self.path_length_penalty(w_style, generated_images)
                    if not torch.isnan(ppl):
                        tracker.add('loss.ppl', ppl)
                        gen_loss = gen_loss + ppl

                gen_loss.backward()

                tracker.add('loss.generator', gen_loss)

            if (idx + 1) % self.log_generated_interval == 0:
                tracker.add('generator', self.generator)
                tracker.add('mapping_network', self.mapping_network)

            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(self.mapping_network.parameters(), max_norm=1.0)

            self.generator_optimizer.step()
            self.mapping_network_optimizer.step()

        if (idx + 1) % self.log_generated_interval == 0:
            tracker.add('generated', torch.cat([generated_images[:6], real_images[:3]], dim=0))
        if (idx + 1) % self.save_checkpoint_interval == 0:
            experiment.save_checkpoint()

        tracker.save()

    def train(self):
        for i in monit.loop(150_000):
            self.step(i)
            if (i + 1) % self.log_generated_interval == 0:
                tracker.new_line()


def main():
    configs = Configs()
    experiment.create(name='stylegan', writers={'screen'})
    experiment.configs(configs, {
        'device.cuda_device': 1,
        'image_size': 64,
        'log_generated_interval': 50 * 4
    })

    configs.init()
    experiment.add_pytorch_models(mapping_network=configs.mapping_network,
                                  generator=configs.generator,
                                  discriminator=configs.discriminator)

    with experiment.start():
        configs.train()


if __name__ == '__main__':
    main()
