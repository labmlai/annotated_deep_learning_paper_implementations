"""
---
title: Cycle GAN
summary: >
  A simple PyTorch implementation/tutorial of Cycle GAN introduced in paper
  Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks.
---

# Cycle GAN

This is a [PyTorch](https://pytorch.org) implementation/tutorial of the paper
[Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593).

I've taken pieces of code from [eriklindernoren/PyTorch-GAN](https://github.com/eriklindernoren/PyTorch-GAN).
It is a very good resource if you want to checkout other GAN variations too.

Cycle GAN does image-to-image translation.
It trains a model to translate an image from given distribution to another, say, images of class A and B.
Images of a certain distribution could be things like images of a certain style, or nature.
The models do not need paired images between A and B.
Just a set of images of each class is enough.
This works very well on changing between image styles, lighting changes, pattern changes, etc.
For example, changing summer to winter, painting style to photos, and horses to zebras.

Cycle GAN trains two generator models and two discriminator models.
One generator translates images from A to B and the other from B to A.
The discriminators test whether the generated images look real.

This file contains the model code as well as the training code.
We also have a Google Colab notebook.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lab-ml/nn/blob/master/labml_nn/gan/cycle_gan/experiment.ipynb)
[![View Run](https://img.shields.io/badge/labml-experiment-brightgreen)](https://app.labml.ai/run/93b11a665d6811ebaac80242ac1c0002)
"""

import itertools
import random
import zipfile
from typing import Tuple

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import make_grid

from labml import lab, tracker, experiment, monit
from labml.configs import BaseConfigs
from labml.utils.download import download_file
from labml.utils.pytorch import get_modules
from labml_helpers.device import DeviceConfigs
from labml_helpers.module import Module


class GeneratorResNet(Module):
    """
    The generator is a residual network.
    """

    def __init__(self, input_channels: int, n_residual_blocks: int):
        super().__init__()
        # This first block runs a $7\times7$ convolution and maps the image to
        # a feature map.
        # The output feature map has the same height and width because we have
        # a padding of $3$.
        # Reflection padding is used because it gives better image quality at edges.
        #
        # `inplace=True` in `ReLU` saves a little bit of memory.
        out_features = 64
        layers = [
            nn.Conv2d(input_channels, out_features, kernel_size=7, padding=3, padding_mode='reflect'),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        ]
        in_features = out_features

        # We down-sample with two $3 \times 3$ convolutions
        # with stride of 2
        for _ in range(2):
            out_features *= 2
            layers += [
                nn.Conv2d(in_features, out_features, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # We take this through `n_residual_blocks`.
        # This module is defined below.
        for _ in range(n_residual_blocks):
            layers += [ResidualBlock(out_features)]

        # Then the resulting feature map is up-sampled
        # to match the original image height and width.
        for _ in range(2):
            out_features //= 2
            layers += [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_features, out_features, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features

        # Finally we map the feature map to an RGB image
        layers += [nn.Conv2d(out_features, input_channels, 7, padding=3, padding_mode='reflect'), nn.Tanh()]

        # Create a sequential module with the layers
        self.layers = nn.Sequential(*layers)

        # Initialize weights to $\mathcal{N}(0, 0.2)$
        self.apply(weights_init_normal)

    def __call__(self, x):
        return self.layers(x)


class ResidualBlock(Module):
    """
    This is the residual block, with two convolution layers.
    """

    def __init__(self, in_features: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features, in_features, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
        )

    def __call__(self, x: torch.Tensor):
        return x + self.block(x)


class Discriminator(Module):
    """
    This is the discriminator.
    """

    def __init__(self, input_shape: Tuple[int, int, int]):
        super().__init__()
        channels, height, width = input_shape

        # Output of the discriminator is also a map of probabilities*
        # whether each region of the image is real or generated
        self.output_shape = (1, height // 2 ** 4, width // 2 ** 4)

        self.layers = nn.Sequential(
            # Each of these blocks will shrink the height and width by a factor of 2
            DiscriminatorBlock(channels, 64, normalize=False),
            DiscriminatorBlock(64, 128),
            DiscriminatorBlock(128, 256),
            DiscriminatorBlock(256, 512),
            # Zero pad on top and left to keep the output height and width same
            # with the $4 \times 4$ kernel
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, kernel_size=4, padding=1)
        )

        # Initialize weights to $\mathcal{N}(0, 0.2)$
        self.apply(weights_init_normal)

    def forward(self, img):
        return self.layers(img)


class DiscriminatorBlock(Module):
    """
    This is the discriminator block module.
    It does a convolution, an optional normalization, and a leaky ReLU.

    It shrinks the height and width of the input feature map by half.
    """

    def __init__(self, in_filters: int, out_filters: int, normalize: bool = True):
        super().__init__()
        layers = [nn.Conv2d(in_filters, out_filters, kernel_size=4, stride=2, padding=1)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_filters))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.layers = nn.Sequential(*layers)

    def __call__(self, x: torch.Tensor):
        return self.layers(x)


def weights_init_normal(m):
    """
    Initialize convolution layer weights to $\mathcal{N}(0, 0.2)$
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)


def load_image(path: str):
    """
    Load an image and change to RGB if in grey-scale.
    """
    image = Image.open(path)
    if image.mode != 'RGB':
        image = Image.new("RGB", image.size).paste(image)

    return image


class ImageDataset(Dataset):
    """
    ### Dataset to load images
    """

    @staticmethod
    def download(dataset_name: str):
        """
        #### Download dataset and extract data
        """
        # URL
        url = f'https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/{dataset_name}.zip'
        # Download folder
        root = lab.get_data_path() / 'cycle_gan'
        if not root.exists():
            root.mkdir(parents=True)
        # Download destination
        archive = root / f'{dataset_name}.zip'
        # Download file (generally ~100MB)
        download_file(url, archive)
        # Extract the archive
        with zipfile.ZipFile(archive, 'r') as f:
            f.extractall(root)

    def __init__(self, dataset_name: str, transforms_, mode: str):
        """
        #### Initialize the dataset

        * `dataset_name` is the name of the dataset
        * `transforms_` is the set of image transforms
        * `mode` is either `train` or `test`
        """
        # Dataset path
        root = lab.get_data_path() / 'cycle_gan' / dataset_name
        # Download if missing
        if not root.exists():
            self.download(dataset_name)

        # Image transforms
        self.transform = transforms.Compose(transforms_)

        # Get image paths
        path_a = root / f'{mode}A'
        path_b = root / f'{mode}B'
        self.files_a = sorted(str(f) for f in path_a.iterdir())
        self.files_b = sorted(str(f) for f in path_b.iterdir())

    def __getitem__(self, index):
        # Return a pair of images.
        # These pairs get batched together, and they do not act like pairs in training.
        # So it is kind of ok that we always keep giving the same pair.
        return {"x": self.transform(load_image(self.files_a[index % len(self.files_a)])),
                "y": self.transform(load_image(self.files_b[index % len(self.files_b)]))}

    def __len__(self):
        # Number of images in the dataset
        return max(len(self.files_a), len(self.files_b))


class ReplayBuffer:
    """
    ### Replay Buffer

    Replay buffer is used to train the discriminator.
    Generated images are added to the replay buffer and sampled from it.

    The replay buffer returns the newly added image with a probability of $0.5$.
    Otherwise, it sends an older generated image and replaces the older image
    with the newly generated image.

    This is done to reduce model oscillation.
    """

    def __init__(self, max_size: int = 50):
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data: torch.Tensor):
        """Add/retrieve an image"""
        data = data.detach()
        res = []
        for element in data:
            if len(self.data) < self.max_size:
                self.data.append(element)
                res.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    res.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    res.append(element)
        return torch.stack(res)


class Configs(BaseConfigs):
    """## Configurations"""

    # `DeviceConfigs` will pick a GPU if available
    device: torch.device = DeviceConfigs()

    # Hyper-parameters
    epochs: int = 200
    dataset_name: str = 'monet2photo'
    batch_size: int = 1

    data_loader_workers = 8

    learning_rate = 0.0002
    adam_betas = (0.5, 0.999)
    decay_start = 100

    # The paper suggests using a least-squares loss instead of
    # negative log-likelihood, at it is found to be more stable.
    gan_loss = torch.nn.MSELoss()

    # L1 loss is used for cycle loss and identity loss
    cycle_loss = torch.nn.L1Loss()
    identity_loss = torch.nn.L1Loss()

    # Image dimensions
    img_height = 256
    img_width = 256
    img_channels = 3

    # Number of residual blocks in the generator
    n_residual_blocks = 9

    # Loss coefficients
    cyclic_loss_coefficient = 10.0
    identity_loss_coefficient = 5.

    sample_interval = 500

    # Models
    generator_xy: GeneratorResNet
    generator_yx: GeneratorResNet
    discriminator_x: Discriminator
    discriminator_y: Discriminator

    # Optimizers
    generator_optimizer: torch.optim.Adam
    discriminator_optimizer: torch.optim.Adam

    # Learning rate schedules
    generator_lr_scheduler: torch.optim.lr_scheduler.LambdaLR
    discriminator_lr_scheduler: torch.optim.lr_scheduler.LambdaLR

    # Data loaders
    dataloader: DataLoader
    valid_dataloader: DataLoader

    def sample_images(self, n: int):
        """Generate samples from test set and save them"""
        batch = next(iter(self.valid_dataloader))
        self.generator_xy.eval()
        self.generator_yx.eval()
        with torch.no_grad():
            data_x, data_y = batch['x'].to(self.generator_xy.device), batch['y'].to(self.generator_yx.device)
            gen_y = self.generator_xy(data_x)
            gen_x = self.generator_yx(data_y)

            # Arrange images along x-axis
            data_x = make_grid(data_x, nrow=5, normalize=True)
            data_y = make_grid(data_y, nrow=5, normalize=True)
            gen_x = make_grid(gen_x, nrow=5, normalize=True)
            gen_y = make_grid(gen_y, nrow=5, normalize=True)

            # Arrange images along y-axis
            image_grid = torch.cat((data_x, gen_y, data_y, gen_x), 1)

        # Show samples
        plot_image(image_grid)

    def initialize(self):
        """
        ## Initialize models and data loaders
        """
        input_shape = (self.img_channels, self.img_height, self.img_width)

        # Create the models
        self.generator_xy = GeneratorResNet(self.img_channels, self.n_residual_blocks).to(self.device)
        self.generator_yx = GeneratorResNet(self.img_channels, self.n_residual_blocks).to(self.device)
        self.discriminator_x = Discriminator(input_shape).to(self.device)
        self.discriminator_y = Discriminator(input_shape).to(self.device)

        # Create the optmizers
        self.generator_optimizer = torch.optim.Adam(
            itertools.chain(self.generator_xy.parameters(), self.generator_yx.parameters()),
            lr=self.learning_rate, betas=self.adam_betas)
        self.discriminator_optimizer = torch.optim.Adam(
            itertools.chain(self.discriminator_x.parameters(), self.discriminator_y.parameters()),
            lr=self.learning_rate, betas=self.adam_betas)

        # Create the learning rate schedules.
        # The learning rate stars flat until `decay_start` epochs,
        # and then linearly reduce to $0$ at end of training.
        decay_epochs = self.epochs - self.decay_start
        self.generator_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.generator_optimizer, lr_lambda=lambda e: 1.0 - max(0, e - self.decay_start) / decay_epochs)
        self.discriminator_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.discriminator_optimizer, lr_lambda=lambda e: 1.0 - max(0, e - self.decay_start) / decay_epochs)

        # Image transformations
        transforms_ = [
            transforms.Resize(int(self.img_height * 1.12), Image.BICUBIC),
            transforms.RandomCrop((self.img_height, self.img_width)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]

        # Training data loader
        self.dataloader = DataLoader(
            ImageDataset(self.dataset_name, transforms_, 'train'),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.data_loader_workers,
        )

        # Validation data loader
        self.valid_dataloader = DataLoader(
            ImageDataset(self.dataset_name, transforms_, "test"),
            batch_size=5,
            shuffle=True,
            num_workers=self.data_loader_workers,
        )

    def run(self):
        """
        ## Training

        We aim to solve:
        $$G^{*}, F^{*} = \arg \min_{G,F} \max_{D_X, D_Y} \mathcal{L}(G, F, D_X, D_Y)$$

        where,
        $G$ translates images from $X \rightarrow Y$,
        $F$ translates images from $Y \rightarrow X$,
        $D_X$ tests if images are from $X$ space,
        $D_Y$ tests if images are from $Y$ space, and
        \begin{align}
        \mathcal{L}(G, F, D_X, D_Y)
            &= \mathcal{L}_{GAN}(G, D_Y, X, Y) \\
            &+ \mathcal{L}_{GAN}(F, D_X, Y, X) \\
            &+ \lambda_1 \mathcal{L}_{cyc}(G, F) \\
            &+ \lambda_2 \mathcal{L}_{identity}(G, F) \\
        \\
        \mathcal{L}_{GAN}(G, F, D_Y, X, Y)
            &= \mathbb{E}_{y \sim p_{data}(y)} \Big[log D_Y(y)\Big] \\
            &+ \mathbb{E}_{x \sim p_{data}(x)} \bigg[log\Big(1 - D_Y(G(x))\Big)\bigg] \\
            &+ \mathbb{E}_{x \sim p_{data}(x)} \Big[log D_X(x)\Big] \\
            &+ \mathbb{E}_{y \sim p_{data}(y)} \bigg[log\Big(1 - D_X(F(y))\Big)\bigg] \\
        \\
        \mathcal{L}_{cyc}(G, F)
            &= \mathbb{E}_{x \sim p_{data}(x)} \Big[\lVert F(G(x)) - x \lVert_1\Big] \\
            &+ \mathbb{E}_{y \sim p_{data}(y)} \Big[\lVert G(F(y)) - y \rVert_1\Big] \\
        \\
        \mathcal{L}_{identity}(G, F)
            &= \mathbb{E}_{x \sim p_{data}(x)} \Big[\lVert F(x) - x \lVert_1\Big] \\
            &+ \mathbb{E}_{y \sim p_{data}(y)} \Big[\lVert G(y) - y \rVert_1\Big] \\
        \end{align}

        $\mathcal{L}_{GAN}$ is the generative adversarial loss from the original
        GAN paper.

        $\mathcal{L}_{cyc}$ is the cyclic loss, where we try to get $F(G(x))$ to be similar to $x$,
        and $G(F(y))$ to be similar to $y$.
        Basically if the two generators (transformations) are applied in series it should give back the
        original image.
        This is the main contribution of this paper.
        It trains the generators to generate an image of the other distribution that is similar to
        the original image.
        Without this loss $G(x)$ could generate anything that's from the distribution of $Y$.
        Now it needs to generate something from the distribution of $Y$ but still has properties of $x$,
        so that $F(G(x)$ can re-generate something like $x$.

        $\mathcal{L}_{cyc}$ is the identity loss.
        This was used to encourage the mapping to preserve color composition between
        the input and the output.

        To solve $G^{\*}, F^{\*}$,
        discriminators $D_X$ and $D_Y$ should **ascend** on the gradient,
        \begin{align}
        \nabla_{\theta_{D_X, D_Y}} \frac{1}{m} \sum_{i=1}^m
        &\Bigg[
        \log D_Y\Big(y^{(i)}\Big) \\
        &+ \log \Big(1 - D_Y\Big(G\Big(x^{(i)}\Big)\Big)\Big) \\
        &+ \log D_X\Big(x^{(i)}\Big) \\
        & +\log\Big(1 - D_X\Big(F\Big(y^{(i)}\Big)\Big)\Big)
        \Bigg]
        \end{align}
        That is descend on *negative* log-likelihood loss.

        In order to stabilize the training the negative log- likelihood objective
        was replaced by a least-squared loss -
        the least-squared error of discriminator, labelling real images with 1,
        and generated images with 0.
        So we want to descend on the gradient,
        \begin{align}
        \nabla_{\theta_{D_X, D_Y}} \frac{1}{m} \sum_{i=1}^m
        &\Bigg[
            \bigg(D_Y\Big(y^{(i)}\Big) - 1\bigg)^2 \\
            &+ D_Y\Big(G\Big(x^{(i)}\Big)\Big)^2 \\
            &+ \bigg(D_X\Big(x^{(i)}\Big) - 1\bigg)^2 \\
            &+ D_X\Big(F\Big(y^{(i)}\Big)\Big)^2
        \Bigg]
        \end{align}

        We use least-squares for generators also.
        The generators should *descend* on the gradient,
        \begin{align}
        \nabla_{\theta_{F, G}} \frac{1}{m} \sum_{i=1}^m
        &\Bigg[
            \bigg(D_Y\Big(G\Big(x^{(i)}\Big)\Big) - 1\bigg)^2 \\
            &+ \bigg(D_X\Big(F\Big(y^{(i)}\Big)\Big) - 1\bigg)^2 \\
            &+ \mathcal{L}_{cyc}(G, F)
            + \mathcal{L}_{identity}(G, F)
        \Bigg]
        \end{align}

        We use `generator_xy` for $G$ and `generator_yx$ for $F$.
        We use `discriminator_x$ for $D_X$ and `discriminator_y` for $D_Y$.
        """

        # Replay buffers to keep generated samples
        gen_x_buffer = ReplayBuffer()
        gen_y_buffer = ReplayBuffer()

        # Loop through epochs
        for epoch in monit.loop(self.epochs):
            # Loop through the dataset
            for i, batch in monit.enum('Train', self.dataloader):
                # Move images to the device
                data_x, data_y = batch['x'].to(self.device), batch['y'].to(self.device)

                # true labels equal to $1$
                true_labels = torch.ones(data_x.size(0), *self.discriminator_x.output_shape,
                                         device=self.device, requires_grad=False)
                # false labels equal to $0$
                false_labels = torch.zeros(data_x.size(0), *self.discriminator_x.output_shape,
                                           device=self.device, requires_grad=False)

                # Train the generators.
                # This returns the generated images.
                gen_x, gen_y = self.optimize_generators(data_x, data_y, true_labels)

                #  Train discriminators
                self.optimize_discriminator(data_x, data_y,
                                            gen_x_buffer.push_and_pop(gen_x), gen_y_buffer.push_and_pop(gen_y),
                                            true_labels, false_labels)

                # Save training statistics and increment the global step counter
                tracker.save()
                tracker.add_global_step(max(len(data_x), len(data_y)))

                # Save images at intervals
                batches_done = epoch * len(self.dataloader) + i
                if batches_done % self.sample_interval == 0:
                    # Save models when sampling images
                    experiment.save_checkpoint()
                    # Sample images
                    self.sample_images(batches_done)

            # Update learning rates
            self.generator_lr_scheduler.step()
            self.discriminator_lr_scheduler.step()
            # New line
            tracker.new_line()

    def optimize_generators(self, data_x: torch.Tensor, data_y: torch.Tensor, true_labels: torch.Tensor):
        """
        ### Optimize the generators with identity, gan and cycle losses.
        """

        #  Change to training mode
        self.generator_xy.train()
        self.generator_yx.train()

        # Identity loss
        # $$\lVert F(G(x^{(i)})) - x^{(i)} \lVert_1\
        #   \lVert G(F(y^{(i)})) - y^{(i)} \rVert_1$$
        loss_identity = (self.identity_loss(self.generator_yx(data_x), data_x) +
                         self.identity_loss(self.generator_xy(data_y), data_y))

        # Generate images $G(x)$ and $F(y)$
        gen_y = self.generator_xy(data_x)
        gen_x = self.generator_yx(data_y)

        # GAN loss
        # $$\bigg(D_Y\Big(G\Big(x^{(i)}\Big)\Big) - 1\bigg)^2
        #  + \bigg(D_X\Big(F\Big(y^{(i)}\Big)\Big) - 1\bigg)^2$$
        loss_gan = (self.gan_loss(self.discriminator_y(gen_y), true_labels) +
                    self.gan_loss(self.discriminator_x(gen_x), true_labels))

        # Cycle loss
        # $$
        # \lVert F(G(x^{(i)})) - x^{(i)} \lVert_1 +
        # \lVert G(F(y^{(i)})) - y^{(i)} \rVert_1
        # $$
        loss_cycle = (self.cycle_loss(self.generator_yx(gen_y), data_x) +
                      self.cycle_loss(self.generator_xy(gen_x), data_y))

        # Total loss
        loss_generator = (loss_gan +
                          self.cyclic_loss_coefficient * loss_cycle +
                          self.identity_loss_coefficient * loss_identity)

        # Take a step in the optimizer
        self.generator_optimizer.zero_grad()
        loss_generator.backward()
        self.generator_optimizer.step()

        # Log losses
        tracker.add({'loss.generator': loss_generator,
                     'loss.generator.cycle': loss_cycle,
                     'loss.generator.gan': loss_gan,
                     'loss.generator.identity': loss_identity})

        # Return generated images
        return gen_x, gen_y

    def optimize_discriminator(self, data_x: torch.Tensor, data_y: torch.Tensor,
                               gen_x: torch.Tensor, gen_y: torch.Tensor,
                               true_labels: torch.Tensor, false_labels: torch.Tensor):
        """
        ### Optimize the discriminators with gan loss.
        """
        # GAN Loss
        # \begin{align}
        # \bigg(D_Y\Big(y ^ {(i)}\Big) - 1\bigg) ^ 2
        # + D_Y\Big(G\Big(x ^ {(i)}\Big)\Big) ^ 2 + \\
        # \bigg(D_X\Big(x ^ {(i)}\Big) - 1\bigg) ^ 2
        # + D_X\Big(F\Big(y ^ {(i)}\Big)\Big) ^ 2
        # \end{align}
        loss_discriminator = (self.gan_loss(self.discriminator_x(data_x), true_labels) +
                              self.gan_loss(self.discriminator_x(gen_x), false_labels) +
                              self.gan_loss(self.discriminator_y(data_y), true_labels) +
                              self.gan_loss(self.discriminator_y(gen_y), false_labels))

        # Take a step in the optimizer
        self.discriminator_optimizer.zero_grad()
        loss_discriminator.backward()
        self.discriminator_optimizer.step()

        # Log losses
        tracker.add({'loss.discriminator': loss_discriminator})


def train():
    """
    ## Train Cycle GAN
    """
    # Create configurations
    conf = Configs()
    # Create an experiment
    experiment.create(name='cycle_gan')
    # Calculate configurations.
    # It will calculate `conf.run` and all other configs required by it.
    experiment.configs(conf, {'dataset_name': 'summer2winter_yosemite'})
    conf.initialize()

    # Register models for saving and loading.
    # `get_modules` gives a dictionary of `nn.Modules` in `conf`.
    # You can also specify a custom dictionary of models.
    experiment.add_pytorch_models(get_modules(conf))
    # Start and watch the experiment
    with experiment.start():
        # Run the training
        conf.run()


def plot_image(img: torch.Tensor):
    """
    ### Plot an image with matplotlib
    """
    from matplotlib import pyplot as plt

    # Move tensor to CPU
    img = img.cpu()
    # Get min and max values of the image for normalization
    img_min, img_max = img.min(), img.max()
    # Scale image values to be [0...1]
    img = (img - img_min) / (img_max - img_min + 1e-5)
    # We have to change the order of dimensions to HWC.
    img = img.permute(1, 2, 0)
    # Show Image
    plt.imshow(img)
    # We don't need axes
    plt.axis('off')
    # Display
    plt.show()


def evaluate():
    """
    ## Evaluate trained Cycle GAN
    """
    # Set the run UUID from the training run
    trained_run_uuid = 'f73c1164184711eb9190b74249275441'
    # Create configs object
    conf = Configs()
    # Create experiment
    experiment.create(name='cycle_gan_inference')
    # Load hyper parameters set for training
    conf_dict = experiment.load_configs(trained_run_uuid)
    # Calculate configurations. We specify the generators `'generator_xy', 'generator_yx'`
    # so that it only loads those and their dependencies.
    # Configs like `device` and `img_channels` will be calculated, since these are required by
    # `generator_xy` and `generator_yx`.
    #
    # If you want other parameters like `dataset_name` you should specify them here.
    # If you specify nothing, all the configurations will be calculated, including data loaders.
    # Calculation of configurations and their dependencies will happen when you call `experiment.start`
    experiment.configs(conf, conf_dict)
    conf.initialize()

    # Register models for saving and loading.
    # `get_modules` gives a dictionary of `nn.Modules` in `conf`.
    # You can also specify a custom dictionary of models.
    experiment.add_pytorch_models(get_modules(conf))
    # Specify which run to load from.
    # Loading will actually happen when you call `experiment.start`
    experiment.load(trained_run_uuid)

    # Start the experiment
    with experiment.start():
        # Image transformations
        transforms_ = [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]

        # Load your own data. Here we try the test set.
        # I was trying with Yosemite photos, they look awesome.
        # You can use `conf.dataset_name`, if you specified `dataset_name` as something you wanted to be calculated
        # in the call to `experiment.configs`
        dataset = ImageDataset(conf.dataset_name, transforms_, 'train')
        # Get an image from dataset
        x_image = dataset[10]['x']
        # Display the image
        plot_image(x_image)

        # Evaluation mode
        conf.generator_xy.eval()
        conf.generator_yx.eval()

        # We don't need gradients
        with torch.no_grad():
            # Add batch dimension and move to the device we use
            data = x_image.unsqueeze(0).to(conf.device)
            generated_y = conf.generator_xy(data)

        # Display the generated image.
        plot_image(generated_y[0].cpu())


if __name__ == '__main__':
    train()
    # evaluate()
