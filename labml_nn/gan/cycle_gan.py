"""
# Cycle GAN
This is an implementation of paper
[Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593).

### Running the experiment
To train the model you need to download datasets from
`https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/[DATASET NAME].zip`
and extract them into folder `labml_nn/data/cycle_gan/[DATASET NAME]`.
You will also have to `dataset_name` configuration to `[DATASET NAME]`.
This defaults to `monet2photo`.

I've taken pieces of code from [https://github.com/eriklindernoren/PyTorch-GAN](https://github.com/eriklindernoren/PyTorch-GAN).
It is a very good resource if you want to checkout other GAN variations too.
"""

import itertools
import random
from pathlib import PurePath, Path
from typing import Tuple

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.utils import make_grid
from torchvision.utils import save_image

from labml import lab, tracker, experiment, monit, configs
from labml.configs import BaseConfigs
from labml_helpers.device import DeviceConfigs
from labml_helpers.module import Module


class GeneratorResNet(Module):
    """
    The generator is a residual network.
    """

    def __init__(self, input_shape: Tuple[int, int, int], n_residual_blocks: int):
        super().__init__()
        # The number of channels in the input image, which is 3 for RGB images.
        channels = input_shape[0]

        # This first block runs a $7\times7$ convolution and maps the image to
        # a feature map.
        # The output feature map has same height and width because we have
        # a padding of $3$.
        # Reflection padding is used because it gives better image quality at edges.
        #
        # `inplace=True` in `ReLU` saves a little bit of memory.
        out_features = 64
        layers = [
            nn.Conv2d(channels, out_features, kernel_size=7, padding=3, padding_mode='reflect'),
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
        layers += [nn.Conv2d(out_features, channels, 7, padding=3, padding_mode='reflect'), nn.Tanh()]

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

        # Output of the discriminator is also map of probabilities*
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
    It does a convolution, an optional normalization, and a leaky relu.

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
    Loads an image and change to RGB if in grey-scale.
    """
    image = Image.open(path)
    if image.mode != 'RGB':
        image = Image.new("RGB", image.size).paste(image)

    return image


class ImageDataset(Dataset):
    """
    Dataset to load images
    """

    def __init__(self, root: PurePath, transforms_, unaligned: bool, mode: str):
        root = Path(root)
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.files_A = sorted(str(f) for f in (root / f'{mode}A').iterdir())
        self.files_B = sorted(str(f) for f in (root / f'{mode}B').iterdir())

    def __getitem__(self, index):
        return {"a": self.transform(load_image(self.files_A[index % len(self.files_A)])),
                "b": self.transform(load_image(self.files_B[index % len(self.files_B)]))}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))


class ReplayBuffer:
    """
    Replay buffer is used to train the discriminator.
    Generated images are added to the replay buffer and sampled from it.

    The replay buffer returns the newly added image with a probability of $0.5$.
    Otherwise it sends an older generated image and and replaces the older image
    with the new generated image.

    This is done to reduce model oscillation.
    """

    def __init__(self, max_size: int = 50):
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
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
    """This is the configurations for the experiment"""
    device: torch.device = DeviceConfigs()
    epochs: int = 200
    dataset_name: str = 'monet2photo'
    batch_size: int = 1

    data_loader_workers = 0
    is_save_models = True

    learning_rate = 0.0002
    adam_betas = (0.5, 0.999)
    decay_start = 100

    # The paper suggests using a least-squares loss instead of
    # negative log-likelihood, at it is found to be more stable.
    gan_loss = torch.nn.MSELoss()

    # L1 loss is used for cycle loss and identity loss
    cycle_loss = torch.nn.L1Loss()
    identity_loss = torch.nn.L1Loss()

    batch_step = 'cycle_gan_batch_step'

    img_height = 256
    img_width = 256
    img_channels = 3

    n_residual_blocks = 9

    cyclic_loss_coefficient = 10.0
    identity_loss_coefficient = 5.

    sample_interval = 500

    generator_ab: GeneratorResNet
    generator_ba: GeneratorResNet
    discriminator_a: Discriminator
    discriminator_b: Discriminator

    generator_optimizer: torch.optim.Adam
    discriminator_optimizer: torch.optim.Adam

    generator_lr_scheduler: torch.optim.lr_scheduler.LambdaLR
    discriminator_lr_scheduler: torch.optim.lr_scheduler.LambdaLR

    dataloader: DataLoader
    valid_dataloader: DataLoader

    def sample_images(self, n: int):
        """Generate samples from test set and save them"""
        batch = next(iter(self.valid_dataloader))
        self.generator_ab.eval()
        self.generator_ba.eval()
        with torch.no_grad():
            real_a, real_b = batch['a'].to(self.generator_ab.device), batch['b'].to(self.generator_ba.device)
            fake_b = self.generator_ab(real_a)
            fake_a = self.generator_ba(real_b)

            # Arange images along x-axis
            real_a = make_grid(real_a, nrow=5, normalize=True)
            real_b = make_grid(real_b, nrow=5, normalize=True)
            fake_a = make_grid(fake_a, nrow=5, normalize=True)
            fake_b = make_grid(fake_b, nrow=5, normalize=True)

            # arange images along y-axis
            image_grid = torch.cat((real_a, fake_b, real_b, fake_a), 1)
        save_image(image_grid, f"images/{self.dataset_name}/{n}.png", normalize=False)

    def optimize_generators(self, real_a: torch.Tensor, real_b: torch.Tensor, true_labels: torch.Tensor):
        #  Change to training mode
        self.generator_ab.train()
        self.generator_ba.train()

        # Identity loss
        loss_identity = (self.identity_loss(self.generator_ba(real_a), real_a) +
                         self.identity_loss(self.generator_ab(real_b), real_b))

        # Generate images
        fake_b = self.generator_ab(real_a)
        fake_a = self.generator_ba(real_b)

        # GAN loss
        loss_gan = (self.gan_loss(self.discriminator_b(fake_b), true_labels) +
                    self.gan_loss(self.discriminator_a(fake_a), true_labels))

        # Cycle loss
        loss_cycle = (self.cycle_loss(self.generator_ba(fake_b), real_a) +
                      self.cycle_loss(self.generator_ab(fake_a), real_b))

        # Total loss
        loss_generator = (loss_gan +
                          self.cyclic_loss_coefficient * loss_cycle +
                          self.identity_loss_coefficient * loss_identity)

        self.generator_optimizer.zero_grad()
        loss_generator.backward()
        self.generator_optimizer.step()

        tracker.add({'loss.generator': loss_generator,
                     'loss.generator.cycle': loss_cycle,
                     'loss.generator.gan': loss_gan,
                     'loss.generator.identity': loss_identity})

        return fake_a, fake_b

    def optimize_discriminator(self, real_a: torch.Tensor, real_b: torch.Tensor,
                               fake_a: torch.Tensor, fake_b: torch.Tensor,
                               true_labels: torch.Tensor, false_labels: torch.Tensor):
        loss_discriminator = (self.gan_loss(self.discriminator_a(real_a), true_labels) +
                              self.gan_loss(self.discriminator_a(fake_a), false_labels) +
                              self.gan_loss(self.discriminator_b(real_b), true_labels) +
                              self.gan_loss(self.discriminator_b(fake_b), false_labels))

        self.discriminator_optimizer.zero_grad()
        loss_discriminator.backward()
        self.discriminator_optimizer.step()

        tracker.add({'loss.discriminator': loss_discriminator})

    def run(self):
        # Replay buffers to keep generated samples
        fake_a_buffer = ReplayBuffer()
        fake_b_buffer = ReplayBuffer()

        for epoch in monit.loop(self.epochs):
            for i, batch in enumerate(self.dataloader):
                # Move images to the device
                real_a, real_b = batch['a'].to(self.device), batch['b'].to(self.device)

                # true labels equal to $1$
                true_labels = torch.ones(real_a.size(0), *self.discriminator_a.output_shape,
                                         device=self.device, requires_grad=False)
                # false labels equal to $0$
                false_labels = torch.zeros(real_a.size(0), *self.discriminator_a.output_shape,
                                           device=self.device, requires_grad=False)

                # Train the generators
                fake_a, fake_b = self.optimize_generators(real_a, real_b, true_labels)

                #  Train discriminators
                self.optimize_discriminator(real_a, real_b,
                                            fake_a_buffer.push_and_pop(fake_a), fake_b_buffer.push_and_pop(fake_b),
                                            true_labels, false_labels)

                # Save training statistics
                tracker.save()

                # If at sample interval save image
                batches_done = epoch * len(self.dataloader) + i
                if batches_done % self.sample_interval == 0:
                    self.sample_images(batches_done)

                tracker.add_global_step(max(len(real_a), len(real_b)))

            # Update learning rates
            self.generator_lr_scheduler.step()
            self.discriminator_lr_scheduler.step()


@configs.setup([Configs.generator_ab, Configs.generator_ba, Configs.discriminator_a, Configs.discriminator_b,
                Configs.generator_optimizer, Configs.discriminator_optimizer,
                Configs.generator_lr_scheduler, Configs.discriminator_lr_scheduler])
def setup_models(self: Configs):
    input_shape = (self.img_channels, self.img_height, self.img_width)
    self.generator_ab = GeneratorResNet(input_shape, self.n_residual_blocks).to(self.device)
    self.generator_ba = GeneratorResNet(input_shape, self.n_residual_blocks).to(self.device)
    self.discriminator_a = Discriminator(input_shape).to(self.device)
    self.discriminator_b = Discriminator(input_shape).to(self.device)

    self.generator_optimizer = torch.optim.Adam(
        itertools.chain(self.generator_ab.parameters(), self.generator_ba.parameters()),
        lr=self.learning_rate, betas=self.adam_betas)
    self.discriminator_optimizer = torch.optim.Adam(
        itertools.chain(self.discriminator_a.parameters(), self.discriminator_b.parameters()),
        lr=self.learning_rate, betas=self.adam_betas)

    decay_epochs = self.epochs - self.decay_start
    self.generator_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        self.generator_optimizer, lr_lambda=lambda e: 1.0 - max(0, e - self.decay_start) / decay_epochs)
    self.discriminator_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        self.discriminator_optimizer, lr_lambda=lambda e: 1.0 - max(0, e - self.decay_start) / decay_epochs)


@configs.setup([Configs.dataloader, Configs.valid_dataloader])
def setup_dataloader(self: Configs):
    images_path = lab.get_data_path() / 'cycle_gan' / self.dataset_name
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
        ImageDataset(images_path, transforms_, True, 'train'),
        batch_size=self.batch_size,
        shuffle=True,
        num_workers=self.data_loader_workers,
    )
    # Test data loader
    self.valid_dataloader = DataLoader(
        ImageDataset(images_path, transforms_, True, "test"),
        batch_size=5,
        shuffle=True,
        num_workers=self.data_loader_workers,
    )


def main():
    conf = Configs()
    experiment.create(name='cycle_gan')
    experiment.configs(conf, 'run')
    with experiment.start():
        conf.run()


if __name__ == '__main__':
    main()
