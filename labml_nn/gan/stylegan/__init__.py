import math
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.nn.functional as F
import torch.utils.data
import torchvision
from PIL import Image
from torch import nn

from labml import tracker, lab, monit, experiment
from labml.configs import BaseConfigs
from labml_helpers.device import DeviceConfigs
from labml_nn.gan.wasserstein import DiscriminatorLoss, GeneratorLoss
from labml_nn.gan.wasserstein.gradient_penalty import GradientPenalty


class MappingNetwork(nn.Module):
    def __init__(self, features: int, n_layers: int):
        super().__init__()

        layers = []
        for i in range(n_layers):
            layers.append(EqualizedLinear(features, features, lr_mul=0.01))
            layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        x = F.normalize(x, dim=1)
        return self.net(x)


class EqualizedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, lr_mul: float = 1.):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) / lr_mul)
        self.bias = nn.Parameter(torch.zeros(out_features))

        he_std = 1 / math.sqrt(in_features)
        self.runtime_coef = lr_mul * he_std

    def __call__(self, x: torch.Tensor):
        return F.linear(x, self.weight * self.runtime_coef, bias=self.bias * self.runtime_coef)


class EqualizedConv2D(nn.Module):
    def __init__(self, in_features: int, out_features: int, kernel_size: int, padding: int, lr_mul: float = 1.):
        super().__init__()
        self.padding = padding
        self.weight = nn.Parameter(torch.randn((out_features, in_features, kernel_size, kernel_size)) / lr_mul)
        self.bias = nn.Parameter(torch.zeros(out_features))

        he_std = 1 / math.sqrt(in_features * kernel_size * kernel_size)
        self.runtime_coef = lr_mul * he_std

    def forward(self, x: torch.Tensor):
        return F.conv2d(x, self.weight * self.runtime_coef, bias=self.bias * self.runtime_coef, padding=self.padding)


class Discriminator(nn.Module):
    def __init__(self, n_layers, n_image_features=3, n_features=32, max_features=512):
        super().__init__()

        self.from_rgb = nn.Sequential(
            nn.Conv2d(n_image_features, n_features, 1),
            nn.LeakyReLU(0.2, True),
        )

        blocks = []
        out_features = n_features
        for i in range(n_layers):
            in_features = out_features
            out_features = min(out_features * 2, max_features)
            blocks.append(DiscriminatorBlock(in_features, out_features))

        self.blocks = nn.Sequential(*blocks)

        self.conv = nn.Conv2d(out_features, out_features, 4)
        self.logits = EqualizedLinear(out_features, 1)

    def __call__(self, x: torch.Tensor):
        x = self.from_rgb(x)
        x = self.blocks(x)

        x = self.conv(x)
        x = x.reshape(x.shape[0], -1)
        x = self.logits(x)
        return x


class DiscriminatorBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.residual = nn.Conv2d(in_features, out_features, kernel_size=1, stride=2)

        self.block = nn.Sequential(
            EqualizedConv2D(in_features, in_features, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
            EqualizedConv2D(in_features, out_features, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
        )

        self.scale = 1 / math.sqrt(2)

    def forward(self, x):
        residual = self.residual(x)
        x = self.block(x)
        x = F.interpolate(x, (x.shape[2] // 2, x.shape[3] // 2), mode='bilinear', align_corners=False)

        return (x + residual) * self.scale


class Generator(nn.Module):
    def __init__(self, n_layers, d_latent, n_features=32, max_features=512):
        super().__init__()

        blocks = []
        in_features = n_features
        for i in range(n_layers):
            out_features = in_features
            in_features = min(in_features * 2, max_features)
            blocks.append(GeneratorBlock(d_latent, in_features, out_features))

        self.blocks = nn.ModuleList(list(reversed(blocks)))

        self.upsample = nn.Upsample(scale_factor=2,
                                    mode='bilinear',
                                    align_corners=False)
        self.initial_constant = nn.Parameter(torch.randn((1, in_features, 4, 4)))

    def forward(self, styles: torch.Tensor, input_noise: torch.Tensor):
        batch_size = styles.shape[0]

        x = self.initial_constant.expand(batch_size, -1, -1, -1)

        rgb = None

        for i in range(len(self.blocks)):
            if i != 0:
                x = self.upsample(x)

            # or input_noise[:, i]?
            x, rgb = self.blocks[i](x, rgb, styles[:, i], input_noise)

        return rgb


class GeneratorBlock(nn.Module):
    def __init__(self, latent_dim, in_features, out_features):
        super().__init__()
        self.to_style1 = EqualizedLinear(latent_dim, in_features)
        self.to_noise1 = EqualizedLinear(1, out_features)
        self.conv1 = Conv2dWeightModulate(in_features, out_features, kernel_size=3)

        self.to_style2 = EqualizedLinear(latent_dim, out_features)
        self.to_noise2 = EqualizedLinear(1, out_features)
        self.conv2 = Conv2dWeightModulate(out_features, out_features, kernel_size=3)

        self.activation = nn.LeakyReLU(0.2, True)
        self.to_rgb = ToRGB(latent_dim, out_features)

    def forward(self, x: torch.Tensor, prev_rgb: torch.Tensor, istyle: torch.Tensor, inoise: torch.Tensor):
        inoise = inoise[:, :x.shape[2], :x.shape[3], :]
        noise1 = self.to_noise1(inoise).permute((0, 3, 2, 1))
        noise2 = self.to_noise2(inoise).permute((0, 3, 2, 1))

        style1 = self.to_style1(istyle)
        x = self.conv1(x, style1)
        x = self.activation(x + noise1)

        style2 = self.to_style2(istyle)
        x = self.conv2(x, style2)
        x = self.activation(x + noise2)

        rgb = self.to_rgb(x, prev_rgb, istyle)
        return x, rgb


class ToRGB(nn.Module):
    def __init__(self, latent_dim, in_features):
        super().__init__()
        self.input_channel = in_features
        self.to_style = EqualizedLinear(latent_dim, in_features)

        self.conv = Conv2dWeightModulate(in_features, 3, kernel_size=1, demodulate=False)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x, prev_rgb, istyle):
        style = self.to_style(istyle)
        rgb = self.conv(x, style)

        if prev_rgb is not None:
            rgb = rgb + self.upsample(prev_rgb)

        return rgb


class Conv2dWeightModulate(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, demodulate=True, lr_mul=1., eps=1e-8):
        super().__init__()
        self.filters = out_features
        self.demodulate = demodulate
        self.kernel = kernel_size
        self.weight = nn.Parameter(torch.randn((out_features, in_features, kernel_size, kernel_size)) / lr_mul)
        self.eps = eps

        he_std = 1 / math.sqrt(in_features * kernel_size * kernel_size)
        self.runtime_coef = lr_mul * he_std

    def forward(self, x, style):
        b, c, h, w = x.shape

        w1 = style[:, None, :, None, None]
        w2 = self.weight[None, :, :, :, :] * self.runtime_coef
        weights = w2 * (w1 + 1)

        if self.demodulate:
            d = torch.rsqrt((weights ** 2).sum(dim=(2, 3, 4), keepdim=True) + self.eps)
            weights = weights * d

        x = x.reshape(1, -1, h, w)

        _, _, *ws = weights.shape
        weights = weights.reshape(b * self.filters, *ws)

        padding = (self.kernel - 1) // 2
        x = F.conv2d(x, weights, padding=padding, groups=b)

        x = x.reshape(-1, self.filters, h, w)

        return x


class PathLengthPenalty(nn.Module):
    def __init__(self, beta: float):
        super().__init__()
        self.beta = beta
        self.steps = nn.Parameter(torch.tensor(0.), requires_grad=False)
        self.moving_average = nn.Parameter(torch.tensor(0.), requires_grad=False)

    def forward(self, styles: torch.Tensor, images: torch.Tensor):
        device = images.device
        num_pixels = images.shape[2] * images.shape[3]
        pl_noise = torch.randn(images.shape, device=device) / math.sqrt(num_pixels)
        outputs = (images * pl_noise).sum()

        gradients, *_ = torch.autograd.grad(outputs=outputs,
                                            inputs=styles,
                                            grad_outputs=torch.ones(outputs.shape, device=device),
                                            create_graph=True)

        path_lengths = (gradients ** 2).sum(dim=2).mean(dim=1).sqrt()

        if self.steps > 0:
            a = self.moving_average / (1 - self.beta ** self.steps)
            loss = torch.mean((path_lengths - a) ** 2)
        else:
            loss = path_lengths.new_tensor(0)

        mean = path_lengths.mean().detach()

        if not torch.isnan(mean):
            self.moving_average.mul_(self.beta).add_(mean, alpha=1 - self.beta)
            self.steps.add(1.)

        return loss


class Dataset(torch.utils.data.Dataset):
    def __init__(self, image_size):
        super().__init__()

        self.folder = lab.get_data_path() / 'celebA' / '512'
        self.image_size = image_size
        self.paths = [p for p in Path(f'{self.folder}').glob(f'**/*.jpg')]

        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(image_size),
            torchvision.transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)


def cycle(iterable):
    while True:
        for i in iterable:
            yield i


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

    batch_size: int = 24
    d_latent: int = 512
    image_size: int = 128
    n_layers: int
    mapping_network_layers: int = 8
    learning_rate: float = 2e-4
    mapping_network_learning_rate: float = 2e-4
    gradient_accumulate_steps: int = 1

    path_length_penalty: PathLengthPenalty

    dataset: Dataset
    loader: Iterator

    def init(self):
        self.dataset = Dataset(self.image_size)
        dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size,
                                                 num_workers=32,
                                                 shuffle=True, drop_last=True, pin_memory=True)
        self.loader = cycle(dataloader)
        self.n_layers = int(math.log2(self.image_size)) - 1

        self.discriminator = Discriminator(self.n_layers - 1).to(self.device)
        self.generator = Generator(self.n_layers, self.d_latent).to(self.device)
        self.mapping_network = MappingNetwork(self.d_latent, self.mapping_network_layers).to(self.device)
        self.path_length_penalty = PathLengthPenalty(0.99).to(self.device)

        self.discriminator_loss = DiscriminatorLoss()
        self.generator_loss = GeneratorLoss()

        self.discriminator_optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.learning_rate, betas=(0.5, 0.9)
        )
        self.generator_optimizer = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.learning_rate, betas=(0.5, 0.9)
        )
        self.mapping_network_optimizer = torch.optim.Adam(
            self.mapping_network.parameters(),
            lr=self.mapping_network_learning_rate, betas=(0.5, 0.9)
        )
        tracker.set_image("generated", True)

    def z_noise(self, batch_size):
        return torch.randn(batch_size, self.d_latent).to(self.device)

    def z_to_w(self, z: torch.Tensor):
        w = self.mapping_network(z)
        return w[:, None, :].expand(-1, self.n_layers, -1)

    def image_noise(self, batch_size):
        return torch.rand(batch_size, self.image_size, self.image_size, 1, device=self.device)

    def generate_images(self, batch_size):
        z_style = self.z_noise(batch_size)
        w_style = self.z_to_w(z_style)
        noise = self.image_noise(batch_size)

        generated_images = self.generator(w_style, noise)

        return generated_images, w_style

    def step(self, idx):
        apply_gradient_penalty = idx % 4 == 0
        apply_path_penalty = idx > 5000 and idx % 32 == 0

        # setup losses

        # train discriminator
        self.discriminator_optimizer.zero_grad()

        for i in range(self.gradient_accumulate_steps):
            generated_images, _ = self.generate_images(self.batch_size)
            fake_output = self.discriminator(generated_images.detach())

            x = next(self.loader).to(self.device)
            x.requires_grad_()
            real_output = self.discriminator(x)

            real_loss, fake_loss = self.discriminator_loss(real_output, fake_output)
            disc_loss = real_loss + fake_loss

            if apply_gradient_penalty:
                gp = self.gradient_penalty(x, real_output)
                tracker.add('loss.gp', gp)
                disc_loss = disc_loss + gp

            disc_loss.backward()

            tracker.add('loss.discriminator', disc_loss)

        self.discriminator_optimizer.step()

        # train generator
        self.generator_optimizer.zero_grad()
        self.mapping_network_optimizer.zero_grad()

        for i in range(self.gradient_accumulate_steps):
            generated_images, w_style = self.generate_images(self.batch_size)
            fake_output = self.discriminator(generated_images)

            gen_loss = self.generator_loss(fake_output)

            if apply_path_penalty:
                ppl = self.path_length_penalty(w_style, generated_images)
                if not torch.isnan(ppl):
                    gen_loss = gen_loss + ppl

            gen_loss.backward()

            tracker.add('loss.generator', gen_loss)

        self.generator_optimizer.step()
        self.mapping_network_optimizer.step()

        if (idx + 1) % 200 == 0:
            tracker.add('generated', generated_images[:4])
        if (idx + 1) % 2_000 == 0:
            experiment.save_checkpoint()

        tracker.save()

    def run(self):
        for i in monit.loop(150_000):
            self.step(i)
            if (i + 1) % 200 == 0:
                tracker.new_line()


def main():
    configs = Configs()
    experiment.create(name='stylegan')
    experiment.configs(configs)

    configs.init()
    experiment.add_pytorch_models(mapping_network=configs.mapping_network,
                                  generator=configs.generator,
                                  discriminator=configs.discriminator)


    with experiment.start():
        configs.run()


if __name__ == '__main__':
    main()
