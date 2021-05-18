"""
# Style GAN2

This is a [PyTorch](https://pytorch.org) implementation of the paper [Analyzing and Improving the Image Quality of StyleGAN](https://arxiv.org/abs/1912.04958) which introduces **Style GAN2**. Style GAN2 is an improvement over **Style GAN** from the paper [A Style-Based Generator Architecture for Generative Adversarial Networks](https://arxiv.org/abs/1812.04948). And Style GAN is based on **Progressive GAN** from the paper
[Progressive Growing of GANs for Improved Quality, Stability, and Variation](https://arxiv.org/abs/1710.10196). All three papers are from same set of authors from [NVIDIA AI](https://twitter.com/NVIDIAAI).

*Our implementation is a minimalistic Style GAN2 model training code. Only single GPU training is supported to keep the implementation simple. We managed to shrink it to keep it at a little more than 350 lines of code.*

We'll first introduce the three papers at a high level.

## Generative Adversarial Networks

Generative adversarial networks have two components; the generator and the discriminator. The generator network takes a random latent vector ($z \in \mathcal{Z}$) and tries to generate a realistic image. The discriminator network tries to differentiate the real images from generated images. When we train the two networks together the generator starts generating images indistinguishable from real images.

## Progressive GAN

Progressive GAN generate high-resolution images ($1080 \times 1080$) of size. It does so by *progressively* increasing the image size. First, it trains a network that produces a $4 \times 4$ image, then $8 \times 8$ , then an $16 \times 16$  image, and so on upto the desired image resolution.

At each resolution the generator network produces an image in latent space which is converted into RGB,
with a $1 \times 1$  convolution. When we progress from a lower resolution to a higher resolution (say from $4 \times 4$  to $8 \times 8$ ) we scale the latent image by $2\times$ and add a new block (two $3 \times 3$  convolution layers) and a new $1 \times 1$  layer to get RGB. The transition is done smoothly by adding a residual connection to the $2\times$ scaled $4 \times 4$  RGB image. The weight of this residual connection is slowly reduced, to let the new block to take over.

The discriminator is a mirror image of the generator network. The progressive growing of the disciminator is done similarly.

![progressive_gan.svg](progressive_gan.svg)

They use **minibatch standard deviation** to increase variation and **equalized learning rate** which we discussed below in the implementation. They also use **pixel-wise normalization** where at each pixel the feature vector is normalized. They apply this to all the convlution layer outputs (except RGB).


## Style GAN

Style GAN improves the generator of Progressive GAN keeping the discriminator architecture same.

#### Mapping Network

It maps the random latent vector ($z \in \mathcal{Z}$)i nto a different latent space ($w \in \mathcal{W}$), with a 8-layer neural network. This gives a intemediate latent space $\mathcal{W}$ where the factors of variations are more linear (disentangled).

#### AdaIN

Then $w$ is transformed into two vectors (***styles***) per layer, $i$, $y_i = (y_{s,i}, y_{b,i}) = f_{A_i}(w)$ and used for scaling and shifting (biasing) in each layer with $\text{AdaIN}$ operator (normalize and scale):
$$
\text{AdaIN}(x_i, y_i) = y_{s, i} \frac{x_i - \mu(x_i)}{\sigma(x_i)} + y_{b,i}
$$

#### Style Mixing

To prevent the generator from assuming adjacent styles are correlated, they randomly use different styles for different blocks. That is, they sample two latent vectors $(z_1, z_2)$ and corresponding $(w_1, w_2)$ and use $w_1$ based styles for some blocks and $w_2$ based styles for some blacks randomly.

#### Stocastic Variation

Noise is made available to each block which helps generator create more realistic images. Noise is scaled per channel by a learned weight.

#### Bilinear Up and Down Sampling

All the up and down sampling operations are accompanied by bilinear smoothing.

![style_gan.svg](style_gan.svg)

## Style GAN 2

Style GAN 2 changes both the generator and the discriminator of Style GAN.

#### Weight Modulation and Demodulation

They remove the $\text{AdaIN}$ operator and replace it weight modulation and demodulation step. This is supposed to improve what they call droplet artifacts that are present in generated images, which are caused by the normalization in $\text{AdaIN}$ operator. Style vector per layer is calculated from $w_i \in \mathcal{W}$ as $s_i = f_{A_i}(w_i)$.

Then the convolution weights $w$ are modulated as follows. ($w$ here on refers to weights not intermediate latent space, we are sticking to the same notation as the paper.)
$$
w'_{i, j, k} = s_i \cdot w_{i, j, k} \\
$$
Then it's demodulated by normalizing,
$$
w''_{i,j,k} = \frac{w'_{i,j,k}}{\sqrt{\sum_{i,k}{w'_{i, j, k}}^2 + \epsilon}}
$$
where $i$ is the input channel, $j$ is the output channel, and $k$ is the kernel index.

#### Path Length Regularization

Path length regularization encourages a fixed-size step in $\mathcal{W}$ to result in a non-zero, fixed-magnitude change in generated image.

#### No Progressive Growing

StyleGAN2 uses residual connections (with downsampling) in the discriminator and skip connections in the generator with upsampling (the RGB outputs from each layer are added - no residual connections in feature maps). They show that with experiemnts that the contribution of low resolution layers is higher at beginning of the training and then high-resolution layers take over.
"""

import math
from pathlib import Path
from typing import Any, Iterator, Tuple

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
            layers.append(EqualizedLinear(features, features, lr_mul=1.))
            layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        x = F.normalize(x, dim=1)
        return self.net(x)


class Discriminator(nn.Module):
    """
    ![style_gan2_disc.svg](style_gan2_disc.svg)
    """
    def __init__(self, n_layers, n_image_features=3, n_features=32, max_features=512):
        super().__init__()

        self.from_rgb = nn.Sequential(
            EqualizedConv2d(n_image_features, n_features, 1),
            nn.LeakyReLU(0.2, True),
        )

        blocks = []
        out_features = n_features
        for i in range(n_layers):
            in_features = out_features
            out_features = min(out_features * 2, max_features)
            blocks.append(DiscriminatorBlock(in_features, out_features))

        self.blocks = nn.Sequential(*blocks)

        self.conv = EqualizedConv2d(out_features, out_features, 4)
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
        self.residual = EqualizedConv2d(in_features, out_features, kernel_size=1, stride=2)

        self.block = nn.Sequential(
            EqualizedConv2d(in_features, in_features, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
            EqualizedConv2d(in_features, out_features, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
        )

        self.down_sample = DownSample(out_features)
        self.scale = 1 / math.sqrt(2)

    def forward(self, x):
        residual = self.residual(x)
        x = self.block(x)
        x = self.down_sample(x)

        return (x + residual) * self.scale


class Generator(nn.Module):
    """
    ![style_gan2.svg](style_gan2.svg)
    """
    def __init__(self, n_layers, d_latent, n_features=32, max_features=512):
        super().__init__()

        blocks = []
        in_features = n_features
        for i in range(n_layers):
            out_features = in_features
            in_features = min(in_features * 2, max_features)
            blocks.append(GeneratorBlock(d_latent, in_features, out_features))

        self.blocks = nn.ModuleList(list(reversed(blocks)))

        self.up_sample = UpSample()
        self.initial_constant = nn.Parameter(torch.randn((1, in_features, 4, 4)))

    def forward(self, styles: torch.Tensor, input_noise: torch.Tensor):
        batch_size = styles.shape[0]

        x = self.initial_constant.expand(batch_size, -1, -1, -1)

        rgb = None

        for i in range(len(self.blocks)):
            if i != 0:
                x = self.up_sample(x)

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
        self.up_sample = UpSample()

    def forward(self, x, prev_rgb, istyle):
        style = self.to_style(istyle)
        rgb = self.conv(x, style)

        if prev_rgb is not None:
            rgb = rgb + self.up_sample(prev_rgb)

        return rgb


class Conv2dWeightModulate(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, demodulate=True, lr_mul=1., eps=1e-8):
        super().__init__()
        self.filters = out_features
        self.demodulate = demodulate
        self.kernel = kernel_size
        self.padding = (self.kernel - 1) // 2

        he_std = 1 / math.sqrt(in_features * kernel_size * kernel_size)
        self.weight = nn.Parameter(torch.randn((out_features, in_features, kernel_size, kernel_size)) / lr_mul)
        self.eps = eps

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

        x = F.conv2d(x, weights, padding=self.padding, groups=b)

        x = x.reshape(-1, self.filters, h, w)

        return x


class DownSample(nn.Module):
    def __init__(self, features: int):
        super().__init__()
        self.smooth = Smooth()
        self.conv = EqualizedConv2d(features, features, kernel_size=3, padding=1, stride=2)

    def forward(self, x: torch.Tensor):
        return self.conv(self.smooth(x))


class UpSample(nn.Module):
    def __init__(self):
        super().__init__()
        self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.smooth = Smooth()

    def forward(self, x: torch.Tensor):
        return self.smooth(self.up_sample(x))


class Smooth(nn.Module):
    def __init__(self):
        super().__init__()
        kernel = [[1, 2, 1],
                  [2, 4, 2],
                  [1, 2, 1]]
        kernel = torch.tensor([[kernel]], dtype=torch.float)
        kernel /= kernel.sum()
        self.kernel = nn.Parameter(kernel, requires_grad=False)

    def forward(self, x: torch.Tensor):
        b, c, h, w = x.shape
        x = x.view(-1, 1, h, w)

        x = F.conv2d(x, self.kernel, padding=1)

        return x.view(b, c, h, w)


class EqualizedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, lr_mul: float = 1.):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) / lr_mul)
        self.bias = nn.Parameter(torch.zeros(out_features))

        he_std = 1 / math.sqrt(in_features)
        self.runtime_coef = lr_mul * he_std
        self.lr_mul = lr_mul

    def __call__(self, x: torch.Tensor):
        return F.linear(x, self.weight * self.runtime_coef, bias=self.bias * self.lr_mul)


class EqualizedConv2d(nn.Module):
    def __init__(self, in_features: int, out_features: int,
                 kernel_size: int, padding: int = 0, stride: int = 1,
                 lr_mul: float = 1.):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.weight = nn.Parameter(torch.randn((out_features, in_features, kernel_size, kernel_size)) / lr_mul)
        self.bias = nn.Parameter(torch.zeros(out_features))

        he_std = 1 / math.sqrt(in_features * kernel_size * kernel_size)
        self.runtime_coef = lr_mul * he_std
        self.lr_mul = lr_mul

    def forward(self, x: torch.Tensor):
        return F.conv2d(x, self.weight * self.runtime_coef, bias=self.bias * self.lr_mul,
                        padding=self.padding, stride=self.stride)


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
    image_size: int = 32
    n_layers: int
    mapping_network_layers: int = 8
    learning_rate: float = 1e-3
    mapping_network_learning_rate: float = 1e-5
    gradient_accumulate_steps: int = 1
    betas: Tuple[float, float] = (0.0, 0.99)

    path_length_penalty: PathLengthPenalty

    dataset: Dataset
    loader: Iterator

    lazy_gradient_penalty_interval: int = 4
    lazy_path_penalty_interval: int = 32
    lazy_path_penalty_after: int = 5_000

    log_generated_interval: int = 500
    save_checkpoint_interval: int = 2_000

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
        # Train Discriminator
        self.discriminator_optimizer.zero_grad()

        for i in range(self.gradient_accumulate_steps):
            generated_images, _ = self.generate_images(self.batch_size)
            fake_output = self.discriminator(generated_images.detach())

            x = next(self.loader).to(self.device)
            x.requires_grad_()
            real_output = self.discriminator(x)

            real_loss, fake_loss = self.discriminator_loss(real_output, fake_output)
            disc_loss = real_loss + fake_loss

            if (idx + 1) % self.lazy_gradient_penalty_interval == 0:
                gp = self.gradient_penalty(x, real_output)
                tracker.add('loss.gp', gp)
                disc_loss = disc_loss + gp

            disc_loss.backward()

            tracker.add('loss.discriminator', disc_loss)

        self.discriminator_optimizer.step()

        # Train Generator & Mapping Network
        self.generator_optimizer.zero_grad()
        self.mapping_network_optimizer.zero_grad()

        for i in range(self.gradient_accumulate_steps):
            generated_images, w_style = self.generate_images(self.batch_size)
            fake_output = self.discriminator(generated_images)

            gen_loss = self.generator_loss(fake_output)

            if idx > self.lazy_path_penalty_after and (idx + 1) % self.lazy_path_penalty_interval == 0:
                ppl = self.path_length_penalty(w_style, generated_images)
                if not torch.isnan(ppl):
                    gen_loss = gen_loss + ppl

            gen_loss.backward()

            tracker.add('loss.generator', gen_loss)

        self.generator_optimizer.step()
        self.mapping_network_optimizer.step()

        if (idx + 1) % self.log_generated_interval == 0:
            tracker.add('generated', generated_images)
        if (idx + 1) % self.save_checkpoint_interval == 0:
            experiment.save_checkpoint()

        tracker.save()

    def run(self):
        for i in monit.loop(150_000):
            self.step(i)
            if (i + 1) % self.log_generated_interval == 0:
                tracker.new_line()


def main():
    configs = Configs()
    experiment.create(name='stylegan')
    experiment.configs(configs, {
        'device.cuda_device': 0,
        'image_size': 32,
    })

    configs.init()
    experiment.add_pytorch_models(mapping_network=configs.mapping_network,
                                  generator=configs.generator,
                                  discriminator=configs.discriminator)

    with experiment.start():
        configs.run()


if __name__ == '__main__':
    main()
