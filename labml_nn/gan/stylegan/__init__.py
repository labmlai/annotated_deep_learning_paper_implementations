"""
---
title: Style GAN 2
summary: >
 An annotated PyTorch implementation of StyleGAN2.
---

# Style GAN 2

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
from typing import Any, Iterator, Tuple, Optional, List

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
import torchvision
from PIL import Image
from torch import nn

from labml import tracker, lab, monit, experiment
from labml.configs import BaseConfigs
from labml_helpers.device import DeviceConfigs
from labml_helpers.train_valid import ModeState, hook_model_outputs
from labml_nn.gan.wasserstein import DiscriminatorLoss, GeneratorLoss


class MappingNetwork(nn.Module):
    """
    ## Mapping Network

    ![Mapping Network](mapping_network.svg)

    This is a MLP with 8 linear layers.
    The mapping network maps the latent vector $z \in \mathcal{W}$
    to an intermediate latent space $w \in \mathcal{W}$.
    $\mathcal{W}$ space will be disentangled from the image space
    where the factors of variation become more linear.
    """

    def __init__(self, features: int, n_layers: int):
        super().__init__()

        # Create the MLP
        layers = []
        for i in range(n_layers):
            # [Equalized linear layers](#equalized_linear)
            layers.append(EqualizedLinear(features, features))
            # Leaky Relu
            layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))

        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor):
        # Normalize $z$
        z = F.normalize(z, dim=1)
        # Map $z$ to $w$
        return self.net(z)


class Generator(nn.Module):
    def __init__(self, log_resolution, d_latent, n_features=32, max_features=512):
        super().__init__()

        features = [min(max_features, n_features * (2 ** i)) for i in range(log_resolution - 2, -1, -1)]
        self.n_blocks = len(features)

        self.style_block = StyleBlock(d_latent, features[0], features[0])
        self.to_rgb = ToRGB(d_latent, features[0])

        blocks = [GeneratorBlock(d_latent, features[i - 1], features[i]) for i in range(1, self.n_blocks)]
        print(features)

        self.blocks = nn.ModuleList(blocks)

        self.up_sample = UpSample()
        self.initial_constant = nn.Parameter(torch.randn((1, features[0], 4, 4)))

    def forward(self, styles: torch.Tensor, input_noise: List[Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]]):
        batch_size = styles.shape[1]

        x = self.initial_constant.expand(batch_size, -1, -1, -1)

        x = self.style_block(x, styles[0], input_noise[0][1])
        rgb = self.to_rgb(x, None, styles[0])

        for i in range(1, self.n_blocks):
            x = self.up_sample(x)
            x, rgb = self.blocks[i - 1](x, rgb, styles[i], input_noise[i])

        return rgb


class GeneratorBlock(nn.Module):
    def __init__(self, d_latent, in_features, out_features):
        super().__init__()
        self.style_block1 = StyleBlock(d_latent, in_features, out_features)
        self.style_block2 = StyleBlock(d_latent, out_features, out_features)

        self.to_rgb = ToRGB(d_latent, out_features)

    def forward(self, x: torch.Tensor, prev_rgb: torch.Tensor, w: torch.Tensor,
                noise: Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]):
        x = self.style_block1(x, w, noise[0])
        x = self.style_block2(x, w, noise[1])

        rgb = self.to_rgb(x, prev_rgb, w)
        return x, rgb


class StyleBlock(nn.Module):
    def __init__(self, d_latent, in_features, out_features):
        super().__init__()
        self.to_style = EqualizedLinear(d_latent, in_features, bias=1.0)
        self.conv = Conv2dWeightModulate(in_features, out_features, kernel_size=3)
        self.scale_noise = nn.Parameter(torch.zeros(1))
        self.bias = nn.Parameter(torch.zeros(out_features))

        self.activation = nn.LeakyReLU(0.2, True)

    def forward(self, x: torch.Tensor, w: torch.Tensor, noise: Optional[torch.Tensor]):
        style = self.to_style(w)
        x = self.conv(x, style)
        if noise is not None:
            x = x + self.scale_noise[None, :, None, None] * noise
        return self.activation(x + self.bias[None, :, None, None])


class ToRGB(nn.Module):
    def __init__(self, d_latent, in_features):
        super().__init__()
        self.input_channel = in_features
        self.to_style = EqualizedLinear(d_latent, in_features, bias=1.0)

        self.conv = Conv2dWeightModulate(in_features, 3, kernel_size=1, demodulate=False)
        self.up_sample = UpSample()
        self.activation = nn.LeakyReLU(0.2, True)

    def forward(self, x, prev_rgb, style):
        style = self.to_style(style)
        rgb = self.activation(self.conv(x, style))

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

        self.weight = EqualizedWeight([out_features, in_features, kernel_size, kernel_size], lr_mul)
        self.eps = eps

    def forward(self, x, style):
        b, c, h, w = x.shape

        style = style[:, None, :, None, None]
        weights = self.weight()[None, :, :, :, :]
        weights = weights * style

        if self.demodulate:
            sigma_inv = torch.rsqrt((weights ** 2).sum(dim=(2, 3, 4), keepdim=True) + self.eps)
            weights = weights * sigma_inv

        x = x.reshape(1, -1, h, w)

        _, _, *ws = weights.shape
        weights = weights.reshape(b * self.filters, *ws)

        x = F.conv2d(x, weights, padding=self.padding, groups=b)

        x = x.reshape(-1, self.filters, h, w)

        return x


class Discriminator(nn.Module):
    """
    ## Style GAN2 Discriminator
    ![Discriminator](style_gan2_disc.svg)
    """

    def __init__(self, log_resolution, n_image_features=3, n_features=64, max_features=512):
        super().__init__()

        self.from_rgb = nn.Sequential(
            EqualizedConv2d(n_image_features, n_features, 1),
            nn.LeakyReLU(0.2, True),
        )

        features = [min(max_features, n_features * (2 ** i)) for i in range(log_resolution - 1)]
        n_blocks = len(features) - 1
        blocks = [DiscriminatorBlock(features[i], features[i + 1]) for i in range(n_blocks)]

        self.blocks = nn.Sequential(*blocks)

        self.std_dev = MiniBatchStdDev()

        total_features = features[-1] + 1
        self.conv = EqualizedConv2d(total_features, total_features, 3)
        self.logits = EqualizedLinear(2 * 2 * total_features, 1)

    def forward(self, x: torch.Tensor):
        x = x - 0.5
        x = self.from_rgb(x)
        x = self.blocks(x)

        x = self.std_dev(x)
        x = self.conv(x)
        x = x.reshape(x.shape[0], -1)
        x = self.logits(x)
        return x


class DiscriminatorBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.residual = nn.Sequential(DownSample(),
                                      EqualizedConv2d(in_features, out_features, kernel_size=1))

        self.block = nn.Sequential(
            EqualizedConv2d(in_features, in_features, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
            EqualizedConv2d(in_features, out_features, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
        )

        self.down_sample = DownSample()
        self.scale = 1 / math.sqrt(2)

    def forward(self, x):
        residual = self.residual(x)
        x = self.block(x)
        x = self.down_sample(x)

        return (x + residual) * self.scale


class MiniBatchStdDev(nn.Module):
    def __init__(self, group_size: int = 4):
        super().__init__()
        self.group_size = group_size

    def forward(self, x: torch.Tensor):
        assert x.shape[0] % self.group_size == 0
        std = torch.sqrt(x.view(self.group_size, -1).var(dim=0) + 1e-8)
        std = std.mean().view(1, 1, 1, 1)
        b, _, h, w = x.shape
        return torch.cat([x, std.expand(b, -1, h, w)], dim=1)


class DownSample(nn.Module):
    def __init__(self):
        super().__init__()
        self.smooth = Smooth()

    def forward(self, x: torch.Tensor):
        x = self.smooth(x)
        x = F.interpolate(x, (x.shape[2] // 2, x.shape[3] // 2), mode='bilinear', align_corners=False)
        return x


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
        self.pad = nn.ReplicationPad2d(1)

    def forward(self, x: torch.Tensor):
        b, c, h, w = x.shape
        x = x.view(-1, 1, h, w)

        x = self.pad(x)
        x = F.conv2d(x, self.kernel)

        return x.view(b, c, h, w)


class EqualizedLinear(nn.Module):
    """
    <a id="equalized_linear"></a>
    ## Equalized Linear Layer
    """
    def __init__(self, in_features: int, out_features: int, lr_mul: float = 1., bias: float = 0.):
        super().__init__()
        self.weight = EqualizedWeight([out_features, in_features], lr_mul)
        self.bias = nn.Parameter(torch.ones(out_features) * bias)

        self.lr_mul = lr_mul

    def forward(self, x: torch.Tensor):
        return F.linear(x, self.weight(), bias=self.bias * self.lr_mul)


class EqualizedConv2d(nn.Module):
    def __init__(self, in_features: int, out_features: int,
                 kernel_size: int, padding: int = 0, stride: int = 1,
                 lr_mul: float = 1.,
                 bias: float = 0.):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.weight = EqualizedWeight([out_features, in_features, kernel_size, kernel_size], lr_mul)
        self.bias = nn.Parameter(torch.ones(out_features) * bias)

        self.lr_mul = lr_mul

    def forward(self, x: torch.Tensor):
        return F.conv2d(x, self.weight(), bias=self.bias * self.lr_mul,
                        padding=self.padding, stride=self.stride)


class EqualizedWeight(nn.Module):
    def __init__(self, shape: List[int], lr_mul=1.):
        super().__init__()

        he_std = 1 / math.sqrt(np.prod(shape[1:]))
        self.weight = nn.Parameter(torch.randn(shape) / lr_mul)
        self.runtime_coef = lr_mul * he_std

    def forward(self):
        return self.weight * self.runtime_coef


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
            self.steps.add_(1.)

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


class GradientPenalty(nn.Module):
    """
    ## Gradient Penalty
    """

    def forward(self, x: torch.Tensor, f: torch.Tensor):
        """
        * `x` is $x \sim \mathbb{P}_r$
        * `f` is $D(x)$

        $\hat{x} \leftarrow x$
        since we set $\epsilon = 1$ for this implementation.
        """

        # Get batch size
        batch_size = x.shape[0]

        # Calculate gradients of $D(x)$ with respect to $x$.
        # `grad_outputs` is set to ones since we want the gradients of $D(x)$,
        # and we need to create and retain graph since we have to compute gradients
        # with respect to weight on this loss.
        gradients, *_ = torch.autograd.grad(outputs=f,
                                            inputs=x,
                                            grad_outputs=f.new_ones(f.shape),
                                            create_graph=True)

        # Reshape gradients to calculate the norm
        gradients = gradients.reshape(batch_size, -1)
        # Calculate the norm $\Vert \nabla_{\hat{x}} D(\hat{x}) \Vert_2$
        norm = gradients.norm(2, dim=-1)
        # Return the loss $\big(\Vert \nabla_{\hat{x}} D(\hat{x}) \Vert_2 - 1\big)^2$
        return torch.mean(norm ** 2)


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
        self.loader = cycle(dataloader)
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
