"""
---
title: StyleGAN 2
summary: >
 An annotated PyTorch implementation of StyleGAN2.
---

# StyleGAN 2

This is a [PyTorch](https://pytorch.org) implementation of the paper
 [Analyzing and Improving the Image Quality of StyleGAN](https://arxiv.org/abs/1912.04958)
 which introduces **StyleGAN 2**.
StyleGAN 2 is an improvement over **StyleGAN** from the paper
 [A Style-Based Generator Architecture for Generative Adversarial Networks](https://arxiv.org/abs/1812.04948).
And StyleGAN is based on **Progressive GAN** from the paper
 [Progressive Growing of GANs for Improved Quality, Stability, and Variation](https://arxiv.org/abs/1710.10196).
All three papers are from the same authors from [NVIDIA AI](https://twitter.com/NVIDIAAI).

*Our implementation is a minimalistic StyleGAN 2 model training code.
Only single GPU training is supported to keep the implementation simple.
We managed to shrink it to keep it at less than 500 lines of code, including the training loop.*

**🏃 Here's the training code: [`experiment.py`](experiment.html).**

![Generated Images](generated_64.png)

*<small>These are $64 \times 64$ images generated after training for about 80K steps.</small>*


We'll first introduce the three papers at a high level.

## Generative Adversarial Networks

Generative adversarial networks have two components; the generator and the discriminator.
The generator network takes a random latent vector ($z \in \mathcal{Z}$)
 and tries to generate a realistic image.
The discriminator network tries to differentiate the real images from generated images.
When we train the two networks together the generator starts generating images indistinguishable from real images.

## Progressive GAN

Progressive GAN generates high-resolution images ($1080 \times 1080$) of size.
It does so by *progressively* increasing the image size.
First, it trains a network that produces a $4 \times 4$ image, then $8 \times 8$ ,
 then an $16 \times 16$  image, and so on up to the desired image resolution.

At each resolution, the generator network produces an image in latent space which is converted into RGB,
with a $1 \times 1$  convolution.
When we progress from a lower resolution to a higher resolution
 (say from $4 \times 4$  to $8 \times 8$ ) we scale the latent image by $2\times$
 and add a new block (two $3 \times 3$  convolution layers)
 and a new $1 \times 1$  layer to get RGB.
The transition is done smoothly by adding a residual connection to
 the $2\times$ scaled $4 \times 4$  RGB image.
The weight of this residual connection is slowly reduced, to let the new block take over.

The discriminator is a mirror image of the generator network.
The progressive growth of the discriminator is done similarly.

![progressive_gan.svg](progressive_gan.svg)

*<small>$2\times$ and $0.5\times$ denote feature map resolution scaling and scaling.
$4\times4$, $8\times4$, ... denote feature map resolution at the generator or discriminator block.
Each discriminator and generator block consists of 2 convolution layers with leaky ReLU activations.</small>*

They use **minibatch standard deviation** to increase variation and
 **equalized learning rate** which we discussed below in the implementation.
They also use **pixel-wise normalization** where at each pixel the feature vector is normalized.
They apply this to all the convolution layer outputs (except RGB).


## StyleGAN

StyleGAN improves the generator of Progressive GAN keeping the discriminator architecture the same.

#### Mapping Network

It maps the random latent vector ($z \in \mathcal{Z}$)
 into a different latent space ($w \in \mathcal{W}$),
 with an 8-layer neural network.
This gives an intermediate latent space $\mathcal{W}$
where the factors of variations are more linear (disentangled).

#### AdaIN

Then $w$ is transformed into two vectors (***styles***) per layer,
 $i$, $y_i = (y_{s,i}, y_{b,i}) = f_{A_i}(w)$ and used for scaling and shifting (biasing)
 in each layer with $\text{AdaIN}$ operator (normalize and scale):
$$\text{AdaIN}(x_i, y_i) = y_{s, i} \frac{x_i - \mu(x_i)}{\sigma(x_i)} + y_{b,i}$$

#### Style Mixing

To prevent the generator from assuming adjacent styles are correlated,
 they randomly use different styles for different blocks.
That is, they sample two latent vectors $(z_1, z_2)$ and corresponding $(w_1, w_2)$ and
 use $w_1$ based styles for some blocks and $w_2$ based styles for some blacks randomly.

#### Stochastic Variation

Noise is made available to each block which helps the generator create more realistic images.
Noise is scaled per channel by a learned weight.

#### Bilinear Up and Down Sampling

All the up and down-sampling operations are accompanied by bilinear smoothing.

![style_gan.svg](style_gan.svg)

*<small>$A$ denotes a linear layer.
$B$ denotes a broadcast and scaling operation (noise is a single channel).
StyleGAN also uses progressive growing like Progressive GAN</small>*

## StyleGAN 2

StyleGAN 2 changes both the generator and the discriminator of StyleGAN.

#### Weight Modulation and Demodulation

They remove the $\text{AdaIN}$ operator and replace it with
 the weight modulation and demodulation step.
This is supposed to improve what they call droplet artifacts that are present in generated images,
 which are caused by the normalization in $\text{AdaIN}$ operator.
Style vector per layer is calculated from $w_i \in \mathcal{W}$ as $s_i = f_{A_i}(w_i)$.

Then the convolution weights $w$ are modulated as follows.
($w$ here on refers to weights not intermediate latent space,
 we are sticking to the same notation as the paper.)

$$w'_{i, j, k} = s_i \cdot w_{i, j, k}$$
Then it's demodulated by normalizing,
$$w''_{i,j,k} = \frac{w'_{i,j,k}}{\sqrt{\sum_{i,k}{w'_{i, j, k}}^2 + \epsilon}}$$
where $i$ is the input channel, $j$ is the output channel, and $k$ is the kernel index.

#### Path Length Regularization

Path length regularization encourages a fixed-size step in $\mathcal{W}$ to result in a non-zero,
 fixed-magnitude change in the generated image.

#### No Progressive Growing

StyleGAN2 uses residual connections (with down-sampling) in the discriminator and skip connections
 in the generator with up-sampling
  (the RGB outputs from each layer are added - no residual connections in feature maps).
They show that with experiments that the contribution of low-resolution layers is higher
 at beginning of the training and then high-resolution layers take over.
"""

import math
from typing import Tuple, Optional, List

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
from torch import nn


class MappingNetwork(nn.Module):
    """
    <a id="mapping_network"></a>
    ## Mapping Network

    ![Mapping Network](mapping_network.svg)

    This is an MLP with 8 linear layers.
    The mapping network maps the latent vector $z \in \mathcal{W}$
    to an intermediate latent space $w \in \mathcal{W}$.
    $\mathcal{W}$ space will be disentangled from the image space
    where the factors of variation become more linear.
    """

    def __init__(self, features: int, n_layers: int):
        """
        * `features` is the number of features in $z$ and $w$
        * `n_layers` is the number of layers in the mapping network.
        """
        super().__init__()

        # Create the MLP
        layers = []
        for i in range(n_layers):
            # [Equalized learning-rate linear layers](#equalized_linear)
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
    """
    <a id="generator"></a>
    ## StyleGAN2 Generator

    ![Generator](style_gan2.svg)

    *<small>$A$ denotes a linear layer.
    $B$ denotes a broadcast and scaling operation (noise is a single channel).
    [*toRGB*](#to_rgb) also has a style modulation which is not shown in the diagram to keep it simple.</small>*

    The generator starts with a learned constant.
    Then it has a series of blocks. The feature map resolution is doubled at each block
    Each block outputs an RGB image and they are scaled up and summed to get the final RGB image.
    """

    def __init__(self, log_resolution: int, d_latent: int, n_features: int = 32, max_features: int = 512):
        """
        * `log_resolution` is the $\log_2$ of image resolution
        * `d_latent` is the dimensionality of $w$
        * `n_features` number of features in the convolution layer at the highest resolution (final block)
        * `max_features` maximum number of features in any generator block
        """
        super().__init__()

        # Calculate the number of features for each block
        #
        # Something like `[512, 512, 256, 128, 64, 32]`
        features = [min(max_features, n_features * (2 ** i)) for i in range(log_resolution - 2, -1, -1)]
        # Number of generator blocks
        self.n_blocks = len(features)

        # Trainable $4 \times 4$ constant
        self.initial_constant = nn.Parameter(torch.randn((1, features[0], 4, 4)))

        # First style block for $4 \times 4$ resolution and layer to get RGB
        self.style_block = StyleBlock(d_latent, features[0], features[0])
        self.to_rgb = ToRGB(d_latent, features[0])

        # Generator blocks
        blocks = [GeneratorBlock(d_latent, features[i - 1], features[i]) for i in range(1, self.n_blocks)]
        self.blocks = nn.ModuleList(blocks)

        # $2 \times$ up sampling layer. The feature space is up sampled
        # at each block
        self.up_sample = UpSample()

    def forward(self, w: torch.Tensor, input_noise: List[Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]]):
        """
        * `w` is $w$. In order to mix-styles (use different $w$ for different layers), we provide a separate
        $w$ for each [generator block](#generator_block). It has shape `[n_blocks, batch_size, d_latent]1.
        * `input_noise` is the noise for each block.
        It's a list of pairs of noise sensors because each block (except the initial) has two noise inputs
        after each convolution layer (see the diagram).
        """

        # Get batch size
        batch_size = w.shape[1]

        # Expand the learned constant to match batch size
        x = self.initial_constant.expand(batch_size, -1, -1, -1)

        # The first style block
        x = self.style_block(x, w[0], input_noise[0][1])
        # Get first rgb image
        rgb = self.to_rgb(x, w[0])

        # Evaluate rest of the blocks
        for i in range(1, self.n_blocks):
            # Up sample the feature map
            x = self.up_sample(x)
            # Run it through the [generator block](#generator_block)
            x, rgb_new = self.blocks[i - 1](x, w[i], input_noise[i])
            # Up sample the RGB image and add to the rgb from the block
            rgb = self.up_sample(rgb) + rgb_new

        # Return the final RGB image
        return rgb


class GeneratorBlock(nn.Module):
    """
    <a id="generator_block"></a>
    ### Generator Block

    ![Generator block](generator_block.svg)

    *<small>$A$ denotes a linear layer.
    $B$ denotes a broadcast and scaling operation (noise is a single channel).
    [*toRGB*](#to_rgb) also has a style modulation which is not shown in the diagram to keep it simple.</small>*

    The generator block consists of two [style blocks](#style_block) ($3 \times 3$ convolutions with style modulation)
    and an RGB output.
    """

    def __init__(self, d_latent: int, in_features: int, out_features: int):
        """
        * `d_latent` is the dimensionality of $w$
        * `in_features` is the number of features in the input feature map
        * `out_features` is the number of features in the output feature map
        """
        super().__init__()

        # First [style block](#style_block) changes the feature map size to `out_features`
        self.style_block1 = StyleBlock(d_latent, in_features, out_features)
        # Second [style block](#style_block)
        self.style_block2 = StyleBlock(d_latent, out_features, out_features)

        # *toRGB* layer
        self.to_rgb = ToRGB(d_latent, out_features)

    def forward(self, x: torch.Tensor, w: torch.Tensor, noise: Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]):
        """
        * `x` is the input feature map of shape `[batch_size, in_features, height, width]`
        * `w` is $w$ with shape `[batch_size, d_latent]`
        * `noise` is a tuple of two noise tensors of shape `[batch_size, 1, height, width]`
        """
        # First style block with first noise tensor.
        # The output is of shape `[batch_size, out_features, height, width]`
        x = self.style_block1(x, w, noise[0])
        # Second style block with second noise tensor.
        # The output is of shape `[batch_size, out_features, height, width]`
        x = self.style_block2(x, w, noise[1])

        # Get RGB image
        rgb = self.to_rgb(x, w)

        # Return feature map and rgb image
        return x, rgb


class StyleBlock(nn.Module):
    """
    <a id="style_block"></a>
    ### Style Block

    ![Style block](style_block.svg)

    *<small>$A$ denotes a linear layer.
    $B$ denotes a broadcast and scaling operation (noise is single channel).</small>*

    Style block has a weight modulation convolution layer.
    """

    def __init__(self, d_latent: int, in_features: int, out_features: int):
        """
        * `d_latent` is the dimensionality of $w$
        * `in_features` is the number of features in the input feature map
        * `out_features` is the number of features in the output feature map
        """
        super().__init__()
        # Get style vector from $w$ (denoted by $A$ in the diagram) with
        # an [equalized learning-rate linear layer](#equalized_linear)
        self.to_style = EqualizedLinear(d_latent, in_features, bias=1.0)
        # Weight modulated convolution layer
        self.conv = Conv2dWeightModulate(in_features, out_features, kernel_size=3)
        # Noise scale
        self.scale_noise = nn.Parameter(torch.zeros(1))
        # Bias
        self.bias = nn.Parameter(torch.zeros(out_features))

        # Activation function
        self.activation = nn.LeakyReLU(0.2, True)

    def forward(self, x: torch.Tensor, w: torch.Tensor, noise: Optional[torch.Tensor]):
        """
        * `x` is the input feature map of shape `[batch_size, in_features, height, width]`
        * `w` is $w$ with shape `[batch_size, d_latent]`
        * `noise` is a tensor of shape `[batch_size, 1, height, width]`
        """
        # Get style vector $s$
        s = self.to_style(w)
        # Weight modulated convolution
        x = self.conv(x, s)
        # Scale and add noise
        if noise is not None:
            x = x + self.scale_noise[None, :, None, None] * noise
        # Add bias and evaluate activation function
        return self.activation(x + self.bias[None, :, None, None])


class ToRGB(nn.Module):
    """
    <a id="to_rgb"></a>
    ### To RGB

    ![To RGB](to_rgb.svg)

    *<small>$A$ denotes a linear layer.</small>*

    Generates an RGB image from a feature map using $1 \times 1$ convolution.
    """

    def __init__(self, d_latent: int, features: int):
        """
        * `d_latent` is the dimensionality of $w$
        * `features` is the number of features in the feature map
        """
        super().__init__()
        # Get style vector from $w$ (denoted by $A$ in the diagram) with
        # an [equalized learning-rate linear layer](#equalized_linear)
        self.to_style = EqualizedLinear(d_latent, features, bias=1.0)

        # Weight modulated convolution layer without demodulation
        self.conv = Conv2dWeightModulate(features, 3, kernel_size=1, demodulate=False)
        # Bias
        self.bias = nn.Parameter(torch.zeros(3))
        # Activation function
        self.activation = nn.LeakyReLU(0.2, True)

    def forward(self, x: torch.Tensor, w: torch.Tensor):
        """
        * `x` is the input feature map of shape `[batch_size, in_features, height, width]`
        * `w` is $w$ with shape `[batch_size, d_latent]`
        """
        # Get style vector $s$
        style = self.to_style(w)
        # Weight modulated convolution
        x = self.conv(x, style)
        # Add bias and evaluate activation function
        return self.activation(x + self.bias[None, :, None, None])


class Conv2dWeightModulate(nn.Module):
    """
    ### Convolution with Weight Modulation and Demodulation

    This layer scales the convolution weights by the style vector and demodulates by normalizing it.
    """

    def __init__(self, in_features: int, out_features: int, kernel_size: int,
                 demodulate: float = True, eps: float = 1e-8):
        """
        * `in_features` is the number of features in the input feature map
        * `out_features` is the number of features in the output feature map
        * `kernel_size` is the size of the convolution kernel
        * `demodulate` is flag whether to normalize weights by its standard deviation
        * `eps` is the $\epsilon$ for normalizing
        """
        super().__init__()
        # Number of output features
        self.out_features = out_features
        # Whether to normalize weights
        self.demodulate = demodulate
        # Padding size
        self.padding = (kernel_size - 1) // 2

        # [Weights parameter with equalized learning rate](#equalized_weight)
        self.weight = EqualizedWeight([out_features, in_features, kernel_size, kernel_size])
        # $\epsilon$
        self.eps = eps

    def forward(self, x: torch.Tensor, s: torch.Tensor):
        """
        * `x` is the input feature map of shape `[batch_size, in_features, height, width]`
        * `s` is style based scaling tensor of shape `[batch_size, in_features]`
        """

        # Get batch size, height and width
        b, _, h, w = x.shape

        # Reshape the scales
        s = s[:, None, :, None, None]
        # Get [learning rate equalized weights](#equalized_weight)
        weights = self.weight()[None, :, :, :, :]
        # $$w`_{i,j,k} = s_i * w_{i,j,k}$$
        # where $i$ is the input channel, $j$ is the output channel, and $k$ is the kernel index.
        #
        # The result has shape `[batch_size, out_features, in_features, kernel_size, kernel_size]`
        weights = weights * s

        # Demodulate
        if self.demodulate:
            # $$\sigma_j = \sqrt{\sum_{i,k} (w'_{i, j, k})^2 + \epsilon}$$
            sigma_inv = torch.rsqrt((weights ** 2).sum(dim=(2, 3, 4), keepdim=True) + self.eps)
            # $$w''_{i,j,k} = \frac{w'_{i,j,k}}{\sqrt{\sum_{i,k} (w'_{i, j, k})^2 + \epsilon}}$$
            weights = weights * sigma_inv

        # Reshape `x`
        x = x.reshape(1, -1, h, w)

        # Reshape weights
        _, _, *ws = weights.shape
        weights = weights.reshape(b * self.out_features, *ws)

        # Use grouped convolution to efficiently calculate the convolution with sample wise kernel.
        # i.e. we have a different kernel (weights) for each sample in the batch
        x = F.conv2d(x, weights, padding=self.padding, groups=b)

        # Reshape `x` to `[batch_size, out_features, height, width]` and return
        return x.reshape(-1, self.out_features, h, w)


class Discriminator(nn.Module):
    """
    <a id="discriminator"></a>
    ## StyleGAN 2 Discriminator

    ![Discriminator](style_gan2_disc.svg)

    Discriminator first transforms the image to a feature map of the same resolution and then
    runs it through a series of blocks with residual connections.
    The resolution is down-sampled by $2 \times$ at each block while doubling the
    number of features.
    """

    def __init__(self, log_resolution: int, n_features: int = 64, max_features: int = 512):
        """
        * `log_resolution` is the $\log_2$ of image resolution
        * `n_features` number of features in the convolution layer at the highest resolution (first block)
        * `max_features` maximum number of features in any generator block
        """
        super().__init__()

        # Layer to convert RGB image to a feature map with `n_features` number of features.
        self.from_rgb = nn.Sequential(
            EqualizedConv2d(3, n_features, 1),
            nn.LeakyReLU(0.2, True),
        )

        # Calculate the number of features for each block.
        #
        # Something like `[64, 128, 256, 512, 512, 512]`.
        features = [min(max_features, n_features * (2 ** i)) for i in range(log_resolution - 1)]
        # Number of [discirminator blocks](#discriminator_block)
        n_blocks = len(features) - 1
        # Discriminator blocks
        blocks = [DiscriminatorBlock(features[i], features[i + 1]) for i in range(n_blocks)]
        self.blocks = nn.Sequential(*blocks)

        # [Mini-batch Standard Deviation](#mini_batch_std_dev)
        self.std_dev = MiniBatchStdDev()
        # Number of features after adding the standard deviations map
        final_features = features[-1] + 1
        # Final $3 \times 3$ convolution layer
        self.conv = EqualizedConv2d(final_features, final_features, 3)
        # Final linear layer to get the classification
        self.final = EqualizedLinear(2 * 2 * final_features, 1)

    def forward(self, x: torch.Tensor):
        """
        * `x` is the input image of shape `[batch_size, 3, height, width]`
        """

        # Try to normalize the image (this is totally optional, but sped up the early training a little)
        x = x - 0.5
        # Convert from RGB
        x = self.from_rgb(x)
        # Run through the [discriminator blocks](#discriminator_block)
        x = self.blocks(x)

        # Calculate and append [mini-batch standard deviation](#mini_batch_std_dev)
        x = self.std_dev(x)
        # $3 \times 3$ convolution
        x = self.conv(x)
        # Flatten
        x = x.reshape(x.shape[0], -1)
        # Return the classification score
        return self.final(x)


class DiscriminatorBlock(nn.Module):
    """
    <a id="discriminator_black"></a>
    ### Discriminator Block

    ![Discriminator block](discriminator_block.svg)

    Discriminator block consists of two $3 \times 3$ convolutions with a residual connection.
    """

    def __init__(self, in_features, out_features):
        """
        * `in_features` is the number of features in the input feature map
        * `out_features` is the number of features in the output feature map
        """
        super().__init__()
        # Down-sampling and $1 \times 1$ convolution layer for the residual connection
        self.residual = nn.Sequential(DownSample(),
                                      EqualizedConv2d(in_features, out_features, kernel_size=1))

        # Two $3 \times 3$ convolutions
        self.block = nn.Sequential(
            EqualizedConv2d(in_features, in_features, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
            EqualizedConv2d(in_features, out_features, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
        )

        # Down-sampling layer
        self.down_sample = DownSample()

        # Scaling factor $\frac{1}{\sqrt 2}$ after adding the residual
        self.scale = 1 / math.sqrt(2)

    def forward(self, x):
        # Get the residual connection
        residual = self.residual(x)

        # Convolutions
        x = self.block(x)
        # Down-sample
        x = self.down_sample(x)

        # Add the residual and scale
        return (x + residual) * self.scale


class MiniBatchStdDev(nn.Module):
    """
    <a id="mini_batch_std_dev"></a>

    ### Mini-batch Standard Deviation

    Mini-batch standard deviation calculates the standard deviation
    across a mini-batch (or a subgroups within the mini-batch)
    for each feature in the feature map. Then it takes the mean of all
    the standard deviations and appends it to the feature map as one extra feature.
    """

    def __init__(self, group_size: int = 4):
        """
        * `group_size` is the number of samples to calculate standard deviation across.
        """
        super().__init__()
        self.group_size = group_size

    def forward(self, x: torch.Tensor):
        """
        * `x` is the feature map
        """
        # Check if the batch size is divisible by the group size
        assert x.shape[0] % self.group_size == 0
        # Split the samples into groups of `group_size`, we flatten the feature map to a single dimension
        # since we want to calculate the standard deviation for each feature.
        grouped = x.view(self.group_size, -1)
        # Calculate the standard deviation for each feature among `group_size` samples
        # $$\mu_{i} = \frac{1}{N} \sum_g x_{g,i} \\
        #   \sigma_{i} = \sqrt{\frac{1}{N} \sum_g (x_{g,i} - \mu_i)^2  + \epsilon}$$
        std = torch.sqrt(grouped.var(dim=0) + 1e-8)
        # Get the mean standard deviation
        std = std.mean().view(1, 1, 1, 1)
        # Expand the standard deviation to append to the feature map
        b, _, h, w = x.shape
        std = std.expand(b, -1, h, w)
        # Append (concatenate) the standard deviations to the feature map
        return torch.cat([x, std], dim=1)


class DownSample(nn.Module):
    """
    <a id="down_sample"></a>
    ### Down-sample

    The down-sample operation [smoothens](#smooth) each feature channel and
     scale $2 \times$ using bilinear interpolation.
    This is based on the paper
     [Making Convolutional Networks Shift-Invariant Again](https://arxiv.org/abs/1904.11486).
    """

    def __init__(self):
        super().__init__()
        # Smoothing layer
        self.smooth = Smooth()

    def forward(self, x: torch.Tensor):
        # Smoothing or blurring
        x = self.smooth(x)
        # Scaled down
        return F.interpolate(x, (x.shape[2] // 2, x.shape[3] // 2), mode='bilinear', align_corners=False)


class UpSample(nn.Module):
    """
    <a id="up_sample"></a>
    ### Up-sample

    The up-sample operation scales the image up by $2 \times$ and [smoothens](#smooth) each feature channel.
    This is based on the paper
     [Making Convolutional Networks Shift-Invariant Again](https://arxiv.org/abs/1904.11486).
    """

    def __init__(self):
        super().__init__()
        # Up-sampling layer
        self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        # Smoothing layer
        self.smooth = Smooth()

    def forward(self, x: torch.Tensor):
        # Up-sample and smoothen
        return self.smooth(self.up_sample(x))


class Smooth(nn.Module):
    """
    <a id="smooth"></a>
    ### Smoothing Layer

    This layer blurs each channel
    """

    def __init__(self):
        super().__init__()
        # Blurring kernel
        kernel = [[1, 2, 1],
                  [2, 4, 2],
                  [1, 2, 1]]
        # Convert the kernel to a PyTorch tensor
        kernel = torch.tensor([[kernel]], dtype=torch.float)
        # Normalize the kernel
        kernel /= kernel.sum()
        # Save kernel as a fixed parameter (no gradient updates)
        self.kernel = nn.Parameter(kernel, requires_grad=False)
        # Padding layer
        self.pad = nn.ReplicationPad2d(1)

    def forward(self, x: torch.Tensor):
        # Get shape of the input feature map
        b, c, h, w = x.shape
        # Reshape for smoothening
        x = x.view(-1, 1, h, w)

        # Add padding
        x = self.pad(x)

        # Smoothen (blur) with the kernel
        x = F.conv2d(x, self.kernel)

        # Reshape and return
        return x.view(b, c, h, w)


class EqualizedLinear(nn.Module):
    """
    <a id="equalized_linear"></a>
    ## Learning-rate Equalized Linear Layer

    This uses [learning-rate equalized weights]($equalized_weights) for a linear layer.
    """

    def __init__(self, in_features: int, out_features: int, bias: float = 0.):
        """
        * `in_features` is the number of features in the input feature map
        * `out_features` is the number of features in the output feature map
        * `bias` is the bias initialization constant
        """

        super().__init__()
        # [Learning-rate equalized weights]($equalized_weights)
        self.weight = EqualizedWeight([out_features, in_features])
        # Bias
        self.bias = nn.Parameter(torch.ones(out_features) * bias)

    def forward(self, x: torch.Tensor):
        # Linear transformation
        return F.linear(x, self.weight(), bias=self.bias)


class EqualizedConv2d(nn.Module):
    """
    <a id="equalized_conv2d"></a>
    ## Learning-rate Equalized 2D Convolution Layer

    This uses [learning-rate equalized weights]($equalized_weights) for a convolution layer.
    """

    def __init__(self, in_features: int, out_features: int,
                 kernel_size: int, padding: int = 0):
        """
        * `in_features` is the number of features in the input feature map
        * `out_features` is the number of features in the output feature map
        * `kernel_size` is the size of the convolution kernel
        * `padding` is the padding to be added on both sides of each size dimension
        """
        super().__init__()
        # Padding size
        self.padding = padding
        # [Learning-rate equalized weights]($equalized_weights)
        self.weight = EqualizedWeight([out_features, in_features, kernel_size, kernel_size])
        # Bias
        self.bias = nn.Parameter(torch.ones(out_features))

    def forward(self, x: torch.Tensor):
        # Convolution
        return F.conv2d(x, self.weight(), bias=self.bias, padding=self.padding)


class EqualizedWeight(nn.Module):
    """
    <a id="equalized_weight"></a>
    ## Learning-rate Equalized Weights Parameter

    This is based on equalized learning rate introduced in the Progressive GAN paper.
    Instead of initializing weights at $\mathcal{N}(0,c)$ they initialize weights
    to $\mathcal{N}(0, 1)$ and then multiply them by $c$ when using it.
    $$w_i = c \hat{w}_i$$

    The gradients on stored parameters $\hat{w}$ get multiplied by $c$ but this doesn't have
    an affect since optimizers such as Adam normalize them by a running mean of the squared gradients.

    The optimizer updates on $\hat{w}$ are proportionate to the learning rate $\lambda$.
    But the effective weights $w$ get updated proportionately to $c \lambda$.
    Without equalized learning rate, the effective weights will get updated proportionately to just $\lambda$.

    So we are effectively scaling the learning rate by $c$ for these weight parameters.
    """

    def __init__(self, shape: List[int]):
        """
        * `shape` is the shape of the weight parameter
        """
        super().__init__()

        # He initialization constant
        self.c = 1 / math.sqrt(np.prod(shape[1:]))
        # Initialize the weights with $\mathcal{N}(0, 1)$
        self.weight = nn.Parameter(torch.randn(shape))
        # Weight multiplication coefficient

    def forward(self):
        # Multiply the weights by $c$ and return
        return self.weight * self.c


class GradientPenalty(nn.Module):
    """
    <a id="gradient_penalty"></a>
    ## Gradient Penalty

    This is the $R_1$ regularization penality from the paper
    [Which Training Methods for GANs do actually Converge?](https://arxiv.org/abs/1801.04406).

    $$R_1(\psi) = \frac{\gamma}{2} \mathbb{E}_{p_\mathcal{D}(x)}
    \Big[\Vert \nabla_x D_\psi(x)^2 \Vert\Big]$$

    That is we try to reduce the L2 norm of gradients of the discriminator with
    respect to images, for real images ($P_\mathcal{D}$).
    """

    def forward(self, x: torch.Tensor, d: torch.Tensor):
        """
        * `x` is $x \sim \mathcal{D}$
        * `d` is $D(x)$
        """

        # Get batch size
        batch_size = x.shape[0]

        # Calculate gradients of $D(x)$ with respect to $x$.
        # `grad_outputs` is set to $1$ since we want the gradients of $D(x)$,
        # and we need to create and retain graph since we have to compute gradients
        # with respect to weight on this loss.
        gradients, *_ = torch.autograd.grad(outputs=d,
                                            inputs=x,
                                            grad_outputs=d.new_ones(d.shape),
                                            create_graph=True)

        # Reshape gradients to calculate the norm
        gradients = gradients.reshape(batch_size, -1)
        # Calculate the norm $\Vert \nabla_{x} D(x)^2 \Vert$
        norm = gradients.norm(2, dim=-1)
        # Return the loss $\Vert \nabla_x D_\psi(x)^2 \Vert$
        return torch.mean(norm ** 2)


class PathLengthPenalty(nn.Module):
    """
    <a id="path_length_penalty"></a>
    ## Path Length Penalty

    This regularization encourages a fixed-size step in $w$ to result in a fixed-magnitude
    change in the image.

    $$\mathbb{E}_{w \sim f(z), y \sim \mathcal{N}(0, \mathbf{I})}
      \Big(\Vert \mathbf{J}^\top_{w} y \Vert_2 - a \Big)^2$$

    where $\mathbf{J}_w$ is the Jacobian
    $\mathbf{J}_w = \frac{\partial g}{\partial w}$,
    $w$ are sampled from $w \in \mathcal{W}$ from the mapping network, and
    $y$ are images with noise $\mathcal{N}(0, \mathbf{I})$.

    $a$ is the exponential moving average of $\Vert \mathbf{J}^\top_{w} y \Vert_2$
    as the training progresses.

    $\mathbf{J}^\top_{w} y$ is calculated without explicitly calculating the Jacobian using
    $$\mathbf{J}^\top_{w} y = \nabla_w \big(g(w) \cdot y \big)$$
    """

    def __init__(self, beta: float):
        """
        * `beta` is the constant $\beta$ used to calculate the exponential moving average $a$
        """
        super().__init__()

        # $\beta$
        self.beta = beta
        # Number of steps calculated $N$
        self.steps = nn.Parameter(torch.tensor(0.), requires_grad=False)
        # Exponential sum of $\mathbf{J}^\top_{w} y$
        # $$\sum^N_{i=1} \beta^{(N - i)}[\mathbf{J}^\top_{w} y]_i$$
        # where $[\mathbf{J}^\top_{w} y]_i$ is the value of it at $i$-th step of training
        self.exp_sum_a = nn.Parameter(torch.tensor(0.), requires_grad=False)

    def forward(self, w: torch.Tensor, x: torch.Tensor):
        """
        * `w` is the batch of $w$ of shape `[batch_size, d_latent]`
        * `x` are the generated images of shape `[batch_size, 3, height, width]`
        """

        # Get the device
        device = x.device
        # Get number of pixels
        image_size = x.shape[2] * x.shape[3]
        # Calculate $y \in \mathcal{N}(0, \mathbf{I})$
        y = torch.randn(x.shape, device=device)
        # Calculate $\big(g(w) \cdot y \big)$ and normalize by the square root of image size.
        # This is scaling is not mentioned in the paper but was present in
        # [their implementation](https://github.com/NVlabs/stylegan2/blob/master/training/loss.py#L167).
        output = (x * y).sum() / math.sqrt(image_size)

        # Calculate gradients to get $\mathbf{J}^\top_{w} y$
        gradients, *_ = torch.autograd.grad(outputs=output,
                                            inputs=w,
                                            grad_outputs=torch.ones(output.shape, device=device),
                                            create_graph=True)

        # Calculate L2-norm of $\mathbf{J}^\top_{w} y$
        norm = (gradients ** 2).sum(dim=2).mean(dim=1).sqrt()

        # Regularize after first step
        if self.steps > 0:
            # Calculate $a$
            # $$\frac{1}{1 - \beta^N} \sum^N_{i=1} \beta^{(N - i)}[\mathbf{J}^\top_{w} y]_i$$
            a = self.exp_sum_a / (1 - self.beta ** self.steps)
            # Calculate the penalty
            # $$\mathbb{E}_{w \sim f(z), y \sim \mathcal{N}(0, \mathbf{I})}
            # \Big(\Vert \mathbf{J}^\top_{w} y \Vert_2 - a \Big)^2$$
            loss = torch.mean((norm - a) ** 2)
        else:
            # Return a dummy loss if we can't calculate $a$
            loss = norm.new_tensor(0)

        # Calculate the mean of $\Vert \mathbf{J}^\top_{w} y \Vert_2$
        mean = norm.mean().detach()
        # Update exponential sum
        self.exp_sum_a.mul_(self.beta).add_(mean, alpha=1 - self.beta)
        # Increment $N$
        self.steps.add_(1.)

        # Return the penalty
        return loss
