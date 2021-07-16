"""
---
title: Deep Residual Learning for Image Recognition (ResNet)
summary: >
 A PyTorch implementation/tutorial of Deep Residual Learning for Image Recognition (ResNet).
---

# Deep Residual Learning for Image Recognition (ResNet)

This is a [PyTorch](https://pytorch.org) implementation of the paper
[Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385).

ResNets train layers as residual functions to overcome the
*degradation problem*.
The degradation problem is the accuracy of deep neural networks degrading when
the number of layers becomes very high.
The accuracy increases as the number of layers increase, then saturates,
and then starts to degrade.

The paper argues that deeper models should perform at least as well as shallower
models because the extra layers can just learn to perform an identity mapping.

## Residual Learning

If $\mathcal{H}(x)$ is the mapping that needs to be learned by a few layers,
they train the residual function

$$\mathcal{F}(x) = \mathcal{H}(x) - x$$

instead. And the original function becomes $\mathcal{F}(x) + x$.

In this case, learning identity mapping for $\mathcal{H}(x)$ is
equivalent to learning $\mathcal{F}(x)$ to be $0$, which is easier to
learn.

In the parameterized form this can be written as,

$$\mathcal{F}(x, \{W_i\}) + x$$

and when the feature map sizes of $\mathcal{F}(x, {W_i})$ and $x$ are different
the paper suggests doing a linear projection, with learned weights $W_s$.

$$\mathcal{F}(x, \{W_i\}) + W_s x$$

Paper experimented with zero padding instead of linear projections and found linear projections
to work better. Also when the feature map sizes match they found identity mapping
to be better than linear projections.

$\mathcal{F}$ should have more than one layer, otherwise the sum $\mathcal{F}(x, \{W_i\}) + W_s x$
also won't have non-linearities and will be like a linear layer.

Here is [the training code](experiment.html) for training a ResNet on CIFAR-10.

[![View Run](https://img.shields.io/badge/labml-experiment-brightgreen)](https://app.labml.ai/run/fc5ad600e4af11ebbafd23b8665193c1)
"""

from typing import List, Optional

import torch
from torch import nn

from labml_helpers.module import Module


class ShortcutProjection(Module):
    """
    ## Linear projections for shortcut connection

    This does the $W_s x$ projection described above.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int):
        """
        * `in_channels` is the number of channels in $x$
        * `out_channels` is the number of channels in $\mathcal{F}(x, \{W_i\})$
        * `stride` is the stride length in the convolution operation for $F$.
        We do the same stride on the shortcut connection, to match the feature-map size.
        """
        super().__init__()

        # Convolution layer for linear projection $W_s x$
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        # Paper suggests adding batch normalization after each convolution operation
        self.bn = nn.BatchNorm2d(out_channels)

    def __call__(self, x: torch.Tensor):
        # Convolution and batch normalization
        return self.bn(self.conv(x))


class ResidualBlock(Module):
    """
    <a id="residual_block"></a>
    ## Residual Block

    This implements the residual block described in the paper.
    It has two $3 \times 3$ convolution layers.

    ![Residual Block](residual_block.svg)

    The first convolution layer maps from `in_channels` to `out_channels`,
    where the `out_channels` is higher than `in_channels` when we reduce the
    feature map size with a stride length greater than $1$.

    The second convolution layer maps from `out_channels` to `out_channels` and
    always has a stride length of 1.

    Both convolution layers are followed by batch normalization.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int):
        """
        * `in_channels` is the number of channels in $x$
        * `out_channels` is the number of output channels
        * `stride` is the stride length in the convolution operation.
        """
        super().__init__()

        # First $3 \times 3$ convolution layer, this maps to `out_channels`
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        # Batch normalization after the first convolution
        self.bn1 = nn.BatchNorm2d(out_channels)
        # First activation function (ReLU)
        self.act1 = nn.ReLU()

        # Second $3 \times 3$ convolution layer
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        # Batch normalization after the second convolution
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection should be a projection if the stride length is not $1$
        # of if the number of channels change
        if stride != 1 or in_channels != out_channels:
            # Projection $W_s x$
            self.shortcut = ShortcutProjection(in_channels, out_channels, stride)
        else:
            # Identity $x$
            self.shortcut = nn.Identity()

        # Second activation function (ReLU) (after adding the shortcut)
        self.act2 = nn.ReLU()

    def __call__(self, x: torch.Tensor):
        """
        * `x` is the input of shape `[batch_size, in_channels, height, width]`
        """
        # Get the shortcut connection
        shortcut = self.shortcut(x)
        # First convolution and activation
        x = self.act1(self.bn1(self.conv1(x)))
        # Second convolution
        x = self.bn2(self.conv2(x))
        # Activation function after adding the shortcut
        return self.act2(x + shortcut)


class BottleneckResidualBlock(Module):
    """
    <a id="bottleneck_residual_block"></a>
    ## Bottleneck Residual Block

    This implements the bottleneck block described in the paper.
    It has $1 \times 1$, $3 \times 3$, and $1 \times 1$ convolution layers.

    ![Bottlenext Block](bottleneck_block.svg)

    The first convolution layer maps from `in_channels` to `bottleneck_channels` with a $1x1$
    convolution,
    where the `bottleneck_channels` is lower than `in_channels`.

    The second $3x3$ convolution layer maps from `bottleneck_channels` to `bottleneck_channels`.
    This can have a stride length greater than $1$ when we want to compress the
    feature map size.

    The third, final $1x1$ convolution layer maps to `out_channels`.
    `out_channels` is higher than `in_channels` if the stride length is greater than $1$;
    otherwise, $out_channels$ is equal to `in_channels`.

    `bottleneck_channels` is less than `in_channels` and the $3x3$ convolution is performed
    on this shrunk space (hence the bottleneck). The two $1x1$ convolution decreases and increases
    the number of channels.
    """

    def __init__(self, in_channels: int, bottleneck_channels: int, out_channels: int, stride: int):
        """
        * `in_channels` is the number of channels in $x$
        * `bottleneck_channels` is the number of channels for the $3x3$ convlution
        * `out_channels` is the number of output channels
        * `stride` is the stride length in the $3x3$ convolution operation.
        """
        super().__init__()

        # First $1 \times 1$ convolution layer, this maps to `bottleneck_channels`
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, stride=1)
        # Batch normalization after the first convolution
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)
        # First activation function (ReLU)
        self.act1 = nn.ReLU()

        # Second $3 \times 3$ convolution layer
        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, stride=stride, padding=1)
        # Batch normalization after the second convolution
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        # Second activation function (ReLU)
        self.act2 = nn.ReLU()

        # Third $1 \times 1$ convolution layer, this maps to `out_channels`.
        self.conv3 = nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1, stride=1)
        # Batch normalization after the second convolution
        self.bn3 = nn.BatchNorm2d(out_channels)

        # Shortcut connection should be a projection if the stride length is not $1$
        # of if the number of channels change
        if stride != 1 or in_channels != out_channels:
            # Projection $W_s x$
            self.shortcut = ShortcutProjection(in_channels, out_channels, stride)
        else:
            # Identity $x$
            self.shortcut = nn.Identity()

        # Second activation function (ReLU) (after adding the shortcut)
        self.act3 = nn.ReLU()

    def __call__(self, x: torch.Tensor):
        """
        * `x` is the input of shape `[batch_size, in_channels, height, width]`
        """
        # Get the shortcut connection
        shortcut = self.shortcut(x)
        # First convolution and activation
        x = self.act1(self.bn1(self.conv1(x)))
        # Second convolution and activation
        x = self.act2(self.bn2(self.conv2(x)))
        # Third convolution
        x = self.bn3(self.conv3(x))
        # Activation function after adding the shortcut
        return self.act3(x + shortcut)


class ResNetBase(Module):
    """
    ## ResNet Model

    This is a the base of the resnet model without
    the final linear layer and softmax for classification.

    The resnet is made of stacked [residual blocks](#residual_block) or
    [bottleneck residual blocks](#bottleneck_residual_block).
    The feature map size is halved after a few blocks with a block of stride length $2$.
    The number of channels is increased when the feature map size is reduced.
    Finally the feature map is average pooled to get a vector representation.
    """

    def __init__(self, n_blocks: List[int], n_channels: List[int],
                 bottlenecks: Optional[List[int]] = None,
                 img_channels: int = 3, first_kernel_size: int = 7):
        """
        * `n_blocks` is a list of of number of blocks for each feature map size.
        * `n_channels` is the number of channels for each feature map size.
        * `bottlenecks` is the number of channels the bottlenecks.
        If this is `None`, [residual blocks](#residual_block) are used.
        * `img_channels` is the number of channels in the input.
        * `first_kernel_size` is the kernel size of the initial convolution layer
        """
        super().__init__()

        # Number of blocks and number of channels for each feature map size
        assert len(n_blocks) == len(n_channels)
        # If [bottleneck residual blocks](#bottleneck_residual_block) are used,
        # the number of channels in bottlenecks should be provided for each feature map size
        assert bottlenecks is None or len(bottlenecks) == len(n_channels)

        # Initial convolution layer maps from `img_channels` to number of channels in the first
        # residual block (`n_channels[0]`)
        self.conv = nn.Conv2d(img_channels, n_channels[0],
                              kernel_size=first_kernel_size, stride=1, padding=first_kernel_size // 2)
        # Batch norm after initial convolution
        self.bn = nn.BatchNorm2d(n_channels[0])

        # List of blocks
        blocks = []
        # Number of channels from previous layer (or block)
        prev_channels = n_channels[0]
        # Loop through each feature map size
        for i, channels in enumerate(n_channels):
            # The first block for the new feature map size, will have a stride length of $2$
            # except fro the very first block
            stride = 2 if len(blocks) == 0 else 1

            if bottlenecks is None:
                # [residual blocks](#residual_block) that maps from `prev_channels` to `channels`
                blocks.append(ResidualBlock(prev_channels, channels, stride=stride))
            else:
                # [bottleneck residual blocks](#bottleneck_residual_block)
                # that maps from `prev_channels` to `channels`
                blocks.append(BottleneckResidualBlock(prev_channels, bottlenecks[i], channels,
                                                      stride=stride))

            # Change the number of channels
            prev_channels = channels
            # Add rest of the blocks - no change in feature map size or channels
            for _ in range(n_blocks[i] - 1):
                if bottlenecks is None:
                    # [residual blocks](#residual_block)
                    blocks.append(ResidualBlock(channels, channels, stride=1))
                else:
                    # [bottleneck residual blocks](#bottleneck_residual_block)
                    blocks.append(BottleneckResidualBlock(channels, bottlenecks[i], channels, stride=1))

        # Stack the blocks
        self.blocks = nn.Sequential(*blocks)

    def __call__(self, x: torch.Tensor):
        """
        * `x` has shape `[batch_size, img_channels, height, width]`
        """

        # Initial convolution and batch normalization
        x = self.bn(self.conv(x))
        # Residual (or bottleneck) blocks
        x = self.blocks(x)
        # Change `x` from shape `[batch_size, channels, h, w]` to `[batch_size, channels, h * w]`
        x = x.view(x.shape[0], x.shape[1], -1)
        # Global average pooling
        return x.mean(dim=-1)
