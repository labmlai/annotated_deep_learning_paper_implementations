"""
---
title: Instance Normalization
summary: >
 A PyTorch implementation/tutorial of instance normalization.
---

# Instance Normalization

This is a [PyTorch](https://pytorch.org) implementation of
[Instance Normalization: The Missing Ingredient for Fast Stylization](https://arxiv.org/abs/1607.08022).

Instance normalization was introduced to improve [style transfer](https://paperswithcode.com/task/style-transfer).
It is based on the observation that stylization should not depend on the contrast of the content image.
The "contrast normalization" is

$$y_{t,i,j,k} = \frac{x_{t,i,j,k}}{\sum_{l=1}^H \sum_{m=1}^W x_{t,i,l,m}}$$

where $x$ is a batch of images with dimensions image index $t$,
feature channel $i$, and
spatial position $j, k$.

Since it's hard for a convolutional network to learn "contrast normalization", this paper
introduces instance normalization which does that.

Here's a [CIFAR 10 classification model](experiment.html) that uses instance normalization.
"""

import torch
from torch import nn

from labml_helpers.module import Module


class InstanceNorm(Module):
    r"""
    ## Instance Normalization Layer

    Instance normalization layer $\text{IN}$ normalizes the input $X$ as follows:

    When input $X \in \mathbb{R}^{B \times C \times H \times W}$ is a batch of image representations,
    where $B$ is the batch size, $C$ is the number of channels, $H$ is the height and $W$ is the width.
    $\gamma \in \mathbb{R}^{C}$ and $\beta \in \mathbb{R}^{C}$. The affine transformation with $gamma$ and
    $beta$ are optional.

    $$\text{IN}(X) = \gamma
    \frac{X - \underset{H, W}{\mathbb{E}}[X]}{\sqrt{\underset{H, W}{Var}[X] + \epsilon}}
    + \beta$$
    """

    def __init__(self, channels: int, *,
                 eps: float = 1e-5, affine: bool = True):
        """
        * `channels` is the number of features in the input
        * `eps` is $\epsilon$, used in $\sqrt{Var[X] + \epsilon}$ for numerical stability
        * `affine` is whether to scale and shift the normalized value
        """
        super().__init__()

        self.channels = channels

        self.eps = eps
        self.affine = affine
        # Create parameters for $\gamma$ and $\beta$ for scale and shift
        if self.affine:
            self.scale = nn.Parameter(torch.ones(channels))
            self.shift = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor):
        """
        `x` is a tensor of shape `[batch_size, channels, *]`.
        `*` denotes any number of (possibly 0) dimensions.
         For example, in an image (2D) convolution this will be
        `[batch_size, channels, height, width]`
        """
        # Keep the original shape
        x_shape = x.shape
        # Get the batch size
        batch_size = x_shape[0]
        # Sanity check to make sure the number of features is the same
        assert self.channels == x.shape[1]

        # Reshape into `[batch_size, channels, n]`
        x = x.view(batch_size, self.channels, -1)

        # Calculate the mean across last dimension
        # i.e. the means for each feature  $\mathbb{E}[x_{t,i}]$
        mean = x.mean(dim=[-1], keepdim=True)
        # Calculate the squared mean across first and last dimension;
        # i.e. the means for each feature $\mathbb{E}[(x_{t,i}^2]$
        mean_x2 = (x ** 2).mean(dim=[-1], keepdim=True)
        # Variance for each feature $Var[x_{t,i}] = \mathbb{E}[x_{t,i}^2] - \mathbb{E}[x_{t,i}]^2$
        var = mean_x2 - mean ** 2

        # Normalize $$\hat{x}_{t,i} = \frac{x_{t,i} - \mathbb{E}[x_{t,i}]}{\sqrt{Var[x_{t,i}] + \epsilon}}$$
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        x_norm = x_norm.view(batch_size, self.channels, -1)

        # Scale and shift $$y_{t,i} =\gamma_i \hat{x}_{t,i} + \beta_i$$
        if self.affine:
            x_norm = self.scale.view(1, -1, 1) * x_norm + self.shift.view(1, -1, 1)

        # Reshape to original and return
        return x_norm.view(x_shape)


def _test():
    """
    Simple test
    """
    from labml.logger import inspect

    x = torch.zeros([2, 6, 2, 4])
    inspect(x.shape)
    bn = InstanceNorm(6)

    x = bn(x)
    inspect(x.shape)


#
if __name__ == '__main__':
    _test()
