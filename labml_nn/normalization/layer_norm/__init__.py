"""
---
title: Layer Normalization
summary: >
 A PyTorch implementation/tutorial of layer normalization.
---

# Layer Normalization

This is a [PyTorch](https://pytorch.org) implementation of
[Layer Normalization](https://arxiv.org/abs/1607.06450).

### Limitations of [Batch Normalization](../batch_norm/index.html)

* You need to maintain running means.
* Tricky for RNNs. Do you need different normalizations for each step?
* Doesn't work with small batch sizes;
large NLP models are usually trained with small batch sizes.
* Need to compute means and variances across devices in distributed training

## Layer Normalization

Layer normalization is a simpler normalization method that works
on a wider range of settings.
Layer normalization transformers the inputs to have zero mean and unit variance
across the features.
*Note that batch normalization, fixes the zero mean and unit variance for each vector.
Layer normalization does it for each batch across all elements.

Layer normalization is generally used for NLP tasks.

We have used layer normalization in most of the
[transformer implementations](../../transformers/gpt/index.html).
"""

import torch
from torch import nn


class LayerNorm(nn.Module):
    """
    ## Layer Normalization
    """

    def __init__(self, channels: int, *,
                 eps: float = 1e-5, momentum: float = 0.1,
                 affine: bool = True, track_running_stats: bool = True):
        """
        * `channels` is the number of features in the input
        * `eps` is $\epsilon$, used in $\sqrt{Var[x^{(k)}] + \epsilon}$ for numerical stability
        * `momentum` is the momentum in taking the exponential moving average
        * `affine` is whether to scale and shift the normalized value
        * `track_running_stats` is whether to calculate the moving averages or mean and variance

        We've tried to use the same names for arguments as PyTorch `BatchNorm` implementation.
        """
        super().__init__()

        self.channels = channels

        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        # Create parameters for $\gamma$ and $\beta$ for scale and shift
        if self.affine:
            self.scale = nn.Parameter(torch.ones(channels))
            self.shift = nn.Parameter(torch.zeros(channels))
        # Create buffers to store exponential moving averages of
        # mean $\mathbb{E}[x^{(k)}]$ and variance $Var[x^{(k)}]$
        if self.track_running_stats:
            self.register_buffer('exp_mean', torch.zeros(channels))
            self.register_buffer('exp_var', torch.ones(channels))

    def forward(self, x: torch.Tensor):
        """
        `x` is a tensor of shape `[batch_size, channels, *]`.
        `*` could be any (even *) dimensions.
         For example, in an image (2D) convolution this will be
        `[batch_size, channels, height, width]`
        """
        # Keep the original shape
        x_shape = x.shape
        # Get the batch size
        batch_size = x_shape[0]
        # Sanity check to make sure the number of features is same
        assert self.channels == x.shape[1]

        # Reshape into `[batch_size, channels, n]`
        x = x.view(batch_size, self.channels, -1)

        # We will calculate the mini-batch mean and variance
        # if we are in training mode or if we have not tracked exponential moving averages
        if self.training or not self.track_running_stats:
            # Calculate the mean across first and last dimension;
            # i.e. the means for each feature $\mathbb{E}[x^{(k)}]$
            mean = x.mean(dim=[0, 2])
            # Calculate the squared mean across first and last dimension;
            # i.e. the means for each feature $\mathbb{E}[(x^{(k)})^2]$
            mean_x2 = (x ** 2).mean(dim=[0, 2])
            # Variance for each feature $Var[x^{(k)}] = \mathbb{E}[(x^{(k)})^2] - \mathbb{E}[x^{(k)}]^2$
            var = mean_x2 - mean ** 2

            # Update exponential moving averages
            if self.training and self.track_running_stats:
                self.exp_mean = (1 - self.momentum) * self.exp_mean + self.momentum * mean
                self.exp_var = (1 - self.momentum) * self.exp_var + self.momentum * var
        # Use exponential moving averages as estimates
        else:
            mean = self.exp_mean
            var = self.exp_var

        # Normalize $$\hat{x}^{(k)} = \frac{x^{(k)} - \mathbb{E}[x^{(k)}]}{\sqrt{Var[x^{(k)}] + \epsilon}}$$
        x_norm = (x - mean.view(1, -1, 1)) / torch.sqrt(var + self.eps).view(1, -1, 1)
        # Scale and shift $$y^{(k)} =\gamma^{(k)} \hat{x}^{(k)} + \beta^{(k)}$$
        if self.affine:
            x_norm = self.scale.view(1, -1, 1) * x_norm + self.shift.view(1, -1, 1)

        # Reshape to original and return
        return x_norm.view(x_shape)
