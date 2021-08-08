"""
---
title: Batch-Channel Normalization
summary: >
 A PyTorch implementation/tutorial of Batch-Channel Normalization.
---

# Batch-Channel Normalization

This is a [PyTorch](https://pytorch.org) implementation of Batch-Channel Normalization from the paper
 [Micro-Batch Training with Batch-Channel Normalization and Weight Standardization](https://arxiv.org/abs/1903.10520).
We also have an [annotated implementation of Weight Standardization](../weight_standardization/index.html).

Batch-Channel Normalization performs batch normalization followed
by a channel normalization (similar to a [Group Normalization](../group_norm/index.html).
When the batch size is small a running mean and variance is used for
batch normalization.

Here is [the training code](../weight_standardization/experiment.html) for training
a VGG network that uses weight standardization to classify CIFAR-10 data.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/normalization/weight_standardization/experiment.ipynb)
[![View Run](https://img.shields.io/badge/labml-experiment-brightgreen)](https://app.labml.ai/run/f4a783a2a7df11eb921d0242ac1c0002)
[![WandB](https://img.shields.io/badge/wandb-run-yellow)](https://wandb.ai/vpj/cifar10/runs/3flr4k8w)
"""

import torch
from torch import nn

from labml_helpers.module import Module
from labml_nn.normalization.batch_norm import BatchNorm


class BatchChannelNorm(Module):
    """
    ## Batch-Channel Normalization

    This first performs a batch normalization - either [normal batch norm](../batch_norm/index.html)
    or a batch norm with
    estimated mean and variance (exponential mean/variance over multiple batches).
    Then a channel normalization performed.
    """

    def __init__(self, channels: int, groups: int,
                 eps: float = 1e-5, momentum: float = 0.1, estimate: bool = True):
        """
        * `channels` is the number of features in the input
        * `groups` is the number of groups the features are divided into
        * `eps` is $\epsilon$, used in $\sqrt{Var[x^{(k)}] + \epsilon}$ for numerical stability
        * `momentum` is the momentum in taking the exponential moving average
        * `estimate` is whether to use running mean and variance for batch norm
        """
        super().__init__()

        # Use estimated batch norm or normal batch norm.
        if estimate:
            self.batch_norm = EstimatedBatchNorm(channels,
                                                 eps=eps, momentum=momentum)
        else:
            self.batch_norm = BatchNorm(channels,
                                        eps=eps, momentum=momentum)

        # Channel normalization
        self.channel_norm = ChannelNorm(channels, groups, eps)

    def __call__(self, x):
        x = self.batch_norm(x)
        return self.channel_norm(x)


class EstimatedBatchNorm(Module):
    """
    ## Estimated Batch Normalization

    When input $X \in \mathbb{R}^{B \times C \times H \times W}$ is a batch of image representations,
    where $B$ is the batch size, $C$ is the number of channels, $H$ is the height and $W$ is the width.
    $\gamma \in \mathbb{R}^{C}$ and $\beta \in \mathbb{R}^{C}$.

    $$\dot{X}_{\cdot, C, \cdot, \cdot} = \gamma_C
    \frac{X_{\cdot, C, \cdot, \cdot} - \hat{\mu}_C}{\hat{\sigma}_C}
    + \beta_C$$

    where,

    \begin{align}
    \hat{\mu}_C &\longleftarrow (1 - r)\hat{\mu}_C + r \frac{1}{B H W} \sum_{b,h,w} X_{b,c,h,w} \\
    \hat{\sigma}^2_C &\longleftarrow (1 - r)\hat{\sigma}^2_C + r \frac{1}{B H W} \sum_{b,h,w} \big(X_{b,c,h,w} - \hat{\mu}_C \big)^2
    \end{align}

    are the running mean and variances. $r$ is the momentum for calculating the exponential mean.
    """
    def __init__(self, channels: int,
                 eps: float = 1e-5, momentum: float = 0.1, affine: bool = True):
        """
        * `channels` is the number of features in the input
        * `eps` is $\epsilon$, used in $\sqrt{Var[x^{(k)}] + \epsilon}$ for numerical stability
        * `momentum` is the momentum in taking the exponential moving average
        * `estimate` is whether to use running mean and variance for batch norm
        """
        super().__init__()

        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.channels = channels

        # Channel wise transformation parameters
        if self.affine:
            self.scale = nn.Parameter(torch.ones(channels))
            self.shift = nn.Parameter(torch.zeros(channels))

        # Tensors for $\hat{\mu}_C$ and $\hat{\sigma}^2_C$
        self.register_buffer('exp_mean', torch.zeros(channels))
        self.register_buffer('exp_var', torch.ones(channels))

    def __call__(self, x: torch.Tensor):
        """
        `x` is a tensor of shape `[batch_size, channels, *]`.
        `*` denotes any number of (possibly 0) dimensions.
         For example, in an image (2D) convolution this will be
        `[batch_size, channels, height, width]`
        """
        # Keep old shape
        x_shape = x.shape
        # Get the batch size
        batch_size = x_shape[0]

        # Sanity check to make sure the number of features is correct
        assert self.channels == x.shape[1]

        # Reshape into `[batch_size, channels, n]`
        x = x.view(batch_size, self.channels, -1)

        # Update $\hat{\mu}_C$ and $\hat{\sigma}^2_C$ in training mode only
        if self.training:
            # No backpropagation through $\hat{\mu}_C$ and $\hat{\sigma}^2_C$
            with torch.no_grad():
                # Calculate the mean across first and last dimensions;
                # $$\frac{1}{B H W} \sum_{b,h,w} X_{b,c,h,w}$$
                mean = x.mean(dim=[0, 2])
                # Calculate the squared mean across first and last dimensions;
                # $$\frac{1}{B H W} \sum_{b,h,w} X^2_{b,c,h,w}$$
                mean_x2 = (x ** 2).mean(dim=[0, 2])
                # Variance for each feature
                # $$\frac{1}{B H W} \sum_{b,h,w} \big(X_{b,c,h,w} - \hat{\mu}_C \big)^2$$
                var = mean_x2 - mean ** 2

                # Update exponential moving averages
                # \begin{align}
                # \hat{\mu}_C &\longleftarrow (1 - r)\hat{\mu}_C + r \frac{1}{B H W} \sum_{b,h,w} X_{b,c,h,w} \\
                # \hat{\sigma}^2_C &\longleftarrow (1 - r)\hat{\sigma}^2_C + r \frac{1}{B H W} \sum_{b,h,w} \big(X_{b,c,h,w} - \hat{\mu}_C \big)^2
                # \end{align}
                self.exp_mean = (1 - self.momentum) * self.exp_mean + self.momentum * mean
                self.exp_var = (1 - self.momentum) * self.exp_var + self.momentum * var

        # Normalize
        # $$\frac{X_{\cdot, C, \cdot, \cdot} - \hat{\mu}_C}{\hat{\sigma}_C}$$
        x_norm = (x - self.exp_mean.view(1, -1, 1)) / torch.sqrt(self.exp_var + self.eps).view(1, -1, 1)
        # Scale and shift
        # $$ \gamma_C
        #     \frac{X_{\cdot, C, \cdot, \cdot} - \hat{\mu}_C}{\hat{\sigma}_C}
        #     + \beta_C$$
        if self.affine:
            x_norm = self.scale.view(1, -1, 1) * x_norm + self.shift.view(1, -1, 1)

        # Reshape to original and return
        return x_norm.view(x_shape)


class ChannelNorm(Module):
    """
    ## Channel Normalization

    This is similar to [Group Normalization](../group_norm/index.html) but affine transform is done group wise.
    """

    def __init__(self, channels, groups,
                 eps: float = 1e-5, affine: bool = True):
        """
        * `groups` is the number of groups the features are divided into
        * `channels` is the number of features in the input
        * `eps` is $\epsilon$, used in $\sqrt{Var[x^{(k)}] + \epsilon}$ for numerical stability
        * `affine` is whether to scale and shift the normalized value
        """
        super().__init__()
        self.channels = channels
        self.groups = groups
        self.eps = eps
        self.affine = affine
        # Parameters for affine transformation.
        #
        # *Note that these transforms are per group, unlike in group norm where
        # they are transformed channel-wise.*
        if self.affine:
            self.scale = nn.Parameter(torch.ones(groups))
            self.shift = nn.Parameter(torch.zeros(groups))

    def __call__(self, x: torch.Tensor):
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

        # Reshape into `[batch_size, groups, n]`
        x = x.view(batch_size, self.groups, -1)

        # Calculate the mean across last dimension;
        # i.e. the means for each sample and channel group $\mathbb{E}[x_{(i_N, i_G)}]$
        mean = x.mean(dim=[-1], keepdim=True)
        # Calculate the squared mean across last dimension;
        # i.e. the means for each sample and channel group $\mathbb{E}[x^2_{(i_N, i_G)}]$
        mean_x2 = (x ** 2).mean(dim=[-1], keepdim=True)
        # Variance for each sample and feature group
        # $Var[x_{(i_N, i_G)}] = \mathbb{E}[x^2_{(i_N, i_G)}] - \mathbb{E}[x_{(i_N, i_G)}]^2$
        var = mean_x2 - mean ** 2

        # Normalize
        # $$\hat{x}_{(i_N, i_G)} =
        # \frac{x_{(i_N, i_G)} - \mathbb{E}[x_{(i_N, i_G)}]}{\sqrt{Var[x_{(i_N, i_G)}] + \epsilon}}$$
        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        # Scale and shift group-wise
        # $$y_{i_G} =\gamma_{i_G} \hat{x}_{i_G} + \beta_{i_G}$$
        if self.affine:
            x_norm = self.scale.view(1, -1, 1) * x_norm + self.shift.view(1, -1, 1)

        # Reshape to original and return
        return x_norm.view(x_shape)
