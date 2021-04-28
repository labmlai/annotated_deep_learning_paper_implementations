"""
---
title: Group Normalization
summary: >
 A PyTorch implementation/tutorial of group normalization.
---

# Group Normalization

This is a [PyTorch](https://pytorch.org) implementation of
the [Group Normalization](https://arxiv.org/abs/1803.08494) paper.

[Batch Normalization](../batch_norm/index.html) works well for large enough batch sizes
but not well for small batch sizes, because it normalizes over the batch.
Training large models with large batch sizes is not possible due to the memory capacity of the
devices.

This paper introduces Group Normalization, which normalizes a set of features together as a group.
This is based on the observation that classical features such as
[SIFT](https://en.wikipedia.org/wiki/Scale-invariant_feature_transform) and
[HOG](https://en.wikipedia.org/wiki/Histogram_of_oriented_gradients) are group-wise features.
The paper proposes dividing feature channels into groups and then separately normalizing
all channels within each group.

## Formulation

All normalization layers can be defined by the following computation.

$$\hat{x}_i = \frac{1}{\sigma_i} (x_i - \mu_i)$$

where $x$ is the tensor representing the batch,
and $i$ is the index of a single value.
For instance, when it's 2D images
$i = (i_N, i_C, i_H, i_W)$ is a 4-d vector for indexing
image within batch, feature channel, vertical coordinate and horizontal coordinate.
$\mu_i$ and $\sigma_i$ are mean and standard deviation.

\begin{align}
\mu_i &= \frac{1}{m} \sum_{k \in \mathcal{S}_i} x_k \\
\sigma_i  &= \sqrt{\frac{1}{m} \sum_{k \in \mathcal{S}_i} (x_k - \mu_i)^2 + \epsilon}
\end{align}

$\mathcal{S}_i$ is the set of indexes across which the mean and standard deviation
are calculated for index $i$.
$m$ is the size of the set $\mathcal{S}_i$ which is the same for all $i$.

The definition of $\mathcal{S}_i$ is different for
[Batch normalization](../batch_norm/index.html),
[Layer normalization](../layer_norm/index.html), and
[Instance normalization](../instance_norm/index.html).

### [Batch Normalization](../batch_norm/index.html)

$$\mathcal{S}_i = \{k | k_C = i_C\}$$

The values that share the same feature channel are normalized together.

### [Layer Normalization](../layer_norm/index.html)

$$\mathcal{S}_i = \{k | k_N = i_N\}$$

The values from the same sample in the batch are normalized together.

### [Instance Normalization](../instance_norm/index.html)

$$\mathcal{S}_i = \{k | k_N = i_N, k_C = i_C\}$$

The values from the same sample and same feature channel are normalized together.

### Group Normalization

$$\mathcal{S}_i = \{k | k_N = i_N,
 \bigg \lfloor \frac{k_C}{C/G} \bigg \rfloor = \bigg \lfloor \frac{i_C}{C/G} \bigg \rfloor\}$$

where $G$ is the number of groups and $C$ is the number of channels.

Group normalization normalizes values of the same sample and the same group of channels together.

Here's a [CIFAR 10 classification model](experiment.html) that uses instance normalization.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lab-ml/nn/blob/master/labml_nn/normalization/group_norm/experiment.ipynb)
[![View Run](https://img.shields.io/badge/labml-experiment-brightgreen)](https://app.labml.ai/run/081d950aa4e011eb8f9f0242ac1c0002)
[![WandB](https://img.shields.io/badge/wandb-run-yellow)](https://wandb.ai/vpj/cifar10/runs/310etthp)
"""

import torch
from torch import nn

from labml_helpers.module import Module


class GroupNorm(Module):
    r"""
    ## Group Normalization Layer
    """

    def __init__(self, groups: int, channels: int, *,
                 eps: float = 1e-5, affine: bool = True):
        """
        * `groups` is the number of groups the features are divided into
        * `channels` is the number of features in the input
        * `eps` is $\epsilon$, used in $\sqrt{Var[x^{(k)}] + \epsilon}$ for numerical stability
        * `affine` is whether to scale and shift the normalized value
        """
        super().__init__()

        assert channels % groups == 0, "Number of channels should be evenly divisible by the number of groups"
        self.groups = groups
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

        # Scale and shift channel-wise
        # $$y_{i_C} =\gamma_{i_C} \hat{x}_{i_C} + \beta_{i_C}$$
        if self.affine:
            x_norm = x_norm.view(batch_size, self.channels, -1)
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
    bn = GroupNorm(2, 6)

    x = bn(x)
    inspect(x.shape)


#
if __name__ == '__main__':
    _test()
