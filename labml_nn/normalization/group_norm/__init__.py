"""
---
title: Group Normalization
summary: >
 A PyTorch implementation/tutorial of group normalization.
---

# Group Normalization

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

        # Reshape into `[batch_size, channels, n]`
        x = x.view(batch_size, self.groups, self.channels // self.groups, -1)

        # Calculate the mean across first and last dimension;
        # i.e. the means for each feature $\mathbb{E}[x^{(k)}]$
        mean = x.mean(dim=[2, 3], keepdims=True)
        # Calculate the squared mean across first and last dimension;
        # i.e. the means for each feature $\mathbb{E}[(x^{(k)})^2]$
        mean_x2 = (x ** 2).mean(dim=[2, 3], keepdims=True)
        # Variance for each feature $Var[x^{(k)}] = \mathbb{E}[(x^{(k)})^2] - \mathbb{E}[x^{(k)}]^2$
        var = mean_x2 - mean ** 2

        # Normalize $$\hat{x}^{(k)} = \frac{x^{(k)} - \mathbb{E}[x^{(k)}]}{\sqrt{Var[x^{(k)}] + \epsilon}}$$
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        x_norm = x_norm.view(batch_size, self.channels, -1)

        # Scale and shift $$y^{(k)} =\gamma^{(k)} \hat{x}^{(k)} + \beta^{(k)}$$
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
    bn = GroupNorm(2, 6)

    x = bn(x)
    inspect(x.shape)


#
if __name__ == '__main__':
    _test()
