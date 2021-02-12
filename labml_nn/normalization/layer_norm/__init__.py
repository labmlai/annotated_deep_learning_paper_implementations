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
* Need to compute means and variances across devices in distributed training.

## Layer Normalization

Layer normalization is a simpler normalization method that works
on a wider range of settings.
Layer normalization transforms the inputs to have zero mean and unit variance
across the features.
*Note that batch normalization fixes the zero mean and unit variance for each element.*
Layer normalization does it for each batch across all elements.

Layer normalization is generally used for NLP tasks.

We have used layer normalization in most of the
[transformer implementations](../../transformers/gpt/index.html).
"""
from typing import Union, List

import torch
from torch import nn, Size

from labml_helpers.module import Module


class LayerNorm(Module):
    r"""
    ## Layer Normalization

    Layer normalization $\text{LN}$ normalizes the input $X$ as follows:

    When input $X \in \mathbb{R}^{B \times C}$ is a batch of embeddings,
    where $B$ is the batch size and $C$ is the number of features.
    $\gamma \in \mathbb{R}^{C}$ and $\beta \in \mathbb{R}^{C}$.
    $$\text{LN}(X) = \gamma
    \frac{X - \underset{C}{\mathbb{E}}[X]}{\sqrt{\underset{C}{Var}[X] + \epsilon}}
    + \beta$$

    When input $X \in \mathbb{R}^{L \times B \times C}$ is a batch of a sequence of embeddings,
    where $B$ is the batch size, $C$ is the number of channels, $L$ is the length of the sequence.
    $\gamma \in \mathbb{R}^{C}$ and $\beta \in \mathbb{R}^{C}$.
    $$\text{LN}(X) = \gamma
    \frac{X - \underset{C}{\mathbb{E}}[X]}{\sqrt{\underset{C}{Var}[X] + \epsilon}}
    + \beta$$

    When input $X \in \mathbb{R}^{B \times C \times H \times W}$ is a batch of image representations,
    where $B$ is the batch size, $C$ is the number of channels, $H$ is the height and $W$ is the width.
    This is not a widely used scenario.
    $\gamma \in \mathbb{R}^{C \times H \times W}$ and $\beta \in \mathbb{R}^{C \times H \times W}$.
    $$\text{LN}(X) = \gamma
    \frac{X - \underset{C, H, W}{\mathbb{E}}[X]}{\sqrt{\underset{C, H, W}{Var}[X] + \epsilon}}
    + \beta$$
    """

    def __init__(self, normalized_shape: Union[int, List[int], Size], *,
                 eps: float = 1e-5,
                 elementwise_affine: bool = True):
        """
        * `normalized_shape` $S$ is the shape of the elements (except the batch).
         The input should then be
         $X \in \mathbb{R}^{* \times S[0] \times S[1] \times ... \times S[n]}$
        * `eps` is $\epsilon$, used in $\sqrt{Var[X] + \epsilon}$ for numerical stability
        * `elementwise_affine` is whether to scale and shift the normalized value

        We've tried to use the same names for arguments as PyTorch `LayerNorm` implementation.
        """
        super().__init__()

        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        # Create parameters for $\gamma$ and $\beta$ for gain and bias
        if self.elementwise_affine:
            self.gain = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x: torch.Tensor):
        """
        `x` is a tensor of shape `[*, S[0], S[1], ..., S[n]]`.
        `*` could be any number of dimensions.
         For example, in an NLP task this will be
        `[seq_len, batch_size, features]`
        """
        # Sanity check to make sure the shapes match
        assert self.normalized_shape == x.shape[-len(self.normalized_shape):]

        # The dimensions to calculate the mean and variance on
        dims = [-(i + 1) for i in range(len(self.normalized_shape))]

        # Calculate the mean of all elements;
        # i.e. the means for each element $\mathbb{E}[X]$
        mean = x.mean(dim=dims, keepdims=True)
        # Calculate the squared mean of all elements;
        # i.e. the means for each element $\mathbb{E}[X^2]$
        mean_x2 = (x ** 2).mean(dim=dims, keepdims=True)
        # Variance of all element $Var[X] = \mathbb{E}[X^2] - \mathbb{E}[X]^2$
        var = mean_x2 - mean ** 2

        # Normalize $$\hat{X} = \frac{X - \mathbb{E}[X]}{\sqrt{Var[X] + \epsilon}}$$
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        # Scale and shift $$\text{LN}(x) = \gamma \hat{X} + \beta$$
        if self.elementwise_affine:
            x_norm = self.gain * x_norm + self.bias

        #
        return x_norm


def _test():
    """
    Simple test
    """
    from labml.logger import inspect

    x = torch.zeros([2, 3, 2, 4])
    inspect(x.shape)
    ln = LayerNorm(x.shape[2:])

    x = ln(x)
    inspect(x.shape)
    inspect(ln.gain.shape)


#
if __name__ == '__main__':
    _test()
