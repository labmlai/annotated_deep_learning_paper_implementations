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
from typing import Union, List

import torch
from torch import nn, Size

from labml_helpers.module import Module


class LayerNorm(Module):
    """
    ## Layer Normalization
    """

    def __init__(self, normalized_shape: Union[int, List[int], Size], *,
                 eps: float = 1e-5,
                 elementwise_affine: bool = True):
        """
        * `normalized_shape` $S$ is shape of the elements (except the batch).
         The input should then be
         $X \in \mathbb{R}^{* \times S[0] \times S[1] \times ... \times S[n]}$
        * `eps` is $\epsilon$, used in $\sqrt{Var[X}] + \epsilon}$ for numerical stability
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
        # Keep the original shape
        x_shape = x.shape
        # Sanity check to make sure the shapes match
        assert self.normalized_shape == x.shape[-len(self.normalized_shape):]

        # Reshape into `[M, S[0], S[1], ..., S[n]]`
        x = x.view(-1, *self.normalized_shape)

        # Calculate the mean across first dimension;
        # i.e. the means for each element $\mathbb{E}[X}]$
        mean = x.mean(dim=0)
        # Calculate the squared mean across first dimension;
        # i.e. the means for each element $\mathbb{E}[X^2]$
        mean_x2 = (x ** 2).mean(dim=0)
        # Variance for each element $Var[X] = \mathbb{E}[X^2] - \mathbb{E}[X]^2$
        var = mean_x2 - mean ** 2

        # Normalize $$\hat{X} = \frac{X} - \mathbb{E}[X]}{\sqrt{Var[X] + \epsilon}}$$
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        # Scale and shift $$\text{LN}(x) = \gamma \hat{X} + \beta$$
        if self.elementwise_affine:
            x_norm = self.gain * x_norm + self.bias

        # Reshape to original and return
        return x_norm.view(x_shape)


def _test():
    from labml.logger import inspect

    x = torch.zeros([2, 3, 2, 4])
    inspect(x.shape)
    ln = LayerNorm(x.shape[2:])

    x = ln(x)
    inspect(x.shape)
    inspect(ln.gain.shape)


if __name__ == '__main__':
    _test()
