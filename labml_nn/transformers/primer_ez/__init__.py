"""
---
title: "Primer: Searching for Efficient Transformers for Language Modeling"
summary: >
  This is an annotated implementation/tutorial of
  Primer: Searching for Efficient Transformers for Language Modeling for Vision in PyTorch.
---

# Primer: Searching for Efficient Transformers for Language Modeling

This is a [PyTorch](https://pytorch.org) implementation of the paper
[Primer: Searching for Efficient Transformers for Language Modeling](https://papers.labml.ai/paper/2109.08668).

The authors do an evolutionary search for transformer architectures.
They name the architecture found using the search Primer (PRIMitives searched transformER).
**Primer EZ** is the architecture with the two most robust modification in Primer compared to original transformer.
Primer EZ trains a lot faster than vanilla transformer.

### Squared ReLU

The most effective modification found by the search is using a square ReLU instead of ReLU in
the [position wise feedforward module](../feed_forward.html).

$$y = {\max(x, 0)}^2$$

### Multi-DConv-Head Attention (MDHA)

The next effective modification is a $3 \times 1$ convolution after multi-head projection for queries, keys and values.
They do a 1-dimensional convolution of kernel size $3$ along the sequence dimension.
The convolution is done per-channel (depth wise).

[Here is the experiment code](experiment.html), for Primer EZ.

[![View Run](https://img.shields.io/badge/labml-experiment-brightgreen)](https://app.labml.ai/run/30adb7aa1ab211eca7310f80a114e8a4)
"""

import torch
from torch import nn

from labml_helpers.module import Module
from labml_nn.transformers import MultiHeadAttention


class SquaredReLU(Module):
    """
    ## Squared ReLU activation

    $$y = {\max(x, 0)}^2$$

    Squared ReLU is used as the activation function in the
     [position wise feedforward module](../feed_forward.html).
    """

    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor):
        # Apply ReLU
        x = self.relu(x)
        # Square it
        return x * x


class SpatialDepthWiseConvolution(Module):
    """
    ## Spatial Depth Wise Convolution
    """

    def __init__(self, kernel_size: int = 3):
        """
        * `d_k` is the number of channels in each head
        """
        super().__init__()
        self.kernel_size = kernel_size
        # We use PyTorch's `Conv1d` module.
        self.conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=(kernel_size,), padding=(kernel_size - 1,))

    def forward(self, x: torch.Tensor):
        """
        `x` has shape `[seq_len, batch_size, heads, d_k]`
        """

        seq_len, batch_size, heads, d_k = x.shape
        x = x.permute(1, 2, 3, 0)
        x = x.view(batch_size * heads * d_k, 1, seq_len)

        x = self.conv(x)
        x = x[:, :, :-(self.kernel_size - 1)]
        x = x.view(batch_size, heads, d_k, seq_len)
        x = x.permute(3, 0, 1, 2)

        return x


class MultiDConvHeadAttention(MultiHeadAttention):
    def __init__(self, heads: int, d_model: int, dropout_prob: float = 0.1):
        super().__init__(heads, d_model, dropout_prob)

        self.query = nn.Sequential(self.query, SpatialDepthWiseConvolution())
        self.key = nn.Sequential(self.key, SpatialDepthWiseConvolution())
        self.value = nn.Sequential(self.value, SpatialDepthWiseConvolution())
