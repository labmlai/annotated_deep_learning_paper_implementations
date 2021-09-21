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
**Primer EZ** is the architecture with the two most robust modifications in Primer compared to
 the original transformer.
Primer EZ trains a lot faster than the vanilla transformer.

### Squared ReLU

The most effective modification found by the search is using a square ReLU instead of ReLU in
the [position-wise feedforward module](../feed_forward.html).

$$y = {\max(x, 0)}^2$$

### Multi-DConv-Head Attention (MDHA)

The next effective modification is a depth-wise $3 \times 1$ convolution after multi-head projection
 for queries, keys, and values.
The convolution is along the sequence dimension and per channel (depth-wise).
To be clear, if the number of channels in each head is $d_k$ the convolution will have $1 \times 3$
kernels for each of the $d_k$ channels.

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

    def __init__(self, d_k: int, kernel_size: int = 3):
        """
        * `d_k` is the number of channels in each head
        """
        super().__init__()
        self.kernel_size = kernel_size
        # We use PyTorch's `Conv1d` module.
        # We set the number of groups to be equal to the number of channels so that it does a separate convolution
        # (with different kernels) for each channel.
        # We add padding to both sides and later crop the right most `kernel_size - 1` results
        self.conv = nn.Conv1d(in_channels=d_k, out_channels=d_k,
                              kernel_size=(kernel_size,), padding=(kernel_size - 1,), groups=d_k)

    def forward(self, x: torch.Tensor):
        """
        `x` has shape `[seq_len, batch_size, heads, d_k]`
        """

        # Get the shape
        seq_len, batch_size, heads, d_k = x.shape
        # Permute to `[batch_size, heads, d_k, seq_len]`
        x = x.permute(1, 2, 3, 0)
        # Change the shape to `[batch_size * heads, d_k, seq_len]`
        x = x.view(batch_size * heads, d_k, seq_len)

        # 1D convolution accepts input of the form `[N, channels, sequence]`
        x = self.conv(x)
        # Crop the right most `kernel_size - 1` results since we padded both sides
        x = x[:, :, :-(self.kernel_size - 1)]
        # Reshape to `[batch_size, heads, d_k, seq_len]`
        x = x.view(batch_size, heads, d_k, seq_len)
        # Permute to `[seq_len, batch_size, heads, d_k]`
        x = x.permute(3, 0, 1, 2)

        #
        return x


class MultiDConvHeadAttention(MultiHeadAttention):
    """
    ## Multi-DConv-Head Attention (MDHA)

    We extend our original implementation of [Multi-Head Attention](../mha.html#MHA)
    and add the spatial depth-wise convolution to query, key and value projections.
        """

    def __init__(self, heads: int, d_model: int, dropout_prob: float = 0.1):
        super().__init__(heads, d_model, dropout_prob)

        # [Multi-Head Attention](../mha.html#MHA) will create query, key and value projection modules
        # `self.query`, `self.key`, and `self.value`.
        #
        # We combine a spatial depth-wise convolution layer to each of them and replace
        # `self.query`, `self.key`, and `self.value`.
        #
        # 📝 *We feel this cleaner implementation is easier to understand since it clearly shows the difference
        # between this and vanilla transformer multi-head attention*.
        self.query = nn.Sequential(self.query, SpatialDepthWiseConvolution(self.d_k))
        self.key = nn.Sequential(self.key, SpatialDepthWiseConvolution(self.d_k))
        self.value = nn.Sequential(self.value, SpatialDepthWiseConvolution(self.d_k))
