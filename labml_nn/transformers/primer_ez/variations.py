"""
---
title: Primer EZ variations
summary: We tried some variations to Primer EZ.
---

# [Primer EZ](index.html) Variations

We tried some variations to see which changes in Primer EZ has most benefits.
"""

import torch
from torch import nn

from labml_helpers.module import Module
from labml_nn.transformers import MultiHeadAttention


class SpatialDepthWiseSharedConvolution(Module):
    """
    ## Spatial Depth Wise Shared Convolution

    We share the same kernel across all channels.
    """

    def __init__(self, kernel_size: int = 3):
        """
        """
        super().__init__()
        self.kernel_size = kernel_size
        # We use PyTorch's `Conv1d` module.
        # We add padding to both sides and later crop the right most `kernel_size - 1` results
        self.conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=(kernel_size,), padding=(kernel_size - 1,))

    def forward(self, x: torch.Tensor):
        """
        `x` has shape `[seq_len, batch_size, heads, d_k]`
        """

        # Get the shape
        seq_len, batch_size, heads, d_k = x.shape
        # Permute to `[batch_size, heads, d_k, seq_len]`
        x = x.permute(1, 2, 3, 0)
        # Change the shape to `[batch_size * heads * d_k, seq_len]`
        x = x.view(batch_size * heads * d_k, 1, seq_len)

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


class MultiDSharedConvHeadAttention(MultiHeadAttention):
    """
    ## Multi-Depth-wise-Shared-Conv-Head Attention

    We extend our original implementation of [Multi-Head Attention](../mha.html#MHA)
    and add the spatial depth-wise shared convolution to query, key and value projections.
    """

    def __init__(self, heads: int, d_model: int, dropout_prob: float = 0.1):
        super().__init__(heads, d_model, dropout_prob)

        # [Multi-Head Attention](../mha.html#MHA) will create query, key and value projection modules
        # `self.query`, `self.key`, and `self.value`.
        #
        # We combine a spatial depth-wise shared convolution layer to each of them and replace
        # `self.query`, `self.key`, and `self.value`.
        self.query = nn.Sequential(self.query, SpatialDepthWiseSharedConvolution())
        self.key = nn.Sequential(self.key, SpatialDepthWiseSharedConvolution())
        self.value = nn.Sequential(self.value, SpatialDepthWiseSharedConvolution())


class SpatialDepthWisePerHeadConvolution(Module):
    """
    ## Spatial Depth Wise Per Head Convolution
    """

    def __init__(self, heads: int, d_k: int, kernel_size: int = 3):
        """
        * `heads` is the number of heads
        * `d_k` is the number of channels in each head
        """
        super().__init__()
        self.kernel_size = kernel_size
        # We use PyTorch's `Conv1d` module.
        # We set the number of groups to be equal to the number of channels from each head
        # so that it does a separate convolution
        # (with different kernels) for each channel and head.
        # We add padding to both sides and later crop the right most `kernel_size - 1` results
        self.conv = nn.Conv1d(in_channels=d_k * heads, out_channels=d_k * heads,
                              kernel_size=(kernel_size,), padding=(kernel_size - 1,), groups=d_k * heads)

    def forward(self, x: torch.Tensor):
        """
        `x` has shape `[seq_len, batch_size, heads, d_k]`
        """

        # Get the shape
        seq_len, batch_size, heads, d_k = x.shape
        # Permute to `[batch_size, heads, d_k, seq_len]`
        x = x.permute(1, 2, 3, 0)
        # Change the shape to `[batch_size heads * d_k, seq_len]`
        x = x.view(batch_size, heads * d_k, seq_len)

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


class MultiDPHConvHeadAttention(MultiHeadAttention):
    """
    ## Multi-per-Head-Depth-wise-Conv-Head Attention

    We extend our original implementation of [Multi-Head Attention](../mha.html#MHA)
    and add the spatial depth-wise convolution to query, key and value projections.
    """

    def __init__(self, heads: int, d_model: int, dropout_prob: float = 0.1):
        super().__init__(heads, d_model, dropout_prob)

        # [Multi-Head Attention](../mha.html#MHA) will create query, key and value projection modules
        # `self.query`, `self.key`, and `self.value`.
        #
        # We combine a spatial per-head depth-wise convolution layer to each of them and replace
        # `self.query`, `self.key`, and `self.value`.
        self.query = nn.Sequential(self.query, SpatialDepthWisePerHeadConvolution(heads, self.d_k))
        self.key = nn.Sequential(self.key, SpatialDepthWisePerHeadConvolution(heads, self.d_k))
        self.value = nn.Sequential(self.value, SpatialDepthWisePerHeadConvolution(heads, self.d_k))
