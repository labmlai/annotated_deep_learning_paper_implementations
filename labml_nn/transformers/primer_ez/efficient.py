import math

import torch
from torch import nn

from labml_helpers.module import Module
from labml_nn.transformers import MultiHeadAttention


class SpatialDepthWiseConvolution(Module):
    """
    ## Spatial Depth Wise Convolution

    This is actually slower
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
        rng = 1 / math.sqrt(kernel_size)
        self.kernels = nn.Parameter(torch.zeros((kernel_size, d_k)).uniform_(-rng, rng))

    def forward(self, x: torch.Tensor):
        """
        `x` has shape `[seq_len, batch_size, heads, d_k]`
        """

        res = x * self.kernels[0].view(1, 1, 1, -1)

        for i in range(1, len(self.kernels)):
            res[i:] += x[:-i] * self.kernels[i].view(1, 1, 1, -1)

        return res


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
        self.query = nn.Sequential(self.query, SpatialDepthWiseConvolution(self.d_k))
        self.key = nn.Sequential(self.key, SpatialDepthWiseConvolution(self.d_k))
        self.value = nn.Sequential(self.value, SpatialDepthWiseConvolution(self.d_k))
