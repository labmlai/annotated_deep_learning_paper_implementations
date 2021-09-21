import torch
from torch import nn

from labml_helpers.module import Module
from labml_nn.transformers import MultiHeadAttention


class SquaredReLU(Module):
    def __init__(self):
        super().__init__()

        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor):
        x = self.relu(x)

        return x * x


class SpatialDepthWiseConvolution(Module):
    def __init__(self, d_k: int, kernel_size: int = 3):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(in_channels=d_k, out_channels=d_k,
                              kernel_size=(kernel_size,), padding=(kernel_size - 1,), groups=d_k)

    def forward(self, x: torch.Tensor):
        """
        `x` has shape `[seq_len, batch_size, heads, d_k]`
        """

        seq_len, batch_size, heads, d_k = x.shape
        x = x.permute(1, 2, 3, 0)
        x = x.view(batch_size * heads, d_k, seq_len)

        x = self.conv(x)
        x = x[:, :, :-(self.kernel_size - 1)]
        x = x.view(batch_size, heads, d_k, seq_len)
        x = x.permute(3, 0, 1, 2)

        return x


class MultiDConvHeadAttention(MultiHeadAttention):
    def __init__(self, heads: int, d_model: int, dropout_prob: float = 0.1):
        super().__init__(heads, d_model, dropout_prob)

        self.query = nn.Sequential(self.query, SpatialDepthWiseConvolution(self.d_k))
        self.key = nn.Sequential(self.key, SpatialDepthWiseConvolution(self.d_k))
        self.value = nn.Sequential(self.value, SpatialDepthWiseConvolution(self.d_k))
