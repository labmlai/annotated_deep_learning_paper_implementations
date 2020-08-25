import copy

import torch
from torch import nn

from labml.helpers.pytorch.module import Module
from transformers.mha import MultiHeadAttention


class PrepareForMultiHeadAttention(Module):
    def __init__(self, d_model: int, heads: int, d_k: int):
        super().__init__()
        self.linear = nn.Linear(d_model, heads * d_k, bias=False)
        self.heads = heads
        self.d_k = d_k

    def __call__(self, x: torch.Tensor):
        seq_len, batch_size, _ = x.shape

        x = self.linear(x)
        x = x.view(seq_len, batch_size, self.heads, self.d_k)

        return x


class RelativeMultiHeadAttention(MultiHeadAttention):
    @staticmethod
    def _rel_shift(x: torch.Tensor):
        zero_pad = torch.zeros((x.shape[0], 1, *x.shape[2:]),
                               device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=1)

        x_padded = x_padded.view(x.shape[1] + 1, x.shape[0], *x.shape[2:])

        x = x_padded[1:].view_as(x)

        ones = torch.ones((x.size(0), x.size(1)), device=x.device)
        lower_triangle = torch.tril(ones, x.size(1) - x.size(0))
        x = x * lower_triangle[:, :, None, None]

        return x

    def __init__(self, heads: int, d_model: int, dropout_prob: float = 0.1):
        super().__init__(heads, d_model, dropout_prob)
        self.max_key_len = 2 ** 12

        self.key_pos_embeddings = nn.Parameter(
            torch.zeros((self.max_key_len, heads, self.d_k)),
            requires_grad=True)
        self.query_pos_bias = nn.Parameter(
            torch.zeros((heads, self.d_k)),
            requires_grad=True)
        self.key_pos_bias = nn.Parameter(
            torch.zeros((self.max_key_len, heads)),
            requires_grad=True)

    def get_scores(self, query: torch.Tensor,
                   key: torch.Tensor, ):
        key_len = key.shape[0]

        ac = torch.einsum('ibhd,jbhd->ijbh', query + self.query_pos_bias[None, None, :, :], key)
        b = torch.einsum('ibhd,jhd->ijbh', query, self.key_pos_embeddings[-key_len:])
        d = self.key_pos_bias[None, -key_len:, None, :]
        bd = self._rel_shift(b + d)

        return ac + bd
