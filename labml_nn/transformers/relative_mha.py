"""
Implementation of "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context"
https://arxiv.org/abs/1901.02860
"""

import torch
from torch import nn

from labml_helpers.module import Module
from labml.logger import inspect
from transformers.mha import MultiHeadAttention


def relative_shift(x: torch.Tensor):
    zero_pad = x.new_zeros(x.shape[0], 1, *x.shape[2:])
    x_padded = torch.cat([x, zero_pad], dim=1)

    x_padded = x_padded.view(x.shape[1] + 1, x.shape[0], *x.shape[2:])

    x = x_padded[:-1].view_as(x)

    return x


class RelativeMultiHeadAttention(MultiHeadAttention):
    def __init__(self, heads: int, d_model: int, dropout_prob: float = 0.1):
        super().__init__(heads, d_model, dropout_prob, False)
        self.P = 2 ** 12

        self.key_pos_embeddings = nn.Parameter(torch.zeros((self.P * 2, heads, self.d_k)), requires_grad=True)
        self.query_pos_bias = nn.Parameter(torch.zeros((heads, self.d_k)), requires_grad=True)
        self.key_pos_bias = nn.Parameter(torch.zeros((self.P * 2, heads)), requires_grad=True)

    def get_scores(self, query: torch.Tensor, key: torch.Tensor):
        key_pos_emb = self.key_pos_embeddings[self.P - query.shape[0]:self.P + key.shape[0]]
        key_pos_bias = self.key_pos_bias[self.P - query.shape[0]:self.P + key.shape[0]]

        ac = torch.einsum('ibhd,jbhd->ijbh', query + self.query_pos_bias[None, None, :, :], key)
        b = torch.einsum('ibhd,jhd->ijbh', query, key_pos_emb)
        d = key_pos_bias[None, :, None, :]
        bd = relative_shift(b + d)
        bd = bd[:, -key.shape[0]:]

        return ac + bd


def _test_relative_shift():
    x = torch.arange(1, 6)[None, :, None, None].repeat(5, 1, 1, 1)
    inspect(x[:, :, 0, 0])
    inspect(relative_shift(x)[:, :, 0, 0])

    x = torch.arange(1, 6)[None, :, None, None].repeat(3, 1, 1, 1)
    inspect(x[:, :, 0, 0])
    inspect(relative_shift(x)[:, :, 0, 0])


if __name__ == '__main__':
    _test_relative_shift()
