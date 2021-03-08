"""
---
title: Fast Weight Systems
summary: >
  This is an annotated implementation/tutorial of
  Linear Transformers Are Secretly Fast Weight Memory Systems in PyTorch.
---
"""

import torch
from torch import nn

from labml_helpers.module import Module
from labml_nn.transformers.feed_forward import FeedForward
from labml_nn.transformers.mha import PrepareForMultiHeadAttention
from labml_nn.utils import clone_module_list


class DPFP(Module):
    def __init__(self, nu: int = 1, eps: float = 1e-6):
        super().__init__()
        self.nu = nu
        self.r = nn.ReLU()
        self.eps = eps

    def __call__(self, x: torch.Tensor):
        x = self.dpfp(x)
        return x / (torch.sum(x, dim=-1, keepdim=True) + self.eps)

    def dpfp(self, x: torch.Tensor):
        x = torch.cat([self.r(x), self.r(-x)], dim=-1)
        x_rolled = torch.cat([x.roll(shifts=j, dims=-1) for j in range(1, self.nu + 1)], dim=-1)
        x_repeat = torch.cat([x] * self.nu, dim=-1)

        return x_repeat * x_rolled


class FastWeightAttention(Module):
    def __init__(self, heads: int, d_model: int, dropout_prob: float, sigma: DPFP):
        super().__init__()

        # Number of features per head
        self.d_k = d_model // heads
        #
        self.heads = heads

        # These transform the `query` multi-headed attention.
        self.query = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=False)
        # These transform the `key` and `value` for multi-headed attention.
        self.key = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=False)
        self.value = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=False)

        self.gate = nn.Sequential(PrepareForMultiHeadAttention(d_model, heads, 1, bias=False),
                                  nn.Sigmoid())

        self.sigma = sigma

        # Output layer
        self.output = nn.Linear(d_model, d_model)
        # Dropout
        self.dropout = nn.Dropout(dropout_prob)

    def __call__(self, x: torch.Tensor):
        seq_len = x.shape[0]
        query = self.sigma(self.query(x))
        key = self.sigma(self.key(x))
        value = self.value(x)
        beta = self.gate(x)

        weights = key.new_zeros((key.shape[1], key.shape[2], value.shape[3], key.shape[3]))
        outputs = []

        for i in range(seq_len):
            value_existing = torch.einsum('bhvk,bhk->bhv', weights, key[i])

            weights = weights + torch.einsum('bhv,bhk->bhvk', beta[i] * (value[i] - value_existing), key[i])

            x = torch.einsum('bhvk,bhk->bhv', weights, query[i])

            # Concatenate multiple heads
            outputs.append(x.reshape(x.shape[0], -1))

        x = torch.stack(outputs)
        # Output layer
        return self.output(x)


class FastWeightAttentionTransformerLayer(Module):
    def __init__(self, *,
                 d_model: int,
                 attn: FastWeightAttention,
                 feed_forward: FeedForward,
                 dropout_prob: float):
        super().__init__()
        # Transformer size $d_{model}$
        self.size = d_model
        #
        self.attn = attn
        self.feed_forward = feed_forward
        self.dropout = nn.Dropout(dropout_prob)

        # Normalization layers
        self.norm_self_attn = nn.LayerNorm([d_model])
        self.norm_ff = nn.LayerNorm([d_model])

    def __call__(self, x: torch.Tensor):
        attn = self.attn(x)
        # Add the self attention results
        x = x + self.dropout(attn)

        # Normalize for feed-forward
        z = self.norm_ff(x)
        # Pass through the feed-forward network
        ff = self.feed_forward(z)
        # Add the feed-forward results back
        x = x + self.dropout(ff)

        #
        return x


class FastWeightAttentionTransformer(Module):
    def __init__(self, layer: FastWeightAttentionTransformerLayer, n_layers: int):
        super().__init__()
        # Make copies of the transformer layer
        self.layers = clone_module_list(layer, n_layers)
        # Final normalization layer
        self.norm = nn.LayerNorm([layer.size])

    def __call__(self, x: torch.Tensor):
        for i, layer in enumerate(self.layers):
            # Get layer output
            x = layer(x)

        # Normalize the output
        return self.norm(x)
