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


class LinearAttentionFunction(Module):
    def __init__(self):
        super().__init__()

    def __call__(self, x: torch.Tensor):
        return x


class FastWeightAttention(Module):
    def __init__(self, heads: int, d_model: int, dropout_prob: float = 0.1):
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

        self.sigma = LinearAttentionFunction()

        # Output layer
        self.output = nn.Linear(d_model, d_model)
        # Dropout
        self.dropout = nn.Dropout(dropout_prob)

    def __call__(self, x: torch.Tensor, weights: torch.Tensor):
        query = self.sigma(self.query(x))
        key = self.sigma(self.key(x))
        value = self.value(x)

        value_existing = torch.einsum('bhvk,bhk->bhv', weights, key)

        beta = self.gate(x)

        weights = weights + torch.einsum('bhv,bhk->bhvk', beta * (value - value_existing), key)

        x = torch.einsum('bhvk,bhk->bhv', weights, query)

        # Concatenate multiple heads
        x = x.reshape(x.shape[0], -1)

        # Output layer
        return self.output(x), weights


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

    def __call__(self, x: torch.Tensor, weights: torch.Tensor):
        attn, weights = self.attn(x, weights)
        # Add the self attention results
        x = x + self.dropout(attn)

        # Normalize for feed-forward
        z = self.norm_ff(x)
        # Pass through the feed-forward network
        ff = self.feed_forward(z)
        # Add the feed-forward results back
        x = x + self.dropout(ff)

        #
        return x, weights


class FastWeightAttentionTransformer(Module):
    def __init__(self, layer: FastWeightAttentionTransformerLayer, n_layers: int):
        super().__init__()
        # Make copies of the transformer layer
        self.layers = clone_module_list(layer, n_layers)
        # Final normalization layer
        self.norm = nn.LayerNorm([layer.size])

    def __call__(self, x_seq: torch.Tensor):
        # Split the input to a list along the sequence axis
        x_seq = torch.unbind(x_seq, dim=0)
        # List to store the outputs
        res = []
        # For each input step
        weights = [torch.zeros() for _ in range(len(self.layers))]

        for x in x_seq:
            # Run through each layer
            for i, layer in enumerate(self.layers):
                # Get layer output
                x = layer(x, weights[i])

            res.append(x)

        # Stack the output tensors
        res = torch.stack(res)
        # Normalize the output
        return self.norm(res)
