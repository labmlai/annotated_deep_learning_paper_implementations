"""
---
title: Feedback Transformer
summary: >
  This implements the Feedback Transformer in PyTorch with explainations.
---
"""

import math
from typing import Optional

import torch
from torch import nn

from labml_helpers.module import Module
from labml_nn.transformers.mha import PrepareForMultiHeadAttention
from labml_nn.transformers.models import FeedForward
from labml_nn.utils import clone_module_list


class PrepareQueryForMultiHeadAttention(Module):
    """
    ## Prepare query for multi-head attention

    This module does a linear transformation and splits the vector into given
    number of heads for multi-head attention.
    This is used to transform **key**, **query**, and **value** vectors.
    """

    def __init__(self, d_model: int, heads: int, d_k: int, bias: bool):
        super().__init__()
        # Linear layer for linear transform
        self.linear = nn.Linear(d_model, heads * d_k, bias=bias)
        # Number of heads
        self.heads = heads
        # Number of dimensions in vectors in each head
        self.d_k = d_k

    def __call__(self, x: torch.Tensor):
        # Input has shape `[seq_len, batch_size, d_model]`
        batch_size, _ = x.shape

        # Linear transform
        x = self.linear(x)
        # Split into heads
        x = x.view(batch_size, self.heads, self.d_k)

        # Output has shape `[seq_len, batch_size, heads, d_k]`
        return x


class FeedbackAttention(Module):
    def __init__(self, heads: int, d_model: int, dropout_prob: float = 0.1):
        super().__init__()

        self.d_k = d_model // heads
        self.heads = heads

        # These transform the `query`, `key` and `value` vectors for multi-headed attention.
        self.query = PrepareQueryForMultiHeadAttention(d_model, heads, self.d_k, False)
        self.key = PrepareForMultiHeadAttention(d_model, heads, self.d_k, False)
        self.value = PrepareForMultiHeadAttention(d_model, heads, self.d_k, False)

        # Output layer
        self.output = nn.Linear(d_model, d_model)
        # Dropout
        self.dropout = nn.Dropout(dropout_prob)
        # Scaling factor before the softmax
        self.scale = 1 / math.sqrt(self.d_k)

        # We store attentions so that it can used for logging, or other computations if needed
        self.attn = None

        self.P = 2 ** 12

        self.key_pos_embeddings = nn.Parameter(torch.zeros((self.P, heads, self.d_k)), requires_grad=True)
        self.key_pos_bias = nn.Parameter(torch.zeros((self.P, heads)), requires_grad=True)
        self.query_pos_bias = nn.Parameter(torch.zeros((heads, self.d_k)), requires_grad=True)
        self.softmax = nn.Softmax(dim=0)

    def get_scores(self, query: torch.Tensor, key: torch.Tensor):
        key_pos_emb = self.key_pos_embeddings[-key.shape[0]:]
        key_pos_bias = self.key_pos_bias[-key.shape[0]:]
        query_pos_bias = self.query_pos_bias[None, :, :]

        ac = torch.einsum('bhd,jbhd->jbh', query + query_pos_bias, key)
        bd = torch.einsum('bhd,jhd->jbh', query, key_pos_emb) + key_pos_bias[:, None, :]

        return ac + bd

    def __call__(self, *,
                 query: torch.Tensor,
                 key: torch.Tensor,
                 value: torch.Tensor):
        # `query`, `key` and `value`  have shape `[seq_len, batch_size, d_model]`
        batch_size, _ = query.shape

        # Prepare `query`, `key` and `value` for attention computation
        # These will then have shape `[seq_len, batch_size, heads, d_k]`
        query = self.query(query)
        key = self.key(key)
        value = self.value(value)

        # Compute attention scores $Q K^T$
        # Results in a tensor of shape `[seq_len, seq_len, batch_size, heads]`
        scores = self.get_scores(query, key)

        # Scale scores $\frac{Q K^T}{\sqrt{d_k}}$
        scores *= self.scale

        attn = self.softmax(scores)

        # Apply dropout
        attn = self.dropout(attn)

        # Multiply by values
        # $$\underset{seq}{softmax}\Bigg(\frac{Q K^T}{\sqrt{d_k}}\Bigg)V$$
        x = torch.einsum("jbh,jbhd->bhd", attn, value)

        # Save attentions for any other calculations
        self.attn = attn.detach()

        # Concatenate multiple heads
        x = x.reshape(batch_size, -1)

        # Output layer
        return self.output(x)


class FeedbackTransformerLayer(Module):
    def __init__(self, *,
                 d_model: int,
                 attn: FeedbackAttention,
                 feed_forward: FeedForward,
                 dropout_prob: float):
        super().__init__()
        self.size = d_model
        self.attn = attn
        self.feed_forward = feed_forward
        self.dropout = nn.Dropout(dropout_prob)
        self.norm_self_attn = nn.LayerNorm([d_model])
        self.norm_ff = nn.LayerNorm([d_model])

    def __call__(self, *,
                 x: torch.Tensor,
                 mem: Optional[torch.Tensor]):
        # Normalize the vectors before doing self attention
        z = self.norm_self_attn(x)
        if mem is not None:
            # Run through self attention, i.e. keys and values are from self
            self_attn = self.attn(query=z, key=mem, value=mem)
            # Add the self attention results
            x = x + self.dropout(self_attn)

        # Normalize for feed-forward
        z = self.norm_ff(x)
        # Pass through the feed-forward network
        ff = self.feed_forward(z)
        # Add the feed-forward results back
        x = x + self.dropout(ff)

        return x


class FeedbackTransformer(Module):
    """
    ## Transformer Encoder
    """

    def __init__(self, layer: FeedbackTransformerLayer, n_layers: int):
        super().__init__()
        # Make copies of the transformer layer
        self.layers = clone_module_list(layer, n_layers)
        self.norm = nn.LayerNorm([layer.size])
        self.weights = nn.Parameter(torch.ones(n_layers + 1), requires_grad=True)
        self.softmax = nn.Softmax(0)

    def __call__(self, x_seq: torch.Tensor):
        # Run through each transformer layer
        x_seq = torch.unbind(x_seq, dim=0)
        res = []
        mem = []
        for x in x_seq:
            emb = [x]
            mem_tensor = None
            if mem:
                mem_tensor = torch.stack(mem)
            for layer in self.layers:
                x = layer(x=x, mem=mem_tensor)
                emb.append(x)
            emb = torch.stack(emb)
            mem.append(torch.einsum('lbd,l->bd', emb, self.softmax(self.weights)))
            # Finally, normalize the vectors
            res.append(x)

        res = torch.stack(res)
        return self.norm(res)
