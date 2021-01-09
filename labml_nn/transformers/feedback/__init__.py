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


class FeedbackAttention(Module):
    """
    ## Feedback Attention

    This is very similar to [Relative Multi-Head Attention](../relative_mha.html)
    but with some modifications.

    📝 Decided not to extend from [Relative Multi-Head Attention](../relative_mha.html)
     or [Multi-Head Attention](../mha.html) to improve readability.
    """

    def __init__(self, heads: int, d_model: int, dropout_prob: float = 0.1):
        super().__init__()

        self.d_k = d_model // heads
        self.heads = heads

        # These transform the `query`, `key` and `value` vectors for multi-headed attention.
        self.query = PrepareForMultiHeadAttention(d_model, heads, self.d_k, False)
        self.key = PrepareForMultiHeadAttention(d_model, heads, self.d_k, False)
        self.value = PrepareForMultiHeadAttention(d_model, heads, self.d_k, False)

        # Output layer
        self.output = nn.Linear(d_model, d_model)
        # Dropout
        self.dropout = nn.Dropout(dropout_prob)
        # Scaling factor before the softmax
        self.scale = 1 / math.sqrt(self.d_k)

        # Softmax for attention along the time dimension of `key`
        self.softmax = nn.Softmax(dim=0)

        # Number of relative positions
        self.P = 2 ** 12

        # Relative positional embeddings for key relative to the query.
        self.key_pos_embeddings = nn.Parameter(torch.zeros((self.P, heads, self.d_k)), requires_grad=True)
        # Relative positional embedding bias for key relative to the query.
        self.key_pos_bias = nn.Parameter(torch.zeros((self.P, heads)), requires_grad=True)
        # Positional embeddings for the query is independent of the position of the query
        self.query_pos_bias = nn.Parameter(torch.zeros((heads, self.d_k)), requires_grad=True)

        # We store attentions so that it can used for logging, or other computations if needed
        self.attn = None

    def get_scores(self, query: torch.Tensor, key: torch.Tensor):
        """
        ### Get relative attention scores

        \begin{align}
        A_{j} &= lin_q(\color{cyan}{X^q + P})^T lin_k(\color{lightgreen}{X^k_j + P_j}) \\
                      &= \color{cyan}{Q^T} \color{lightgreen}{K_j} +
                         \color{cyan}{Q^T} \color{lightgreen}{U_j} +
                         \color{cyan}{V^T} \color{lightgreen}{K_j} +
                         \color{cyan}{V^T} \color{lightgreen}{U_j}
        \end{align}

        where $\color{cyan}{Q}, \color{lightgreen}{K_j}$, are linear transformations of
         original embeddings $\color{cyan}{X^q}, \color{lightgreen}{X^k_j}$
         and $\color{cyan}{V}, \color{lightgreen}{U_j}$ are linear transformations of
         absolute positional encodings $\color{cyan}{P}, \color{lightgreen}{P_j}$.

        We replace $\color{cyan}{V^T} \color{lightgreen}{U_j}$ with
        $S_j$.

        \begin{align}
        A^{rel}_{j} &= \underset{\mathbf{A}}{\color{cyan}{Q^T} \color{lightgreen}{K_j}} +
                       \underset{\mathbf{B}}{\color{cyan}{Q^T} \color{lightgreen}{U_j}} +
                       \underset{\mathbf{C}}{\color{cyan}{V^T} \color{lightgreen}{K_j}} +
                       \underset{\mathbf{D}}{\color{orange}{S_j}}
        \end{align}
        """

        # $\color{lightgreen}{U_j}$
        key_pos_emb = self.key_pos_embeddings[-key.shape[0]:]
        # $\color{orange}{S_j}$
        key_pos_bias = self.key_pos_bias[-key.shape[0]:]
        # $\color{cyan}{V^T}$
        query_pos_bias = self.query_pos_bias[None, :, :]

        # $\underset{\mathbf{A}}{\color{cyan}{Q^T} \color{lightgreen}{K_j}} +
        # \underset{\mathbf{C}}{\color{cyan}{V^T} \color{lightgreen}{K_j}}$
        ac = torch.einsum('bhd,jbhd->jbh', query + query_pos_bias, key)
        # $\underset{\mathbf{B}}{\color{cyan}{Q^T} \color{lightgreen}{U_j}} +
        # \underset{\mathbf{D}}{\color{orange}{S_j}}$
        bd = torch.einsum('bhd,jhd->jbh', query, key_pos_emb) + key_pos_bias[:, None, :]

        return ac + bd

    def __call__(self, *,
                 query: torch.Tensor,
                 key: torch.Tensor,
                 value: torch.Tensor):
        """
        * `query` has shape `[batch_size, d_model]`
        * `key` and `value` has shape `[seq_len, batch_size, d_model]`
        """

        # Prepare `query`, `key` and `value` for attention computation
        # `key` and `value`  will then have shape `[seq_len, batch_size, heads, d_k]`
        # and `query` will have shape `[batch_size, heads, d_k]`
        query = self.query(query)
        key = self.key(key)
        value = self.value(value)

        # Compute attention scores
        # Results in a tensor of shape `[seq_len, batch_size, heads]`
        scores = self.get_scores(query, key)

        # Scale scores $\frac{1}{\sqrt{d_k}}$
        scores *= self.scale

        # Softmax
        attn = self.softmax(scores)

        # Apply dropout
        attn = self.dropout(attn)

        # Multiply by the values
        x = torch.einsum("jbh,jbhd->bhd", attn, value)

        # Concatenate multiple heads
        x = x.reshape(x.shape[0], -1)

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
        if mem is not None:
            # Normalize the vectors before doing self attention
            z = self.norm_self_attn(x)
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
        # Final normalization layer
        self.norm = nn.LayerNorm([layer.size])
        #
        self.weights = nn.Parameter(torch.ones(n_layers + 1), requires_grad=True)
        #
        self.softmax = nn.Softmax(0)

    def __call__(self, x_seq: torch.Tensor):
        x_seq = torch.unbind(x_seq, dim=0)
        res = []
        mem = []
        # For each input step
        for x in x_seq:
            # List of embeddings from each layer
            emb = [x]

            mem_tensor = torch.stack(mem) if mem else None

            # Run through each layer
            for layer in self.layers:
                x = layer(x=x, mem=mem_tensor)
                emb.append(x)

            # Stack embeddings
            emb = torch.stack(emb)
            # Weighted sum of embeddings
            mem.append(torch.einsum('lbd,l->bd', emb, self.softmax(self.weights)))
            # Append the output to results
            res.append(x)

        # Stack the output tensors
        res = torch.stack(res)
        # Normalize
        return self.norm(res)
