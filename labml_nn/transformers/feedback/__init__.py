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


    This module computes recurrent attention similar to attention from original transformers
    paper.

    $$\mathop{Attention}(Q, K, V) = \underset{seq}{\mathop{softmax}}\Bigg(\frac{Q^T K}{\sqrt{d_k}}\Bigg)V$$
    """

    def __init__(self, heads: int, d_model: int, dropout_prob: float = 0.1):
        """
        * 'heads' is the number of attention heads
        * `d_model` is the number of features in the transformer
        * `dropout_prob` is the attention dropout probability
        """

        super().__init__()

        # Number of features per head
        self.d_k = d_model // heads
        #
        self.heads = heads

        # These transform the `query`, `key` and `value` vectors for multi-headed attention.
        self.query = PrepareForMultiHeadAttention(d_model, heads, self.d_k,  bias=False)
        self.key = PrepareForMultiHeadAttention(d_model, heads, self.d_k,  bias=False)
        self.value = PrepareForMultiHeadAttention(d_model, heads, self.d_k,  bias=True)

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
        # Positional embeddings for the query is independent of the position of the query
        self.query_pos_bias = nn.Parameter(torch.zeros((heads, self.d_k)), requires_grad=True)

        # We store attentions so that it can used for logging, or other computations if needed
        self.attn = None

    def get_scores(self, query: torch.Tensor, key: torch.Tensor):
        """
        ### Get attention scores

        \begin{align}
        A_{j} &== Q^T K_j \\
            &= lin_q(\color{cyan}{X^q + P})^T lin_k(\color{lightgreen}{X^k_j + P_j}) \\
            &= (\color{cyan}{Q^T + V^T})(\color{lightgreen}{K_j + U_j)}
        \end{align}

        where $\color{cyan}{Q}, \color{lightgreen}{K_j}$, are linear transformations of
         original embeddings $\color{cyan}{X^q}, \color{lightgreen}{X^k_j}$
         and $\color{cyan}{V}, \color{lightgreen}{U_j}$ are linear transformations of
         absolute positional encodings $\color{cyan}{P}, \color{lightgreen}{P_j}$.
        """

        # $\color{lightgreen}{U_j}$
        key_pos_emb = self.key_pos_embeddings[-key.shape[0]:]
        # $\color{cyan}{V^T}$
        query_pos_bias = self.query_pos_bias[None, :, :]

        # $(\color{cyan}{Q^T + V^T})(\color{lightgreen}{K_j + U_j)}$
        return torch.einsum('bhd,jbhd->jbh', query + query_pos_bias, key + key_pos_emb[:, None, :, :])

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
    """
    ## Feedback Transformer Layer

    This implements a single transformer layer in the feedback transformer.
    """

    def __init__(self, *,
                 d_model: int,
                 attn: FeedbackAttention,
                 feed_forward: FeedForward,
                 dropout_prob: float):
        """
        * `d_model` is the number of features in the transformer
        * `attn` is the feedback attention module
        * `feed_forward` is the position-wise feed forward layer
        * `dropout_prob` is the dropout probability for dropout layers after attention and feed-forward
        """
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

    def __call__(self, *,
                 x: torch.Tensor,
                 mem: Optional[torch.Tensor]):
        # If there is memory
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

        #
        return x


class FeedbackTransformer(Module):
    """
    ## Feedback Transformer Module
    """

    def __init__(self, layer: FeedbackTransformerLayer, n_layers: int):
        """
        * `layer` is the feedback transformer layer, which we clone for each layer
        * `n_layers` is the number of layers in the transformer
        """

        super().__init__()
        # Make copies of the transformer layer
        self.layers = clone_module_list(layer, n_layers)
        # Final normalization layer
        self.norm = nn.LayerNorm([layer.size])
        # Memory vectors are computed as a weighted sum of representations of each layer.
        # This is the weights parameter for that.
        self.weights = nn.Parameter(torch.ones(n_layers + 1), requires_grad=True)
        # Softmax for weights before taking the weighted sum
        self.softmax = nn.Softmax(0)

    def __call__(self, x_seq: torch.Tensor):
        """
        * `x_seq` is the input with shape `[seq_len, batch_size, d_model]`
        """

        # Split the input to a list along the sequence axis
        x_seq = torch.unbind(x_seq, dim=0)
        # List to store the outputs
        res = []
        # List to store the memory vectors
        mem = []
        # For each input step
        for x in x_seq:
            # List to store layer outputs
            layer_outputs = [x]

            # If there is memory, stack them into a vector
            mem_tensor = torch.stack(mem) if mem else None

            # Run through each layer
            for layer in self.layers:
                # Get layer output
                x = layer(x=x, mem=mem_tensor)
                # Append them to the list of layer outputs
                layer_outputs.append(x)

            # Stack the layer outputs to a tensor
            layer_outputs = torch.stack(layer_outputs)
            # Calculate the memory vector as a weighted sum of layer outputs
            mem.append(torch.einsum('lbd,l->bd', layer_outputs, self.softmax(self.weights)))
            # Append the output to results
            res.append(x)

        # Stack the output tensors
        res = torch.stack(res)
        # Normalize the output
        return self.norm(res)
