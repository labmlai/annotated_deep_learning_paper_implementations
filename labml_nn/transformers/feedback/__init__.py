"""
---
title: Feedback Transformer
summary: >
  This is an annotated implementation/tutorial the Feedback Transformer in PyTorch.
---

# Feedback Transformer

This is an implementation of the paper
[Accessing Higher-level Representations in Sequential Transformers with Feedback Memory](https://arxiv.org/abs/2002.09402).

Normal transformers process tokens in parallel and each transformer layer pays attention
to the outputs of the previous layer.
Feedback transformer pays attention to the output of all layers in previous steps.
So this adds recurrence and we need to process token-by-token.
This slows down the training significantly (about 5X - 10X depending on the sequence length).
However when predicting Feedback Transformer is faster because you can predict the next token
if you cache the memory vectors.

In order to speed up the training the paper discusses starting with a short sequence length and
gradually increasing it.
They also discuss using a pretrained parallel transformer as the starting point.

The feedback transformer doesn't keep the outputs of all layers.
Instead it keeps weighted sum of the output of all layers.
This reduces the memory used for caching during prediction.

Here's a notebook for training a feedback transformer on Tiny Shakespeare dataset.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lab-ml/nn/blob/master/labml_nn/transformers/feedback/experiment.ipynb)
[![View Run](https://img.shields.io/badge/labml-experiment-brightgreen)](https://web.lab-ml.com/run?uuid=d8eb9416530a11eb8fb50242ac1c0002)
"""

import math
from typing import Optional

import torch
from torch import nn

from labml_helpers.module import Module
from labml_nn.transformers.mha import PrepareForMultiHeadAttention
from labml_nn.transformers.feed_forward import FeedForward
from labml_nn.utils import clone_module_list


class FeedbackAttention(Module):
    r"""
    ## Feedback Attention


    This module computes recurrent attention similar to attention from original transformers
    paper.

    $$\mathop{Attention}(Q, K, V) = \underset{seq}{\mathop{softmax}}\Bigg(\frac{Q^\top K}{\sqrt{d_k}}\Bigg)V$$
    """

    def __init__(self, heads: int, d_model: int, dropout_prob: float = 0.1, *,
                 is_kv_precomputed: bool = False):
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
        self.query = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=False)
        if not is_kv_precomputed:
            self.key = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=False)
            self.value = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=True)
        else:
            self.key = None
            self.value = None

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
        A_{j} &= Q^\top K_j \\
            &= lin_q(X^q + P_q)^\top lin_k(X^k_j + P_j) \\
            &= (Q + U^Q)^\top(K_j + U^K_j)
        \end{align}

        where $Q, K_j$, are linear transformations of
         original embeddings $X^q, X^k_j$
         and $U^Q, U^K_j$ are linear transformations of
         absolute positional encodings $P_q, P_j$.
        """

        # $U^K_j$
        key_pos_emb = self.key_pos_embeddings[-key.shape[0]:]
        # $U^Q$
        query_pos_bias = self.query_pos_bias[None, :, :]

        # $(Q + U^Q)^\top(K_j + U^K_j)$
        ac = torch.einsum('bhd,jbhd->jbh', query + query_pos_bias, key)
        bd = torch.einsum('bhd,jhd->jbh', query, key_pos_emb)

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
        if self.key:
            key = self.key(key)
        if self.value:
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
                 key: Optional[torch.Tensor],
                 value: Optional[torch.Tensor]):
        # If there is memory
        if key is not None:
            # Normalize the vectors before doing self attention
            z = self.norm_self_attn(x)
            # Run through self attention, i.e. keys and values are from self
            self_attn = self.attn(query=z, key=key, value=value)
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
                x = layer(x=x, key=mem_tensor, value=mem_tensor)
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


class StackFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, memory, memory_grad, last, n):
        ctx._mem_grad = memory_grad
        ctx._n = n
        return memory[:n + 1]

    @staticmethod
    def backward(ctx, grad_output):
        n = ctx._n
        memory_grad = ctx._mem_grad
        memory_grad[:n + 1] += grad_output
        return None, None, memory_grad[n], None


class Stack:
    def __init__(self, max_len: int):
        self.max_len = max_len
        self.memory = None
        self.memory_grad = None
        self.last = None
        self.n = -1
        self.last_get_n = -1

    def append(self, n: int, vector: torch.Tensor):
        assert n == 0 or self.last_get_n == n - 1, f"{n}, {self.last_get_n}"

        with torch.no_grad():
            if self.memory is None or self.memory.shape[1:] != vector.shape:
                assert n == 0
                self.memory = vector.new_zeros(self.max_len, *vector.shape, requires_grad=False)
                self.memory_grad = vector.new_zeros(self.memory.shape, requires_grad=False)
            elif n == 0:
                self.memory_grad.fill_(0.)

            # memory[n] = vector.detach()
            self.memory.data[n] = vector.detach()
            self.n = n

        self.last = vector

    def get(self):
        self.last_get_n = self.n
        return StackFunction.apply(self.memory, self.memory_grad, self.last, self.n)


class FeedbackTransformerKV(Module):
    """
    ## Feedback Transformer Module
    """

    def __init__(self, layer: FeedbackTransformerLayer, n_layers: int, d_model: int, heads: int):
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

        d_k = d_model // heads
        self.key = PrepareForMultiHeadAttention(d_model, heads, d_k, bias=False)
        self.value = PrepareForMultiHeadAttention(d_model, heads, d_k, bias=False)

        self.mem_key = Stack(512)
        self.mem_value = Stack(512)

    def __call__(self, x_seq: torch.Tensor):
        """
        * `x_seq` is the input with shape `[seq_len, batch_size, d_model]`
        """

        # Split the input to a list along the sequence axis
        x_seq = torch.unbind(x_seq, dim=0)
        # List to store the outputs
        res = []
        # For each input step
        for step, x in enumerate(x_seq):
            # List to store layer outputs
            layer_outputs = [x]

            # If there is memory, stack them into a vector
            key_tensor = None
            value_tensor = None
            if step > 0:
                key_tensor = self.mem_key.get()
                value_tensor = self.mem_value.get()

            # Run through each layer
            for layer in self.layers:
                # Get layer output
                x = layer(x=x, key=key_tensor, value=value_tensor)
                # Append them to the list of layer outputs
                layer_outputs.append(x)

            # Stack the layer outputs to a tensor
            layer_outputs = torch.stack(layer_outputs)
            # Calculate the memory vector as a weighted sum of layer outputs
            mem = torch.einsum('lbd,l->bd', layer_outputs, self.softmax(self.weights))
            self.mem_key.append(step, self.key(mem))
            self.mem_value.append(step, self.value(mem))
            # Append the output to results
            res.append(x)

        # Stack the output tensors
        res = torch.stack(res)
        # Normalize the output
        return self.norm(res)
