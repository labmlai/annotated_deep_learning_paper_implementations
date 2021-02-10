"""
---
title: Feedback Transformer
summary: >
  This is an annotated implementation/tutorial the Feedback Transformer in PyTorch.
---

# Feedback Transformer

This is a [PyTorch](https://pytorch.org) implementation of the paper
[Accessing Higher-level Representations in Sequential Transformers with Feedback Memory](https://arxiv.org/abs/2002.09402).

Normal transformers process tokens in parallel. Each transformer layer pays attention
to the outputs of the previous layer.
Feedback transformer pays attention to the output of all layers in previous steps.
So this adds recurrence, and we need to process token-by-token.
This slows down the training significantly (about 5X - 10X depending on the sequence length).
However, when predicting Feedback Transformer is faster because you can predict the next token
if you cache the memory vectors.

In order to speed up the training, the paper discusses starting with a short sequence length and
gradually increasing it.
They also discuss using a pretrained parallel transformer as the starting point.

The original feedback transformer doesn't keep the outputs of all layers.
Instead it keeps weighted sum of the output of all layers.
This reduces the memory used for caching during prediction.
The first half of this file implements this.

The updated feedback transformer shares weights $W^l_k$ and $W^l_v$ used
to calculate keys and values among the layers.
We then calculate the keys and values for each step only once and keep
them cached.
The [second half](#shared_kv) of this file implements this.
We implemented a custom PyTorch function to improve performance.

Here's [the training code](experiment.html) and a notebook for training a feedback transformer on Tiny Shakespeare dataset.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lab-ml/nn/blob/master/labml_nn/transformers/feedback/experiment.ipynb)
[![View Run](https://img.shields.io/badge/labml-experiment-brightgreen)](https://web.lab-ml.com/run?uuid=d8eb9416530a11eb8fb50242ac1c0002)
"""

import math
from typing import Optional

import torch
from torch import nn

from labml_helpers.module import Module
from labml_nn.transformers.feed_forward import FeedForward
from labml_nn.transformers.mha import PrepareForMultiHeadAttention
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
        * `is_kv_precomputed` is whether key, value tensors are already calculated
        """

        super().__init__()

        # Number of features per head
        self.d_k = d_model // heads
        #
        self.heads = heads

        # These transform the `query` multi-headed attention.
        self.query = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=False)
        # These transform the `key` and `value` for multi-headed attention.
        if not is_kv_precomputed:
            self.key = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=False)
            self.value = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=True)
        # Keys and values are already calculated
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
        # Relative positional embedding bias for key relative to the query.
        self.key_pos_bias = nn.Parameter(torch.zeros((self.P, heads)), requires_grad=True)
        # Positional embeddings for the query is independent of the position of the query
        self.query_pos_bias = nn.Parameter(torch.zeros((heads, self.d_k)), requires_grad=True)

        # We store attentions so that it can be used for logging, or other computations if needed
        self.attn = None

    def get_scores(self, query: torch.Tensor, key: torch.Tensor):
        r"""
        ### Get attention scores

        We use relative positional encodings for attention, similar
        to [relative multi-head attention form Transformer-XL paper](../relative_mha.html).

        Attention from current step's query to key in step $j$ (relative to current step) is,

        \begin{align}
        A_{j} &= Q^\top K_j \\
            &= lin_q(X^q + P_q)^\top lin_k(X^k_j + P_j) \\
            &= (Q + U^Q)^\top(K_j + U^K_j) \\
            &= \underset{\color{lightgreen}{A}}{Q^\top K_j} +
               \underset{\color{lightgreen}{B}}{Q^\top U^K_j} +
               \underset{\color{lightgreen}{C}}{{U^Q}^\top K_j} +
               \underset{\color{lightgreen}{D}}{{U^Q}^\top U^K_j}
        \end{align}

        where $Q, K_j$, are linear transformations of
         original embeddings $X^q, X^k_j$
         and $U^Q, U^K_j$ are linear transformations of
         positional encodings $P_q, P_j$.

        We replace term $\color{lightgreen}{D}$ with $S_j$.
        """

        # $U^K_j$
        key_pos_emb = self.key_pos_embeddings[-key.shape[0]:]
        # $U^Q$
        query_pos_bias = self.query_pos_bias[None, :, :]
        # $S_j$
        key_pos_bias = self.key_pos_bias[-key.shape[0]:]

        # $\underset{\color{lightgreen}{A}}{Q^\top K_j} + \underset{\color{lightgreen}{C}}{{U^Q}^\top K_j}$
        ac = torch.einsum('bhd,jbhd->jbh', query + query_pos_bias, key)
        # $\underset{\color{lightgreen}{B}}{Q^\top U^K_j} + \underset{\color{lightgreen}{D}}{S_j}$
        bd = torch.einsum('bhd,jhd->jbh', query, key_pos_emb) + key_pos_bias[:, None, :]

        # $A_j$
        return ac + bd

    def forward(self, *,
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

        # Compute attention scores.
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

    def forward(self, *,
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

    def forward(self, x_seq: torch.Tensor):
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


# <a id="shared_kv">
# # Shared keys and values among layers
# </a>

class StackFunction(torch.autograd.Function):
    """
    ### Stack Function implementation

    We implement a custom function instead of appending to a python list
    and then doing `torch.stack`.
    This greatly improves the performance over calling `torch.stack` at
    each step along the sequence.
    Everytime `torch.stack` is called, it creates a new tensor, while
    this method and the accompanying class `Stack` share memory for each step.
    """

    @staticmethod
    def forward(ctx, memory, memory_grad, last, n):
        """
        * `ctx` is the context of the function (which lets us cache stuff)
        * `memory` is the shared memory tensor where we stack and store the values of each step (keys & values)
        * `memory_grad` is the shared memory tensor to store and accumulate gradients of each step
        * `last` is the last value stacked
        * `n` is the number of steps (i.e. size of the stack)

        This returns the stacked tensor for steps upto `n`.
        """

        # Cache accumulated gradients
        ctx._mem_grad = memory_grad
        # Cache the size of the stack
        ctx._n = n
        # Return the stack
        return memory[:n + 1]

    @staticmethod
    def backward(ctx, grad_output):
        """
        * `grad_output` is the gradient with respect to the output of about `forward` function

        This accumulates the gradients in the shared memory tensor and return the
        gradients with respect to the `last` result in the stack.
        """
        # Get the current size of the stack
        n = ctx._n
        # Get the accumulated gradients
        memory_grad = ctx._mem_grad
        # Add the gradients
        memory_grad[:n + 1] += grad_output
        # Return the gradients w.r.t to last value in the stack
        return None, None, memory_grad[n], None


class Stack:
    """
    ### Stack Module

    This uses the stack function defined above, and does the necessary initializations.
    """
    def __init__(self, max_len: int):
        """
        * `max_len` is the maximum size of the stack
        """
        self.max_len = max_len
        self.memory = None
        self.memory_grad = None
        self.last = None
        self.n = -1
        self.last_get_n = -1

    def append(self, n: int, value: torch.Tensor):
        """
        * `n` is the size of the stack
        * `value` is the tensor that needs to be added to the stack
        """

        # You need to get (use) the stack after adding a value.
        # Otherwise this implementation fails
        assert n == 0 or self.last_get_n == n - 1, f"{n}, {self.last_get_n}"

        # Do this without gradients
        with torch.no_grad():
            # Initialize the shared memory tensor to keep the stack
            if self.memory is None or self.memory.shape[1:] != value.shape:
                # This should only happen when the stack is empty
                assert n == 0
                # Create a tensor for the stack
                self.memory = value.new_zeros(self.max_len, *value.shape, requires_grad=False)
                # Create a tensor to accumulate the gradients
                self.memory_grad = value.new_zeros(self.memory.shape, requires_grad=False)
            # The memory is already initialized but we are resetting the stack.
            #
            # This could have been another function like `reset`, but
            # we found this easier to use.
            elif n == 0:
                # Reset accumulated gradients
                self.memory_grad.fill_(0.)

            # Set the value in the correct position of the stack
            self.memory.data[n] = value.detach()
            # Keep track of the stack (for debugging)
            self.n = n

        # Keep track of the last value added to the stack.
        # We need this to be passed on to `StackFunction` in order
        # to get the gradients propagated backwards.
        self.last = value

    def get(self):
        """
        Returns the stack
        """

        # Keep track of the size of the stack when it was used.
        # This is used for a sanity check in `append`.
        self.last_get_n = self.n
        # Take it all through `StackFunction` so that `StackFunction.backwards`
        # is called by PyTorch during backpropagation.
        return StackFunction.apply(self.memory, self.memory_grad, self.last, self.n)


class FeedbackTransformerKV(Module):
    """
    ## Updated Feedback Transformer Module

    This is the updated feedback transformer module that caches the keys and values.
    """

    def __init__(self, layer: FeedbackTransformerLayer, n_layers: int, d_model: int, heads: int):
        """
        * `layer` is the feedback transformer layer, which we clone for each layer
        * `n_layers` is the number of layers in the transformer
        * `d_model` is the number of features in the transformer
        * 'heads' is the number of attention heads
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

        # Number of features in a head
        d_k = d_model // heads
        # Module to transform embeddings (memory) to get keys
        self.key = PrepareForMultiHeadAttention(d_model, heads, d_k, bias=False)
        # Module to transform embeddings (memory) to get keys
        self.value = PrepareForMultiHeadAttention(d_model, heads, d_k, bias=False)

        # Memory for stacked keys
        self.mem_key = Stack(512)
        # Memory for stacked values
        self.mem_value = Stack(512)

    def forward(self, x_seq: torch.Tensor):
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

            # Stack of keys and values
            key_tensor = None
            value_tensor = None
            # Get the keys and values tensors if we are beyond the initial step
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
            # Calculate the keys from memory and add it to the stack
            self.mem_key.append(step, self.key(mem))
            # Calculate the values from memory and add it to the stack
            self.mem_value.append(step, self.value(mem))
            # Append the output to results
            res.append(x)

        # Stack the output tensors
        res = torch.stack(res)
        # Normalize the output
        return self.norm(res)
