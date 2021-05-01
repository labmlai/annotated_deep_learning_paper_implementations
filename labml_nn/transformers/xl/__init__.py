"""
---
title: Transformer XL
summary: >
  Documented implementation with explanations of a
  Transformer-XL model.
---

# Transformer XL

This is an implementation of
[Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](https://arxiv.org/abs/1901.02860)
in [PyTorch](https://pytorch.org).

Transformer has a limited attention span,
equal to the length of the sequence trained in parallel.
All these positions have a fixed positional encoding.
Transformer XL increases this attention span by letting
each of the positions pay attention to precalculated past embeddings.
For instance if the context length is $l$, it will keep the embeddings of
all layers for previous batch of length $l$ and feed them to current step.
If we use fixed-positional encodings these pre-calculated embeddings will have
the same positions as the current context.
They introduce relative positional encoding, where the positional encodings
are introduced at the attention calculation.

Annotated implementation of relative multi-headed attention is in [`relative_mha.py`](relative_mha.html).

Here's [the training code](experiment.html) and a notebook for training a transformer XL model on Tiny Shakespeare dataset.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lab-ml/nn/blob/master/labml_nn/transformers/xl/experiment.ipynb)
[![View Run](https://img.shields.io/badge/labml-experiment-brightgreen)](https://app.labml.ai/run/d3b6760c692e11ebb6a70242ac1c0002)
"""


from typing import List, Optional

import torch
import torch.nn as nn

from labml_helpers.module import Module
from labml_nn.utils import clone_module_list
from .relative_mha import RelativeMultiHeadAttention
from ..feed_forward import FeedForward


class TransformerXLLayer(Module):
    """
    ## Transformer XL Layer

    The transformer XL model comprises of a number of these layers.
    """
    def __init__(self, *,
                 d_model: int,
                 self_attn: RelativeMultiHeadAttention,
                 feed_forward: FeedForward,
                 dropout_prob: float):
        """
        * `d_model` is the token embedding size
        * `self_attn` is the [self attention module](relative_mha.html)
        * `feed_forward` is the feed forward module
        * `dropout_prob` is the probability of dropping out after self attention and FFN
        """
        super().__init__()
        self.size = d_model
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.dropout = nn.Dropout(dropout_prob)
        self.norm_self_attn = nn.LayerNorm([d_model])
        self.norm_ff = nn.LayerNorm([d_model])

    def forward(self, *,
                x: torch.Tensor,
                mem: Optional[torch.Tensor],
                mask: torch.Tensor):
        """
        * `x` is a tensor of the token level feature vectors of shape `[seq_len, batch_size, d_model]`
        * `mem` is a tensor of the past token level feature vectors of shape `[mem_len, batch_size, d_model]`
        * `mask` is a matrix of shape `[seq_len, mem_len + seq_len, batch_size]` or `[seq_len, mem_len + seq_len, 1]`.
        `mask[i, j]` is  true if token at `i` can see token at `j`.
        """
        # Normalize the vectors before doing self attention
        z = self.norm_self_attn(x)
        # If there is memory
        if mem is not None:
            # Normalize it
            mem = self.norm_self_attn(mem)
            # Concatenate with `z`
            m_z = torch.cat((mem, z), dim=0)
        # Ignore if there is no memory
        else:
            m_z = z
        # Attention
        self_attn = self.self_attn(query=z, key=m_z, value=m_z, mask=mask)
        # Add the attention results
        x = x + self.dropout(self_attn)

        # Normalize for feed-forward
        z = self.norm_ff(x)
        # Pass through the feed-forward network
        ff = self.feed_forward(z)
        # Add the feed-forward results back
        x = x + self.dropout(ff)

        #
        return x


class TransformerXL(Module):
    """
    ## Transformer XL Model

    This consists of multiple transformer XL layers
    """

    def __init__(self, layer: TransformerXLLayer, n_layers: int):
        super().__init__()
        # Make copies of the transformer layer
        self.layers = clone_module_list(layer, n_layers)
        # Final normalization layer
        self.norm = nn.LayerNorm([layer.size])

    def forward(self, x: torch.Tensor, mem: List[torch.Tensor], mask: torch.Tensor):
        """
        * `x` is a tensor of the token embeddings vectors of shape `[seq_len, batch_size, d_model]`
        * `mem` is a list of tensors of the past token level feature vectors of shape
        `[mem_len, batch_size, d_model]`  for each layer
        * `mask` is the masking matrix
        """
        # List to store token level feature vectors,
        # which will become the memories for the next sequential batch.
        new_mem = []
        # Run through each transformer layer
        for i, layer in enumerate(self.layers):
            # Add to the list of feature vectors
            new_mem.append(x.detach())
            # Memory
            m = mem[i] if mem else None
            # Run through the transformer XL layer
            x = layer(x=x, mem=m, mask=mask)
        # Finally, normalize the vectors
        return self.norm(x), new_mem
