"""
---
title: "MLP-Mixer: An all-MLP Architecture for Vision"
summary: >
  This is an annotated implementation/tutorial of MLP-Mixer: An all-MLP Architecture for Vision in PyTorch.
---

# MLP-Mixer: An all-MLP Architecture for Vision

This is a [PyTorch](https://pytorch.org) implementation of the paper
[MLP-Mixer: An all-MLP Architecture for Vision](https://papers.labml.ai/paper/2105.01601).

This paper applies the model on vision tasks.
The model is similar to a transformer with attention layer being replaced by a MLP
that is applied across the patches (or tokens in case of a NLP task).

Our implementation of MLP Mixer is a drop in replacement for the [self-attention layer](../mha.html)
in [our transformer implementation](../models.html).
So it's just a couple of lines of code, transposing the tensor to apply the MLP
across the sequence dimension.

Although the paper applied MLP Mixer on vision tasks,
we tried it on a [masked language model](../mlm/index.html).
[Here is the experiment code](experiment.html).

[![View Run](https://img.shields.io/badge/labml-experiment-brightgreen)](https://app.labml.ai/run/994263d2cdb511eb961e872301f0dbab)
"""

from typing import Optional

import torch
from torch import nn


class MLPMixer(nn.Module):
    """
    ## MLP Mixer

    This module is a drop-in replacement for [self-attention layer](../mha.html).
    It transposes the input tensor before feeding it to the MLP and transposes back,
    so that the MLP is applied across the sequence dimension (across tokens or image patches) instead
    of the feature dimension.
    """

    def __init__(self, mlp: nn.Module):
        """
        * `ffn` is the MLP module.
        """
        super().__init__()
        self.mlp = mlp

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        The [normal attention module](../mha.html) can be fed with different token embeddings for
        $\text{query}$,$\text{key}$, and $\text{value}$ and a mask.

        We follow the same function signature so that we can replace it directly.

        For MLP mixing, $$x = \text{query} = \text{key} = \text{value}$$ and masking is not possible.
        Shape of `query` (and `key` and `value`) is `[seq_len, batch_size, d_model]`.
        """

        # $\text{query}$,$\text{key}$, and $\text{value}$ all should be the same
        assert query is key and key is value
        # MLP mixer doesn't support masking. i.e. all tokens will see all other token embeddings.
        assert mask is None

        # Assign to `x` for clarity
        x = query

        # Transpose so that the last dimension is the sequence dimension.
        # New shape is `[d_model, batch_size, seq_len]`
        x = x.transpose(0, 2)
        # Apply the MLP across tokens
        x = self.mlp(x)
        # Transpose back into original form
        x = x.transpose(0, 2)

        #
        return x
