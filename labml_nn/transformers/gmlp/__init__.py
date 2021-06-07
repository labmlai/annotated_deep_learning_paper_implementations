"""
---
title: Pay Attention to MLPs (gMLP)
summary: >
  This is an annotated implementation/tutorial of Pay Attention to MLPs (gMLP) in PyTorch.
---

# Pay Attention to MLPs (gMLP)

This is a [PyTorch](https://pytorch.org) implementation of the paper
[Pay Attention to MLPs](https://papers.labml.ai/paper/2105.08050).

This paper introduces a Multilayer Perceptron (MLP) based architecture with gating,
which they name **gMLP**. It consists of a stack of $L$ *gMLP* blocks.

Here is [the training code](experiment.html) for a gMLP model based autoregressive model.

[![View Run](https://img.shields.io/badge/labml-experiment-brightgreen)](https://app.labml.ai/run/01bd941ac74c11eb890c1d9196651a4a)
"""

from typing import Optional

import torch
from torch import nn


class GMLPBlock(nn.Module):
    """
    ## gMLP Block

    Each block does the following transformations to input embeddings
    $X \in \mathbb{R}^{n \times d}$ where $n$ is the sequence length
    and $d$ is the dimensionality of the embeddings:

    \begin{align}
    Z &= \sigma(XU) \\
    \tilde{Z} &= s(Z) \\
    Y &= \tilde{Z}V \\
    \end{align}

    where $V$ and $U$ are learnable projection weights.
    $s(\cdot)$ is the Spacial Gating Unit defined below.
    Output dimensionality of $s(\cdot)$ will be half of $Z$.
    $\sigma$ is an activation function such as
    [GeLU](https://pytorch.org/docs/stable/generated/torch.nn.GELU.html).
    """

    def __init__(self, d_model: int, d_ffn: int, seq_len: int):
        """
        `d_model` is the dimensionality ($d$) of $X$
        `d_ffn` is the dimensionality of $Z$
        `seq_len` is the length of the token sequence ($n$)
        """
        super().__init__()
        # Normalization layer fro Pre-Norm
        self.norm = nn.LayerNorm([d_model])
        # Activation function $\sigma$
        self.activation = nn.GELU()
        # Projection layer for $Z = \sigma(XU)$
        self.proj1 = nn.Linear(d_model, d_ffn)
        # Spacial Gating Unit $s(\cdot)$
        self.sgu = SpacialGatingUnit(d_ffn, seq_len)
        # Projection layer for $Y = \tilde{Z}V$
        self.proj2 = nn.Linear(d_ffn // 2, d_model)
        # Embedding size (required by [Encoder](../models.html#Encoder).
        # We use the encoder module from transformer architecture and plug
        # *gMLP* block as a replacement for the [Transformer Layer](../models.html#Encoder).
        self.size = d_model

    def forward(self, *, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        * `x` is the input embedding tensor $X$ of shape `[seq_len, batch_size, d_model]`
        * `mask` is a boolean mask of shape `[seq_len, seq_len, 1]` that controls the visibility of tokens
         among each other.
        """
        # Keep a copy for shortcut connection
        shortcut = x
        # Normalize $X$
        x = self.norm(x)
        # Projection and activation $Z = \sigma(XU)$
        z = self.activation(self.proj1(x))
        # Spacial Gating Unit $\tilde{Z} = s(Z)$
        z = self.sgu(z, mask)
        # Final projection $Y = \tilde{Z}V$
        z = self.proj2(z)

        # Add the shortcut connection
        return z + shortcut


class SpacialGatingUnit(nn.Module):
    """
    ## Spatial Gating Unit

    $$s(Z) = Z_1 \odot f_{W,b}(Z_2)$$

    where $f_{W,b}(Z) = W Z + b$ is a linear transformation along the sequence dimension,
    and $\odot$ is element-wise multiplication.
    $Z$ is split into to parts of equal size $Z_1$ and $Z_2$ along the channel dimension (embedding dimension).
    """
    def __init__(self, d_z: int, seq_len: int):
        """
        * `d_z` is the dimensionality of $Z$
        * `seq_len` is the sequence length
        """
        super().__init__()
        # Normalization layer before applying $f_{W,b}(\cdot)$
        self.norm = nn.LayerNorm([d_z // 2])
        # Weight $W$ in $f_{W,b}(\cdot)$.
        #
        # The paper notes that it's important to initialize weights to small values and the bias to $1$,
        # so that during the initial training $s(\cdot)$ is close to identity (apart from the split).
        self.weight = nn.Parameter(torch.zeros(seq_len, seq_len).uniform_(-0.01, 0.01), requires_grad=True)
        # Weight $b$ in $f_{W,b}(\cdot)$
        #
        # The paper notes that it's important to initialize bias to $1$.
        self.bias = nn.Parameter(torch.ones(seq_len), requires_grad=True)

    def forward(self, z: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        * `z` is the input $Z$ of shape `[seq_len, batch_size, d_z]`
        * `mask` is is a boolean mask of shape `[seq_len, seq_len, 1]` that controls the visibility of tokens
         among each other. The last dimension of size `1` is the batch, which we have in other transformer
         implementations and was left for compatibility.
        """

        # Get sequence length
        seq_len = z.shape[0]
        # Split $Z$ into $Z_1$ and $Z_2$
        z1, z2 = torch.chunk(z, 2, dim=-1)

        # Check mask
        if mask is not None:
            # `mask` has shape `[seq_len_q, seq_len_k, batch_size]`.
            # The batch dimension should be of size `1` because this implementation supports
            # only same mask for all samples in the batch.
            assert mask.shape[0] == 1 or mask.shape[0] == seq_len
            assert mask.shape[1] == seq_len
            # Here we only support the same mask for all samples
            assert mask.shape[2] == 1
            # Remove the batch dimension
            mask = mask[:, :, 0]

        # Normalize $Z_2$ before $f_{W,b}(\cdot)$
        z2 = self.norm(z2)
        # Get the weight matrix; truncate if larger than `seq_len`
        weight = self.weight[:seq_len, :seq_len]
        # Apply mask to the weights.
        #
        # If $W_{i,j}$ is $0$ then $f_{W,b}(Z_2)_i$ will not get any information
        # from token $j$.
        if mask is not None:
            weight = weight * mask

        # $f_{W,b}(Z_2) = W Z_2 + b$
        z2 = torch.einsum('ij,jbd->ibd', weight, z2) + self.bias[:seq_len, None, None]

        # $Z_1 \odot f_{W,b}(Z_2)$
        return z1 * z2
