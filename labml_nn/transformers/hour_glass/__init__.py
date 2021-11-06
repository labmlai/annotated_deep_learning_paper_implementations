"""
---
title: Hierarchical Transformers Are More Efficient Language Models
summary: >
  This is an annotated implementation/tutorial of hourglass model in PyTorch.
---

# Hierarchical Transformers Are More Efficient Language Models

This is a [PyTorch](https://pytorch.org) implementation of the paper
[Hierarchical Transformers Are More Efficient Language Models](https://papers.labml.ai/paper/2110.13711).

This paper introduces a hierarchical transformer architecture to handle long sequences
efficiently. The first half of the transformer layers down-sample tokens and the second
half up-samples with direct skip connections between layers of the same resolution.
This is a little similar to [U-Net](../../diffusion/ddpm/unet.html) for vision tasks.

They try different up-sampling and down-sampling techniques and build a model
with the best performing up and down-sampling techniques which they call the
hourglass model.

Here we have implemented the simplest up-sampling and down-sampling techniques for simplicity.
We will consider adding more complex (and better performing) implementations later.

Here is [the training code](experiment.html) for the hourglass model.

[![View Run](https://img.shields.io/badge/labml-experiment-brightgreen)](https://app.labml.ai/run/855b82363e4911ec9ae4a5b9c69d5061)
"""

from typing import List

import torch
from torch import nn

from labml_helpers.module import Module
from labml_nn.transformers import MultiHeadAttention, TransformerLayer
from labml_nn.transformers.feed_forward import FeedForward
from labml_nn.transformers.utils import subsequent_mask


class HourGlass(Module):
    """
    ## Hourglass model

    This model recursively adds layers to the middle while shortening the sequence by down-sampling.
    The shortened sequence processed by another hourglass model is sandwiched between two normal transformer
    layers. (A transformer layer has a [self-attention layer](../mha.html)
     and a [position-wise feed-forward layer](../feed_forward.html)).
    """

    def __init__(self, n_heads: int, d_model: int, dropout: float, d_ff: int, shortening_factors: List[int]):
        """
        * `n_heads` is the number of heads in [multi-head attention layers](../mha.html)
        * `d_model` is the size of the token embeddings
        * `dropout` is the dropout probability
        * `d_ff` is the dimensionality of the hidden layer in [position-wise feed-forward layers](../feed_forward.html)
        * `shortening_factors` is the list of shortening factors
        """
        super().__init__()

        # The transformer layer before down-sampling
        self.pre = TransformerLayer(d_model=d_model,
                                    # [Multi-head attention layer](../mha.html)
                                    self_attn=MultiHeadAttention(n_heads, d_model, dropout),
                                    # [Position wise feed-forward layers](.. / feed_forward.html)
                                    feed_forward=FeedForward(d_model, d_ff, dropout),
                                    #
                                    dropout_prob=dropout)
        # Auto-regressive mask
        self.mask = AutoregressiveMask()

        # The shortening factor $k$ (or the down-sampling rate)
        k = shortening_factors[0]

        # We shift the tokens to the right by $k - 1$ steps to make sure
        # information doesn't leak from the future tokens to past tokens
        # as a result of down-sampling and up-sampling
        self.shift_right = ShiftRight(k - 1)
        # Shortening or the down-sampling layer. We use the simplest form - average pooling.
        # The paper shows that attention based down sampling works best, which we haven't implemented yet.
        self.shortening = AvgPoolShortening(k)

        # If there are no more shortening (middle of the hourglass)
        if len(shortening_factors) == 1:
            # The center layer is another transformer layer
            self.shortened = TransformerLayer(d_model=d_model,
                                              self_attn=MultiHeadAttention(n_heads, d_model, dropout),
                                              feed_forward=FeedForward(d_model, d_ff, dropout),
                                              dropout_prob=dropout)
            # Autoregressive mask
            self.mask_short = AutoregressiveMask()
            self.hour_glass = None
        else:
            # Insert another hourglass model recursively
            self.hour_glass = HourGlass(n_heads, d_model, dropout, d_ff, shortening_factors[1:])

        # Up-sampling layer. We use naive up-sampling for simplicity and the paper shows attention based up sampling
        # works better.
        self.up_sampling = NaiveUpSampling(k)

        # The final transformer layer after up-sampling
        self.post = TransformerLayer(d_model=d_model,
                                     self_attn=MultiHeadAttention(n_heads, d_model, dropout),
                                     feed_forward=FeedForward(d_model, d_ff, dropout),
                                     dropout_prob=dropout)

    def forward(self, x: torch.Tensor):
        # Initial transformer layer
        # $$x \leftarrow PreVanillaLayers(x)$$
        x = self.pre(x=x, mask=self.mask(x))
        # Shifting and shortening
        # $$x' \leftarrow Shortening(ShiftRight(x,k−1),k)$$
        x_short = self.shortening(self.shift_right(x))

        # If we are at the center of the hourglass,
        # $$\textbf{\small if } \text{\small E\scriptsize MPTY}(shorten\_factors) \textbf{\small then}$$
        if self.hour_glass is None:
            # Center transformer layer
            # $$x' \leftarrow ShortenedLayers(x')$$
            x_short = self.shortened(x=x_short, mask=self.mask_short(x_short))
        # $$\textbf{else}$$
        else:
            # $$x' \leftarrow \text{\small H\scriptsize OURGLASS}(x, shorten\_factors)$$
            x_short = self.hour_glass(x_short)

        # Up-sample the shortened sequence and add a skip connection
        # $$x \leftarrow x + Upsampling(x, x', k)$$
        x = x + self.up_sampling(x, x_short)
        # Final transformer layer
        # $$x \leftarrow PostVanillaLayers(x)$$
        x = self.post(x=x, mask=self.mask(x))

        #
        return x


class ShiftRight(Module):
    """
    ### Shift right operation

    This shifts the sequence to the right by the given number of steps
    """

    def __init__(self, shift: int):
        """
        * `shift` is the number of steps to shift by
        """
        super().__init__()
        # cannot be negative
        assert shift >= 0
        #
        self.shift = shift

    def forward(self, x: torch.Tensor):
        """
        * `x` is a tensor of shape `[seq_len, ...]`
        """
        # If the shift is $0$ return the original
        if self.shift == 0:
            return x
        # Zeros to be appended to the left
        prefix = x.new_zeros([self.shift, *x.shape[1:]])
        # Concatenate the zeros and truncate the right
        return torch.cat([prefix, x[:-self.shift]])


class AvgPoolShortening(Module):
    """
    ### Average pool shortening

    This down-samples by a given factor with average pooling
    """

    def __init__(self, k: int):
        """
        * `k` is the shortening factor
        """
        super().__init__()
        # Average pooling layer
        self.pool = nn.AvgPool1d(k, ceil_mode=True)

    def forward(self, x: torch.Tensor):
        """
        * `x` is of shape `[seq_len, batch_size, d_model]`
        """
        # Pooling layer accepts shape `[batch_size, d_model, seq_len]` so we
        # permute axes.
        return self.pool(x.permute(1, 2, 0)).permute(2, 0, 1)


class NaiveUpSampling(Module):
    """
    ### Naive up-sampling

    This up-samples by repeating
    """

    def __init__(self, k: int):
        """
        * `k` is the shortening factor
        """
        super().__init__()
        self.k = k

    def forward(self, x: torch.Tensor, x_short: torch.Tensor):
        """
        * `x` is the tensor with embeddings before down-sampling
        * `x_short` is the tensor of higher density (to be up-sampled) representations
        """
        # Repeat across the sequence dimension
        expanded = torch.repeat_interleave(x_short, self.k, dim=0)
        # Truncate the extra embeddings at the end
        expanded = expanded[:x.shape[0]]

        #
        return expanded


class AutoregressiveMask(Module):
    """
    ### Generate auto-regressive mask
    """

    def __init__(self):
        super().__init__()
        self.mask = None

    def forward(self, x: torch.Tensor):
        # Create a mask if we haven't created or sizes have changed
        if self.mask is None or self.mask.size(0) != len(x):
            # [Subsequent mask](../utils.html), will mask out tokens from seeing future tokens
            self.mask = subsequent_mask(len(x)).to(x.device)

        #
        return self.mask


class LinearPoolingShortening(Module):
    """
    ### 🚧 Linear pooling for down-sampling

    This concatenates the consecutive tokens embeddings that need to be merged and do a linear
    transformation to map it to the size of a single token embedding.
    """

    def __init__(self):
        super().__init__()
        raise NotImplementedError


class AttentionBasedShortening(Module):
    """
    ### 🚧 Down-sampling with attention

    \begin{align}
    x' &= S(x) + Attention \Big(Q=S(x),K = x, V =x \Big) \\
    x' &= x' + FFN(x')
    \end{align}

    where $S(x)$ is average pooling or linear pooling.
    """

    def __init__(self):
        super().__init__()
        raise NotImplementedError


class LinearUpSampling(Module):
    """
    ### 🚧 Linear projection for up-sampling

    Make a linear projection of dense token embeddings to a size of $d_{\text{model}} k$.
    """

    def __init__(self):
        super().__init__()
        raise NotImplementedError


class AttentionBasedUpSampling(Module):
    """
    ### 🚧 Attention based up-sampling

    \begin{align}
    x &= U(x,x') + Attention \Big(Q=U(x,x'),K = x', V = x' \Big) \\
    x &= x + FFN(x)
    \end{align}

    where $U(x,x') = x + LinearUpsampling(x')$
    """

    def __init__(self):
        super().__init__()
        raise NotImplementedError
