"""
---
title: Attention with Linear Biases (ALiBi)
summary: >
  Documented implementation with explanations of Attention with Linear Biases (ALiBi)
---

# Attention with Linear Biases (ALiBi)

This is an implementation of Attention with Linear Biases (ALiBi) from the paper
[Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation](https://papers.labml.ai/paper/dae907db093711ec9e9dcba33be64600).

This replaces positional encodings with biases added to attention scores (attention logits, before the softmax).
This is a relative scheme tested on autoregressive tasks, and the bias is higher for closeby tokens
and lower for far-away tokens.
The biases decrease linearly in the log scale (because it's before the softmax) and each head has a different slope.

Here's the attention formula for $i$-th token,

\begin{align}
\mathbf{a}_i
&= \text{softmax} \bigg( \mathbf{q}_i \mathbf{K}^\top + m \cdot \big[-(i-1), \dots, 1, 0 \big] \bigg) \\
&= \text{softmax} \bigg( \mathbf{q}_i \mathbf{K}^\top + m \cdot \big[0, 1, \dots, (i - 1) \big] \bigg)
\end{align}

where $\mathbf{q}_i \in \mathbb{R}^d$ is the query of the $i$-th token, $K \in \mathbb{R}^{i \times d}$ are the keys
up to $i$, and $d$ the number of features per head.
Note that the above equality halts because $\text{softmax}$ is invariant to translations
 (you can add any constant to all elements without changing the result).

Here is [the training code](experiment.html) for a ALiBi model.

[![View Run](https://img.shields.io/badge/labml-experiment-brightgreen)](https://app.labml.ai/run/e87bec2a074911ec82cdd1759f10c925)
"""
import math
from typing import Optional

import torch
from torch import nn

from labml.logger import inspect
from labml_nn.transformers.mha import MultiHeadAttention


def get_slopes(n_heads: int):
    """
    ## Get head-specific slope $m$ for each head

    * `n_heads` is the number of heads in the attention layer $n$

    The slope for first head is

    $$2^{-2^{-(\log_2 n) - 3}}$$

    The slopes for the rest of the heads are in a geometric series with a ratio same as above.

    For instance when the number of heads is $8$ the slopes are
    $$\frac{1}{2^1}, \frac{1}{2^2}, \dots, \frac{1}{2^8}$$
    """

    n = 2 ** math.floor(math.log2(n_heads))
    base = 2.0 ** (-8.0 / n)
    slopes = torch.pow(base, torch.arange(1, 1 + n))

    if n < n_heads:
        base = 2.0 ** (-4.0 / n)
        remaining_slopes = torch.pow(base, torch.arange(1, 1 + 2 * (n_heads - n), 2))

        slopes = torch.cat([slopes, remaining_slopes])

    return slopes


@torch.no_grad()
def get_alibi_biases(n_heads: int, mask: torch.Tensor):
    """
    ## Calculate the attention biases matrix

    * `n_heads` is the number of heads in the attention layer
    * `mask` is the the attention mask of shape `[seq_len_q, seq_len_k]`

    This returns a matrix of shape `[n_heads, seq_len_q, seq_len_k]` with attention biases.
    """

    # Get slopes $m$ for each head
    slopes = get_slopes(n_heads)
    # Calculate distances $[0, 1, \dots, N]$
    distance = mask.cumsum(dim=-1)
    # Multiply them pair-wise to get the bias matrix
    return distance[:, :, None] * slopes[None, None, :]


class AlibiMultiHeadAttention(MultiHeadAttention):
    """
    ## Attention with Linear Biases (ALiBi)

    We override [Multi-Head Attention](mha.html) module so we only need to
    write the `get_scores` method.
    """

    def __init__(self, heads: int, d_model: int, dropout_prob: float = 0.1):
        super().__init__(heads, d_model, dropout_prob)

        # To cache AliBi the biases
        self.alibi_biases = None

    def forward(self, *,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None):
        """
        `query`, `key` and `value` are the tensors that store
        collection of *query*, *key* and *value* vectors.
        They have shape `[seq_len, batch_size, d_model]`.

        `mask` has shape `[seq_len, seq_len, batch_size]` and
        `mask[i, j, b]` indicates whether for batch `b`,
        query at position `i` has access to key-value at position `j`.
        """

        assert mask is not None
        assert mask.shape[0] == mask.shape[1] and mask.shape[2] == 1

        # `query`, `key` and `value`  have shape `[seq_len, batch_size, d_model]`
        seq_len, batch_size, _ = query.shape

        if mask is not None:
            mask = self.prepare_mask(mask, query.shape, key.shape)

        # Prepare `query`, `key` and `value` for attention computation.
        # These will then have shape `[seq_len, batch_size, heads, d_k]`.
        query = self.query(query)
        key = self.key(key)
        value = self.value(value)

        # Compute attention scores $Q K^\top$.
        # This gives a tensor of shape `[seq_len, seq_len, batch_size, heads]`.
        scores = self.get_scores(query, key)

        # Scale scores $\frac{Q K^\top}{\sqrt{d_k}}$
        scores *= self.scale

        if self.alibi_biases is None or self.alibi_biases.shape[1] < seq_len:
            # mask has shape [seq_len, seq_len, 1, 1]
            self.alibi_biases = get_alibi_biases(scores.shape[-1], mask[:, :, 0, 0])

        # mask has shape [ seq_len, seq_len, heads] and scores has shape `[seq_len, seq_len, batch_size, heads]`
        scores += self.alibi_biases[:seq_len, :seq_len, None, :]

        # Apply mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # $softmax$ attention along the key sequence dimension
        # $\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_k}}\Bigg)$
        attn = self.softmax(scores)

        # Apply dropout
        attn = self.dropout(attn)

        # Multiply by values
        # $$\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_k}}\Bigg)V$$
        x = torch.einsum("ijbh,jbhd->ibhd", attn, value)

        # Concatenate multiple heads
        x = x.reshape(seq_len, batch_size, -1)

        # Output layer
        return self.output(x)


def _test_alibi():
    """
    Simple test function to see the slopes.
    """
    inspect(get_slopes(12).tolist(), _n=-1)
    from labml_nn.transformers.utils import subsequent_mask

    mask = subsequent_mask(8)[:, :, 0]
    inspect(mask)

    inspect(get_alibi_biases(12, mask)[:, :, 3], _n=-1)


#
if __name__ == '__main__':
    _test_alibi()
