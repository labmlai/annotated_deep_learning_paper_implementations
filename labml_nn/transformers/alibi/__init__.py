"""
---
title: Attention with Linear Biases (ALiBi)
summary: >
  Documented implementation with explanations of Attention with Linear Biases (ALiBi)
---

# Attention with Linear Biases (ALiBi)

This is an implementation of Attention with Linear Biases (ALiBi) from the paper
Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation
[(pdf)](https://ofir.io/train_short_test_long.pdf).

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

    # $$2^{-2^{-(\log_2 n) - 3}}$$
    s = (2 ** (-2 ** -(math.log2(n_heads) - 3)))
    # The geometric sequence
    return [s * (s ** i) for i in range(n_heads)]


def get_biases(n_heads: int, max_len: int):
    """
    ## Calculate the attention biases matrix

    * `n_heads` is the number of heads in the attention layer
    * `max_len` is the maximum sequence length

    This returns a matrix of shape `[n_heads, max_len]` with attention biases.
    """

    # Get slopes $m$ for each head
    slopes = torch.tensor(get_slopes(n_heads))
    # Calculate distances $[0, 1, \dots, N]$
    distance = torch.arange(max_len).to(torch.float)
    # Multiply them pair-wise to get the bias matrix
    return distance[:, None] * slopes[None, :]


class AlibiMultiHeadAttention(MultiHeadAttention):
    """
    ## Attention with Linear Biases (ALiBi)

    We override [Multi-Head Attention](mha.html) module so we only need to
    write the `get_scores` method.
    """

    def __init__(self, heads: int, d_model: int, dropout_prob: float = 0.1, max_len: int = 5_000):
        super().__init__(heads, d_model, dropout_prob)

        # Pre-calculate the biases
        self.bias = nn.Parameter(get_biases(heads, max_len), requires_grad=False)

    def get_scores(self, query: torch.Tensor, key: torch.Tensor):
        r"""
        ### Calculate attention scores and add attention biases
        """

        # Calculate the standard attention score.
        # It has shape `[query_seq_len, key_seq_len, batch_size, head]`
        scores = super().get_scores(query, key)

        # Number of keys
        key_seq_len = scores.shape[1]
        # Add the biases to scores.
        #
        # $$\mathbf{q}_i \mathbf{K}^\top + m \cdot \big[0, 1, \dots, (i - 1) \big]$$
        #
        # Note that we add biases for all keys (not just upto $i$). We can do this since
        # those extra entries will get removed because of the masking later.
        return scores + self.bias[None, :key_seq_len, None, :]


def _test_slopes():
    """
    Simple test function to see the slopes.
    """
    inspect(get_slopes(8))
    inspect(get_slopes(16))


#
if __name__ == '__main__':
    _test_slopes()
