"""
---
title: Rotary Positional Embeddings (RoPE)
summary: >
  Annotated implementation of RoPE from paper
  RoFormer: Enhanced Transformer with Rotary Position Embedding
---

# Rotary Positional Embeddings (RoPE)

This is an implementation of
[Rotary Positional Embeddings (RoPE)](https://papers.labml.ai/paper/2104.09864)
in [PyTorch](https://pytorch.org).

Rotary Positional Embeddings (RoPE) encode position information of tokens
with a rotation matrix that naturally incorporates explicit relative position
dependency.

Here's [the training code](experiment.html) for training a transformer model with RoPE
 on Tiny Shakespeare dataset.

[![View Run](https://img.shields.io/badge/labml-experiment-brightgreen)](https://app.labml.ai/run/1cf508e693be11ecacc98de8b38a61fe)
"""

import torch
from torch import nn

from labml.logger import inspect
from labml_nn.transformers.mha import MultiHeadAttention


class RotaryPositionalEmbeddings(nn.Module):
    """
    ## RoPE module

    Rotary encoding transforms pairs of features by rotating in the 2D plane.
    That is, it organizes the $d$ features as $\frac{d}{2}$ pairs.
    Each pair can be considered a coordinate in a 2D plane, and the encoding will rotate it
    by an angle depending on the position of the token.

    ### For a pair of features

    Let $x^{(1)}_m$ and $x^{(2)}_m$ be two features of the
    key or query of any head at position $m$.
    Or for simplicity assume $x$ has only two features.
    Then the transformation is,

    \begin{align}
    RoPE\big(x^{(1)}_m, x^{(2)}_m, m\big) &=
    \begin{pmatrix}
    \cos m \theta & - \sin m \theta \\
    \sin m \theta & \cos m \theta
    \end{pmatrix}
    \begin{pmatrix}
    x^{(1)}_m \\
    x^{(2)}_m \\
    \end{pmatrix} \\
    &=
    \begin{pmatrix}
    x^{(1)}_m \cos m\theta - x^{(2)}_m \sin m \theta \\
    x^{(2)}_m \cos m\theta + x^{(1)}_m \sin m \theta \\
    \end{pmatrix} \\
    \end{align}

    where $\theta$ is a constant angle. The other pairs of features are transformed similarly.

    ### Attention is relative

    For a pair of features, dot-product attention score between two positions $m$ and $n$ would be

    \begin{align}
    \Big \langle RoPE\big(x^{(1)}_m, x^{(2)}_m, m\big),  RoPE\big(x^{(1)}_n, x^{(2)}_n, n\big) \Big \rangle &= \\
    (x^{(1)}_m \cos m\theta - x^{(2)}_m \sin m \theta)(x^{(1)}_n \cos n\theta - x^{(2)}_n \sin n \theta) &+ \\
    (x^{(2)}_m \cos m\theta + x^{(1)}_m \sin m \theta)(x^{(2)}_n \cos n\theta + x^{(1)}_n \sin n \theta) &= \\
    x^{(1)}_m x^{(1)}_n (\cos m\theta \cos n\theta + \sin m \theta \sin n \theta) &+ \\
    x^{(1)}_m x^{(2)}_n (-\cos m\theta \sin n\theta + \sin m \theta \cos n \theta) &+ \\
    x^{(2)}_m x^{(1)}_n (-\sin m\theta \cos n\theta + \cos m \theta \sin n \theta) &+ \\
    x^{(2)}_m x^{(2)}_n (\sin m\theta \sin n\theta + \cos m \theta \cos n \theta) &= \\

    x^{(1)}_m x^{(1)}_n \cos (m - n) \theta +
    x^{(1)}_m x^{(2)}_n \sin(m - n) \theta &+ \\
    - x^{(2)}_m x^{(1)}_n \sin (m - n) \theta +
    x^{(2)}_m x^{(1)}_n \cos (m - n) \theta &= \\

    \big(x^{(1)}_m \cos (m - n)\theta - x^{(2)}_m \sin (m - n) \theta\big) x^{(1)}_n &+ \\
    \big(x^{(2)}_m \cos (m - n)m\theta + x^{(1)}_m \sin (m - n) \theta\big) x^{(2)}_n  &= \\

    \Big \langle RoPE\big(x^{(1)}_m, x^{(2)}_m, m - n\big),  RoPE\big(x^{(1)}_n, x^{(2)}_n, 0\big) \Big \rangle
    \end{align}

    This shows that for dot-production attention the rotary encodings gives relative attention.

    ### For all features

    The features are grouped into pairs and handled as above. They use a different $\theta$ for each pair.

    The paper suggests using $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
    for the $\frac{d}{2}$ pairs of features.

    We pair feature $i$ with feature $i + \frac{d}{2}$. So for position $m$ we transform

    \begin{align}
    \begin{pmatrix}
    x^{(i)}_m \\
    x^{(i + \frac{d}{2})}_m
    \end{pmatrix}
    \end{align}

    to

    \begin{align}
    \begin{pmatrix}
    x^{(i)}_m \cos m \theta_i - x^{(i + \frac{d}{2})}_m \sin m \theta_i \\
    x^{(i + \frac{d}{2})}_m \cos m\theta_i + x^{(i)}_m \sin m \theta_i \\
    \end{pmatrix} \\
    \end{align}
    """
    def __init__(self, d: int, base: int = 10_000):
        """
        * `d` is the number of features $d$
        * `base` is the constant used for calculating $\Theta$
        """
        super().__init__()
        # $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
        self.theta = nn.Parameter(1. / (base ** (torch.arange(0, d, 2).float() / d)), requires_grad=False)

    def forward(self, x: torch.Tensor):
        """
        * `x` is the Tensor at the head of a key or a query with shape `[seq_len, batch_size, n_heads, d]`
        """
        # Extract the shape
        seq_len, batch_size, n_heads, d = x.shape

        # $\frac{d}{2}$
        d_2 = d // 2

        # Create position indexes `[0, 1, ..., seq_len - 1]`
        seq_idx = torch.arange(seq_len, device=x.device).type_as(self.theta)

        # Calculate the product of position index and $\theta_i$
        idx_theta = torch.einsum('n,d->nd', seq_idx, self.theta)

        # Concatenate so that for row $m$ we have
        # $[m \theta_0, m \theta_1, ..., m \theta_{\frac{d}{2}}, m \theta 0, m \theta 1, ..., m \theta_{\frac{d}{2}}]$
        idx_theta2 = torch.cat([idx_theta, idx_theta], dim=1)

        # Calculate $[-x^{(\frac{d}{2} + 1)}, -x^{(\frac{d}{2} + 2)}, ..., -x^{(d)}, x^{(1)}, x^{(2)}, ..., -x^{(\frac{d}{2})}]$
        neg_half_x = torch.cat([-x[:, :, :, d_2:], x[:, :, :, :d_2]], dim=-1)

        # Calculate
        #
        # \begin{align}
        # \begin{pmatrix}
        # x^{(i)}_m \cos m \theta_i - x^{(i + \frac{d}{2})}_m \sin m \theta_i \\
        # x^{(i + \frac{d}{2})}_m \cos m\theta_i + x^{(i)}_m \sin m \theta_i \\
        # \end{pmatrix} \\
        # \end{align}
        #
        # for $i \in {1, 2, ..., \frac{d}{2}}$
        rx = (x * idx_theta2.cos()[:, None, None, :]) + (neg_half_x * idx_theta2.sin()[:, None, None, :])

        #
        return rx


class RotaryPEMultiHeadAttention(MultiHeadAttention):
    """
    ## Multi-head attention with rotary positional embeddings

    We override [multi-head attention from original transformer](../mha.html).
    """

    def __init__(self, heads: int, d_model: int, dropout_prob: float = 0.1):
        # The linear transformations do not need a bias since we
        # explicitly include it when calculating scores.
        # However having a bias for `value` might make sense.
        super().__init__(heads, d_model, dropout_prob, bias=False)

        # Rotary positional embedding layers
        self.query_rotary_pe = RotaryPositionalEmbeddings(self.d_k)
        self.key_rotary_pe = RotaryPositionalEmbeddings(self.d_k)

    def get_scores(self, query: torch.Tensor, key: torch.Tensor):
        """
        ### Calculate scores between queries and keys
        """

        # Calculate dot-product with RoPE
        return torch.einsum('ibhd,jbhd->ijbh', self.query_rotary_pe(query), self.key_rotary_pe(key))


def _test_rotary():
    """
    Testing RoPE with a simple example
    """
    x = torch.tensor([[1, 2, 3, 4], [4, 5, 6, 7], [7, 8, 9, 10]], dtype=torch.float)
    x = x[:, None, None, :]
    inspect(x)

    rotary_pe = RotaryPositionalEmbeddings(3)
    inspect(rotary_pe(x))


if __name__ == '__main__':
    _test_rotary()
