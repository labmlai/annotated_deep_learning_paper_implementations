"""
---
title: Rotary Positional Embeddings with Relative distance (RoPER)
summary: >
  This is an implementation of RoPER which adds relative distance information to embeddings on
  top of RoPE introduced in RoFormer: Enhanced Transformer with Rotary Position Embedding
---

*RoPER is work by [Georges Harik (@gharik)](https://twitter.com/gharik),
and this implementation is based on his original code.*

# Rotary Positional Embeddings with Relative distance (RoPER)

[Rotary Positional Embeddings (RoPE)](https://papers.labml.ai/paper/2104.09864) includes
relative positions in attention score calculation.
However, the embeddings themselves do not get any positional information
, [except what it can get implicitly from causal attention](https://papers.labml.ai/paper/2c364684b15b11ecac827bce58715ee7).

RoPER adds relative positional information explicitly to value embeddings.
Specifically, it adds the relative positions of the tokens it paid attention to.
We use same rotary positional embeddings to rotate the values in attention,
Then, after taking the weighted sum,
 we rotate the final in the opposite direction.
Which is equivalent to rotating each of the values (before attention) relative to the current position.

Here's [the training code](experiment.html) for training a transformer model with RoPER
 on an arithmetic addition where we can see significant improvement over RoPE.

### Relative distances in embeddings

For any head, let $a_{n,i}$ be the attention from position $n$ to position $i$,
and $v_i$ be the value embeddings at position $i$. Let's denote individual features
as $v^{(1)}_i, v^{(2)}_i, \dots$.

Normally, we would take the weight sum of value embeddings

$$o^{(j)}_n = \sum_i a_{n,i} v^{(j)}_i$$

This doesn't explicitly add any distance information about the positions $i$ to final
result $o^{(j)}_n$.

RoPER pairs features like RoPE and transform them.
For a pair $v^{(1)}_m$ and $v^{(2)}_m$ it transforms them by
 $RoPE\big(v^{(1)}_m, v^{(2)}_m, m\big)$.
Let us donate the transformed features with $\hat{v}^{(1)}_m, \hat{v}^{(2)}_m$.
Then it rotates the weighted sum $\hat{o}^{(j)}_n$ in the the reverse direction with
 $RoPE\big(\hat{o}^{(1)}_n, \hat{o}^{(2)}_n, -n\big)$.
*Note the *$-n$.

Note that,

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

Final output after with the transformations is,

\begin{align}
RoPE\big(\hat{o}^{(1)}_n, \hat{o}^{(2)}_n, -n\big) &= \\
\begin{pmatrix}
\hat{o}^{(1)}_n \cos n\theta + \hat{o}^{(2)}_n \sin n \theta \\
\hat{o}^{(2)}_n \cos n\theta - \hat{o}^{(1)}_n \sin n \theta \\
\end{pmatrix} \\
\end{align}

*Note that *$\sin (-n \theta) = -\sin n \theta$.

Let's expand the first term $\hat{o}^{(1)}_n \cos n\theta + \hat{o}^{(2)}_n \sin n \theta$,

\begin{align}
\hat{o}^{(1)}_n \cos n\theta + \hat{o}^{(2)}_n \sin n \theta &= \\
\sum_i a_{n,i} \hat{v}^{(1)}_i \cos n\theta + \sum_i a_{n,i} \hat{v}^{(2)}_i \sin n \theta &= \\

\sum_i a_{n,i} \Big( v^{(1)}_i \cos i\theta - v^{(2)}_i \sin i \theta \Big) \cos n\theta &+ \\
\sum_i a_{n,i} \Big( v^{(2)}_i \cos i\theta + v^{(1)}_i \sin i \theta \Big) \sin m \theta &= \\

\sum_i a_{n,i} v^{(1)}_i \Big( \cos i\theta \cos n\theta + \sin i \theta \sin n \theta \Big) &+ \\
\sum_i a_{n,i} v^{(2)}_i \Big( \cos i\theta \sin n\theta - \sin i \theta \cos n \theta \Big) &= \\

\sum_i a_{n,i} v^{(1)}_i \cos (i - n) \theta - \sum_i a_{n,i} v^{(2)}_i \sin (i - n) \theta &= \\

\sum_i a_{n,i} v^{(1)}_i \cos (i - n) \theta - \sum_i a_{n,i} v^{(2)}_i \sin (i - n) \theta
\end{align}

Simiarly we can show the second term is equal to,

$$\sum_i a_{n,i} v^{(1)}_i \cos (i - n) \theta + \sum_i a_{n,i} v^{(2)}_i \sin (i - n) \theta$$

Which gives,

\begin{align}
RoPE\big(\hat{o}^{(1)}_n, \hat{o}^{(2)}_n, -n\big) &= \\
\begin{pmatrix}
\sum_i a_{n,i} v^{(1)}_i \cos (i - n) \theta - \sum_i a_{n,i} v^{(2)}_i \sin (i - n) \theta \\
\sum_i a_{n,i} v^{(1)}_i \cos (i - n) \theta + \sum_i a_{n,i} v^{(2)}_i \sin (i - n) \theta \\
\end{pmatrix} &= \\
\sum_i a_{n,i} RoPE \big (v^{(1)}_i, v^{(1)}_i, (i - n) \theta \big)
\end{align}

That is, the weighted average of values rotated relative to current position.

[Here's an experiment](arithmetic_experiment.html) that uses RoPER on an arthmetic addition task.
"""

from typing import Optional

import torch

from labml_nn.transformers.rope import RotaryPositionalEmbeddings, RotaryPEMultiHeadAttention


class ReverseRotaryPositionalEmbeddings(RotaryPositionalEmbeddings):
    """
    ## RoPE module that rotates in the opposite direction

    This inherits from [RoPE rotation implementation](../index.html) and changes the direction.
    """

    def forward(self, x: torch.Tensor):
        """
        * `x` is the Tensor at the head of a key or a query with shape `[seq_len, batch_size, n_heads, d]`
        """
        # Cache $\cos$ and $\sin$ values
        self._build_cache(x)

        # Split the features, we can choose to apply rotary embeddings only to a partial set of features.
        x_rope, x_pass = x[..., :self.d], x[..., self.d:]

        # Calculate
        # $[-x^{(\frac{d}{2} + 1)}, -x^{(\frac{d}{2} + 2)}, ..., -x^{(d)}, x^{(1)}, x^{(2)}, ..., x^{(\frac{d}{2})}]$
        neg_half_x = self._neg_half(x_rope)

        # Calculate
        #
        # \begin{align}
        # \begin{pmatrix}
        # x^{(i)}_m \cos -m \theta_i - x^{(i + \frac{d}{2})}_m \sin -m \theta_i \\
        # x^{(i + \frac{d}{2})}_m \cos -m\theta_i + x^{(i)}_m \sin -m \theta_i \\
        # \end{pmatrix} = \\
        # \begin{pmatrix}
        # x^{(i)}_m \cos m \theta_i + x^{(i + \frac{d}{2})}_m \sin m \theta_i \\
        # x^{(i + \frac{d}{2})}_m \cos m\theta_i - x^{(i)}_m \sin m \theta_i \\
        # \end{pmatrix} \\
        # \end{align}
        #
        # for $i \in {1, 2, ..., \frac{d}{2}}$
        x_rope = (x_rope * self.cos_cached[:x.shape[0]]) - (neg_half_x * self.sin_cached[:x.shape[0]])

        #
        return torch.cat((x_rope, x_pass), dim=-1)


class RotaryValuePEMultiHeadAttention(RotaryPEMultiHeadAttention):
    """
    ## Multi-head attention with rotary positional embeddings

    We override [multi-head attention from original transformer](../mha.html).
    """

    def __init__(self, heads: int, d_model: int,
                 rope_percentage: float = 0.5, rope_value_percentage: float = 0.5,
                 dropout_prob: float = 0.0):
        super().__init__(heads, d_model, rope_percentage, dropout_prob)

        # Rotary positional embedding layers
        d_rope_value = int(self.d_k * rope_value_percentage)

        self.value_rotary_pe = RotaryPositionalEmbeddings(d_rope_value)
        self.value_reverse_rotary_pe = ReverseRotaryPositionalEmbeddings(d_rope_value)

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

        # Apply mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # $softmax$ attention along the key sequence dimension
        # $\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_k}}\Bigg)$
        attn = self.softmax(scores)

        # Apply dropout
        attn = self.dropout(attn)

        # Rotate value embeddings before taking the weighted sum so that they contain positional information
        value = self.value_rotary_pe(value)

        # Multiply by values
        # $$\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_k}}\Bigg)V$$
        x = torch.einsum("ijbh,jbhd->ibhd", attn, self.value_rotary_pe(value))

        # Rotate in the opposite direction so that each embedding hold the relative positions
        x = self.value_reverse_rotary_pe(x)

        # Save attentions for any other calculations
        self.attn = attn.detach()

        # Concatenate multiple heads
        x = x.reshape(seq_len, batch_size, -1)

        # Output layer
        return self.output(x)
