"""
---
title: Linear Transformers Are Secretly Fast Weight Memory Systems
summary: >
  This is an annotated implementation/tutorial of
  Linear Transformers Are Secretly Fast Weight Memory Systems in PyTorch.
---

# Fast weights transformer

This paper compares linear self attention to fast weight systems and makes
modifications to self attention update rule based on that.
It also introduces a simpler, yet effective kernel function.

## Fast weights

Consider a sequence of inputs $\big\\{x^{(i)}\big\\}^L_{i=1}$ or length $L$
and each step is a vector of size $d_{in}$; i.e. $x \in \mathbb{R}^{d_{in}}$.
The fast weight model generates a weight matrix at each step to produce output
$\big\\{y^{(i)}\big\\}^L_{i=1}$, $y \in \mathbb{R}^{d_{out}}$

\begin{align}
a^{(i)}, b^{(i)} &= \color{orange}{W_a} x^{(i)}, \color{orange}{W_b} x^{(i)} \\
\color{cyan}{W^{(i)}} &= \sigma \Big( \color{cyan}{W^{(i-1)}} + a^{(i)} \otimes b^{(i)} \Big) \\
y^{(i)} &= \color{cyan}{W^{(i)}} x^{(i)}
\end{align}

$\otimes$ is the outer product ($a \otimes b = a b^\top$), where elements of the two vectors are multiplied with each other
to give a matrix.
$\sigma$ is an activation function.
$\color{orange}{W_a}$ and $\color{orange}{W_b}$ are trainable weights (parameters).
$\color{cyan}{W^{(i)}}$ are the fast weights that are generated at each step.

## Linear self-attention

Original transformer self-attention is, (omitting $\frac{1}{d_k}$ for clarity)

\begin{align}
y^{(i)} &= \Big[v^{(1)}, v^{(2)}, ..., v^{(i)}\Big] \text{softmax}
 \bigg(
    \Big[k^{(1)}, k^{(2)}, ..., k^{(i)}\Big] ^\top
    q^{(i)}
 \bigg) \\
 &= \sum^i_{j=1} \frac
 { v^{(j)} \kappa(k^{(j)}, q^{(i)}) }
 { \sum^i_{j'=1} \kappa(k^{(j')}, q^{(i)}) } \\
\end{align}

where $\kappa(k, q) = \text{exp}(k \cdot q)$

The idea behind linearizing self attention is to replace softmax
kernel $\kappa$ with a different kernel $\kappa '$ so that we can calculate the
denominator of the self attention function faster:

$$\kappa '(k, q) = \phi(k)^\top \phi(q)$$

This gives

\begin{align}
y^{(i)} &= \frac
 {\Big( \sum^i_{j=1} v^{(j)} \otimes \phi(k^{(j)} \Big) \phi(q^{(i)}) }
 { \Big( \sum^i_{j'=1} \phi(k^{(j')}) \Big) \phi(q^{(i)}) }
\end{align}

With $W^{(i)} = \sum^i_{j=1} v^{(j)} \otimes \phi(k^{(j)})$ and
$z^{(i)} = \sum^i_{j=1} \phi(k^{(j)})$, we can calculate them efficiently:
\begin{align}
W^{(i)} &= W^{(i-1)} + v^{(i)} \otimes \phi(k^{(i)}) \\
z^{(i)} &= z{(i)} + \phi(k^{(i)}) \\
y^{(i)} &= \frac{1}{z^{(i)} \cdot \phi(q^{(i)})} W^{(i)} \phi(q^{(i)})
\end{align}

This is quite similar to fast weights.
"""

import torch
from torch import nn

from labml_helpers.module import Module
from labml_nn.transformers.feed_forward import FeedForward
from labml_nn.transformers.mha import PrepareForMultiHeadAttention
from labml_nn.utils import clone_module_list


class DPFP(Module):
    def __init__(self, nu: int = 1, eps: float = 1e-6):
        super().__init__()
        self.nu = nu
        self.r = nn.ReLU()
        self.eps = eps

    def __call__(self, x: torch.Tensor):
        x = self.dpfp(x)
        return x / (torch.sum(x, dim=-1, keepdim=True) + self.eps)

    def dpfp(self, x: torch.Tensor):
        x = torch.cat([self.r(x), self.r(-x)], dim=-1)
        x_rolled = torch.cat([x.roll(shifts=j, dims=-1) for j in range(1, self.nu + 1)], dim=-1)
        x_repeat = torch.cat([x] * self.nu, dim=-1)

        return x_repeat * x_rolled


class FastWeightAttention(Module):
    def __init__(self, heads: int, d_model: int, dropout_prob: float, phi: DPFP):
        super().__init__()

        # Number of features per head
        self.d_k = d_model // heads
        #
        self.heads = heads

        # These transform the `query` multi-headed attention.
        self.query = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=False)
        # These transform the `key` and `value` for multi-headed attention.
        self.key = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=False)
        self.value = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=False)

        self.gate = nn.Sequential(PrepareForMultiHeadAttention(d_model, heads, 1, bias=False),
                                  nn.Sigmoid())

        self.phi = phi

        # Output layer
        self.output = nn.Linear(d_model, d_model)
        # Dropout
        self.dropout = nn.Dropout(dropout_prob)

    def __call__(self, x: torch.Tensor):
        seq_len = x.shape[0]
        query = self.phi(self.query(x))
        key = self.phi(self.key(x))
        value = self.value(x)
        beta = self.gate(x)

        weights = key.new_zeros((key.shape[1], key.shape[2], value.shape[3], key.shape[3]))
        outputs = []

        for i in range(seq_len):
            value_existing = torch.einsum('bhvk,bhk->bhv', weights, key[i])

            weights = weights + torch.einsum('bhv,bhk->bhvk', beta[i] * (value[i] - value_existing), key[i])

            x = torch.einsum('bhvk,bhk->bhv', weights, query[i])

            # Concatenate multiple heads
            outputs.append(x.reshape(x.shape[0], -1))

        x = torch.stack(outputs)
        # Output layer
        return self.output(x)


class FastWeightAttentionTransformerLayer(Module):
    def __init__(self, *,
                 d_model: int,
                 attn: FastWeightAttention,
                 feed_forward: FeedForward,
                 dropout_prob: float):
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

    def __call__(self, x: torch.Tensor):
        attn = self.attn(x)
        # Add the self attention results
        x = x + self.dropout(attn)

        # Normalize for feed-forward
        z = self.norm_ff(x)
        # Pass through the feed-forward network
        ff = self.feed_forward(z)
        # Add the feed-forward results back
        x = x + self.dropout(ff)

        #
        return x


class FastWeightAttentionTransformer(Module):
    def __init__(self, layer: FastWeightAttentionTransformerLayer, n_layers: int):
        super().__init__()
        # Make copies of the transformer layer
        self.layers = clone_module_list(layer, n_layers)
        # Final normalization layer
        self.norm = nn.LayerNorm([layer.size])

    def __call__(self, x: torch.Tensor):
        for i, layer in enumerate(self.layers):
            # Get layer output
            x = layer(x)

        # Normalize the output
        return self.norm(x)
