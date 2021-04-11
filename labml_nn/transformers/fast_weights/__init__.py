"""
---
title: Linear Transformers Are Secretly Fast Weight Memory Systems
summary: >
  This is an annotated implementation/tutorial of
  Linear Transformers Are Secretly Fast Weight Memory Systems in PyTorch.
---

# Fast weights transformer

The paper
[Linear Transformers Are Secretly Fast Weight Memory Systems in PyTorch](https://arxiv.org/abs/2102.11174)
finds similarities between linear self-attention and fast weight systems
and makes modifications to self-attention update rule based on that.
It also introduces a simpler, yet effective kernel function.

*The authors have provided an [official implementation](https://github.com/ischlag/fast-weight-transformers)
of the paper including other variants they compare with in the paper.*

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

$$\kappa '(k, q) = \color{lightgreen}{\phi(k)}^\top \color{lightgreen}{\phi(q)}$$

This gives

\begin{align}
y^{(i)} &= \frac
 {\Big( \sum^i_{j=1} v^{(j)} \otimes \color{lightgreen}{\phi(k^{(j)})} \Big)
  \color{lightgreen}{\phi(q^{(i)})} }
 { \Big( \sum^i_{j'=1}
   \color{lightgreen}{\phi(k^{(j')})} \Big)
    \color{lightgreen}{\phi(q^{(i)})} }
\end{align}

With $\color{cyan}{W^{(i)}} = \sum^i_{j=1} v^{(j)} \otimes \phi(k^{(j)})$ and
$z^{(i)} = \sum^i_{j=1} \color{lightgreen}{\phi(k^{(j)})}$, we can calculate them efficiently:
\begin{align}
\color{cyan}{W^{(i)}} &= \color{cyan}{W^{(i-1)}} + v^{(i)} \otimes \color{lightgreen}{\phi(k^{(i)})} \\
z^{(i)} &= z{(i)} + \color{lightgreen}{\phi(k^{(i)})} \\
y^{(i)} &= \frac{1}{z^{(i)} \cdot \color{lightgreen}{\phi(q^{(i)})}}
    W^{(i)} \color{lightgreen}{\phi(q^{(i)})}
\end{align}

This is quite similar to fast weights.

The paper introduces a new linear attention projection function $\color{lightgreen}{\phi}$
a new update rule for $\color{cyan}{W^{(i)}} = f(\color{cyan}{W^{(i-1)}})$ and change the normalization
$\frac{1}{z^{(i)} \cdot \color{lightgreen}{\phi(q^{(i)})}}$

Here are [the training code](experiment.html) and a notebook for training a fast weights
 transformer on the Tiny Shakespeare dataset.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lab-ml/nn/blob/master/labml_nn/transformers/fast_weights/experiment.ipynb)
[![View Run](https://img.shields.io/badge/labml-experiment-brightgreen)](https://app.labml.ai/run/928aadc0846c11eb85710242ac1c0002)
"""

import torch
from torch import nn

from labml_helpers.module import Module
from labml_nn.transformers.feed_forward import FeedForward
from labml_nn.transformers.mha import PrepareForMultiHeadAttention
from labml_nn.utils import clone_module_list


class DPFP(Module):
    """
    ## Deterministic Parameter Free Project (DPFP)

    This is the new projection function $\color{lightgreen}{\phi}$ introduced in the paper.
    DPFP projects $k$ of dimensionality $d_{key}$ to dimensionality $d_{dot} = 2 d_{key} \nu$,
    where $\nu \in \\{1, 2, ..., 2 d_{key} - 1 \\}$ is a hyper-parameter.

    $$\color{lightgreen}{\phi_{2 d_{key} (i - 1)  + j}(k)}
     = \text{ReLU}\Big(\big[k, -k\big]\Big)_{j}
                        \text{ReLU}\Big(\big[k, -k\big]\Big)_{i + j}$$

    where $\big[k, -k\big]$ is the concatenation of $k$ and $-k$ to give a vector of
    size $2 d_{key}$, $i \in \\{1, 2, ..., \nu \\}$, and $j \in \\{1, 2, ..., 2 d_{key}\\}$.
    $x_i$ is the $i$-th element of vector $x$ and is rolled around if
    $i$ is larger than the number of elements in $x$.

    Basically, it creates a new vector by multiplying elements of $[k, -k]$ shifted by $i$.

    This produces projections that are sparse (only a few elements of $phi$ are non-zero) and
    orthogonal ($\color{lightgreen}{\phi(k^{(i)})} \cdot \color{lightgreen}{\phi(k^{(j)})}
     \approx 0$ for most $i, j$
    unless $k^{(i)}$ and $k^{(j)}$ are very similar.

    ### Normalization

    Paper introduces a simple normalization for $\color{lightgreen}{\phi}$,

    $$\color{lightgreen}{\phi '(k)} =
     \frac{\color{lightgreen}{\phi(k)}}{\sum^{d_{dot}}_{j=1} \color{lightgreen}{\phi(k)_j}}$$

    *Check the paper for derivation.*
    """

    def __init__(self, nu: int = 1, eps: float = 1e-6):
        """
        * `nu` is the hyper-parameter $\nu$.
        * `eps` is the small value used to make sure there is no division-by-zero when normalizing.
        """
        super().__init__()
        self.nu = nu
        self.relu = nn.ReLU()
        self.eps = eps

    def __call__(self, k: torch.Tensor):
        # Get $\color{lightgreen}{\phi(k)}$
        k = self.dpfp(k)
        # Normalize by $\sum^{d_{dot}}_{j=1} \color{lightgreen}{\phi(k)_j}$
        return k / (torch.sum(k, dim=-1, keepdim=True) + self.eps)

    def dpfp(self, k: torch.Tensor):
        """
        $$\color{lightgreen}{\phi(k)}$$
        """
        # $x = \text{ReLU}\Big(\big[k, -k\big]\Big)$
        x = self.relu(torch.cat([k, -k], dim=-1))
        # Shift and roll by $i \in \\{1, 2, ..., \nu \\}$,
        # to get $$x'_{i,j} = \text{ReLU}\Big(\big[k, -k\big]\Big)_{i+j}$$
        x_rolled = [x.roll(shifts=i, dims=-1) for i in range(1, self.nu + 1)]
        # Concatenate to get
        # $$x'_{2 d_{key} (i - 1)  + j} = \text{ReLU}\Big(\big[k, -k\big]\Big)_{i+j}$$
        x_rolled = torch.cat(x_rolled, dim=-1)
        # Concatenate copies of $x$
        x_repeat = torch.cat([x] * self.nu, dim=-1)

        # Multiply them,
        # $$\color{lightgreen}{\phi_{2 d_{key} (i - 1)  + j}(k)}
        # = \text{ReLU}\Big(\big[k, -k\big]\Big)_{j}
        #                         \text{ReLU}\Big(\big[k, -k\big]\Big)_{i + j}$$
        return x_repeat * x_rolled


class FastWeightsAttention(Module):
    """
    ## Fast Weights Attention

    The paper introduces a new update rule for calculating $\color{cyan}{W^{(i)}}$.
    The model first retrieves the current value
    $\bar{v}^{(i)}$ paired with the key $k^{(i)}$.
    Then stores a combination $v^{(i)}_{new}$
    of the retrieved value $\bar{v}^{̄(i)}$ and the input $v^{(i)}$.

    \begin{align}
    k^{(i)}, v^{(i)}, q^{(i)} &=
     \color{orange}{W_k} x^{(i)}, \color{orange}{W_v} x^{(i)}, \color{orange}{W_q} x^{(i)} \\
    \bar{v}^{(i)} &= \color{cyan}{W^{(i-1)}} \color{lightgreen}{\phi'(k^{(i)})} \\
    \beta^{(i)} &= \sigma \Big(\color{orange}{W_\beta} x^{(i)} \Big) \\
    v^{(i)}_{new} &= \beta^{(i)} v^{(i)} + \Big(1 - \beta^{(i)} \Big) \bar{v}^{(i)} \\
    \color{cyan}{W^{(i)}}
     &= \color{cyan}{W^{(i-1)}} + v^{(i)}_{new} \otimes \color{lightgreen}{\phi'(k^{(i)})} \\
     &= \color{cyan}{W^{(i-1)}} +
     \beta^{(i)} \Big( v^{(i)} - \bar{v}^{(i)} \Big ) \otimes \color{lightgreen}{\phi'(k^{(i)})} \\
    y^{(i)} &= \color{cyan}{W^{(i)}} \color{lightgreen}{\phi'(q^{(i)})}
    \end{align}

    where $\color{orange}{W_\beta}$ is a trainable parameter and $\sigma$ is the sigmoid function.

    Note that we don't need the normalization term $z$ because $\color{lightgreen}{\phi'}$ is normalized.
    """

    def __init__(self, heads: int, d_model: int, dropout_prob: float, phi: DPFP):
        super().__init__()

        # Number of features per head $d_k$
        self.d_k = d_model // heads
        # Number of heads
        self.heads = heads

        # These transform the `query`, `key` and `value` multi-headed attention.
        self.query = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=False)
        self.key = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=False)
        self.value = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=False)

        # Interpolation weight function $\sigma \Big(\color{orange}{W_\beta} x^{(i)} \Big)$ for each head
        self.interpolation_weight = nn.Sequential(
            PrepareForMultiHeadAttention(d_model, heads, 1, bias=False),
            nn.Sigmoid()
        )

        # $\color{lightgreen}{\phi'}$
        self.phi = phi

        # Output layer
        self.output = nn.Linear(d_model, d_model)
        # Dropout
        self.dropout = nn.Dropout(dropout_prob)

    def __call__(self, x: torch.Tensor):
        # Get the number of steps $L$
        seq_len = x.shape[0]
        # $\color{lightgreen}{\phi'(q^{(i)})}$ for all steps and heads
        query = self.phi(self.query(x))
        # $\color{lightgreen}{\phi'(k^{(i)})}$ for all steps and heads
        key = self.phi(self.key(x))
        # $v^{(i)}$ for all steps and heads
        value = self.value(x)
        # $\beta^{(i)}$ for all steps and heads
        beta = self.interpolation_weight(x)

        # $\color{cyan}{W^{(0)}}$
        weights = key.new_zeros((key.shape[1], key.shape[2], value.shape[3], key.shape[3]))
        # List to store outputs $y^{(i)}$
        outputs = []

        # Iterate through steps
        for i in range(seq_len):
            # $$\bar{v}^{(i)} = \color{cyan}{W^{(i-1)}} \color{lightgreen}{\phi'(k^{(i)})}$$
            value_existing = torch.einsum('bhvk,bhk->bhv', weights, key[i])

            # $$\color{cyan}{W^{(i)}}
            #      = \color{cyan}{W^{(i-1)}} +
            #      \beta^{(i)} \Big( v^{(i)} - \bar{v}^{(i)} \Big ) \otimes \color{lightgreen}{\phi'(k^{(i)})}$$
            weights = weights + torch.einsum('bhv,bhk->bhvk', beta[i] * (value[i] - value_existing), key[i])

            # $$y^{(i)} = \color{cyan}{W^{(i)}} \color{lightgreen}{\phi'(q^{(i)})}$$
            y = torch.einsum('bhvk,bhk->bhv', weights, query[i])

            # Merge multiple heads and append to `outputs`
            outputs.append(y.reshape(y.shape[0], -1))

        # Stack outputs at each step into a single tensor
        x = torch.stack(outputs)

        # Output layer
        return self.output(x)


class FastWeightsAttentionTransformerLayer(Module):
    """
    This is a general transformer layer that combines self attention and feedforward network.
    """
    def __init__(self, *,
                 d_model: int,
                 attn: FastWeightsAttention,
                 feed_forward: FeedForward,
                 dropout_prob: float):
        super().__init__()
        # Transformer size $d_{model}$
        self.size = d_model
        # Fast weights attention module
        self.attn = attn
        # Feed-forward network
        self.feed_forward = feed_forward
        # Dropout layer
        self.dropout = nn.Dropout(dropout_prob)

        # Normalization layers
        self.norm_self_attn = nn.LayerNorm([d_model])
        self.norm_ff = nn.LayerNorm([d_model])

    def __call__(self, x: torch.Tensor):
        # Calculate fast weights self attention
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


class FastWeightsAttentionTransformer(Module):
    """
    This is a general transformer module with multiple transformer layers
    """
    def __init__(self, layer: FastWeightsAttentionTransformerLayer, n_layers: int):
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
