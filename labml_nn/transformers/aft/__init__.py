"""
---
title: An Attention Free Transformer
summary: >
  This is an annotated implementation/tutorial the AFT (Attention Free Transformer) in PyTorch.
---

# An Attention Free Transformer

This is a [PyTorch](https://pytorch.org) implementation of the paper
[An Attention Free Transformer](https://papers.labml.ai/paper/2105.14103).

This paper replaces the [self-attention layer](../mha.html) with a new efficient operation,
that has memory complexity of $\mathcal{O}(Td)$, where $T$ is the sequence length
and $d$ is the dimensionality of embeddings.

The paper introduces AFT along with AFT-local and AFT-conv.
Here we have implemented AFT-local which pays attention to closeby tokens
in an autoregressive model.

## Attention Free Transformer

AFT (similar to [MHA](../mha.html)) first transforms the embeddings $X$ into
query $Q = XW^Q$, key $K = XW^K$ and value $V = XW^V$ tensors with learned weights.
The output for each position $t \in [1, T]$ is calculated with the following operation.

$$Y_t = \sigma(Q_t) \odot
 \frac{\sum_{t'=1}^T \exp(K_{t'} + w_{t,t'}) \odot V_{t'}}
 {\sum_{t'=1}^T \exp(K_{t'} + w_{t,t'})}$$

, where $\odot$ is element-wise product, $\sigma$ is a non-linearity (sigmoid) and
$w \in \mathbb{R}^{T \times T}$ is a learned matrix of pair-wise position biases.

This means that we take the weighted average of values
and multiply them by the query. This eliminates the need to calculate the $T \times T$ attention
matrix that [MHA](../mha.html) requires, and therefore reduce the memory requirement.

## AFT Local

AFT Local only apply learned pair-wise position biases locally:

\begin{align}
w'_{t,t'} =
\begin{cases}
w_{t,t'},  & \text{for $\lvert t-t' \rvert \lt s$} \\
0, & \text{otherwise}
\end{cases}
\end{align}

, where $s \le T$ is the local window size.

Although $w'_{t,t'}$ is $0$ outside the local window the AFT operation still uses key-value pairs from
other areas. This is different from local transformers where embeddings outside the local window are
 completely not visible.

Here is [the training code](experiment.html) for a AFT Local model.

[![View Run](https://img.shields.io/badge/labml-experiment-brightgreen)](https://app.labml.ai/run/6348e504c3a511eba9529daa283fb495)
"""

from typing import Optional

import torch
from torch import nn

from labml_helpers.module import Module


class AFTLocalAutoregressive(Module):
    """
    ### AFT Local Operation for Auto-Regression

    This is an implementation of AFT Local for auto-regression, where $Y_t$
    only has visibility to tokens before $t$:

    $$Y_t = \sigma(Q_t) \odot
     \frac{\sum_{t'=1}^t \exp(K_{t'} + w_{t,t'}) \odot V_{t'}}
     {\sum_{t'=1}^t \exp(K_{t'} + w_{t,t'})}$$
    """

    def __init__(self, d_model: int, seq_len: int, s: int, bias: bool = True):
        """
        * `d_model` is the number of features in the `query`, `key` and `value` vectors.
        * `seq_len` is $T$
        * `s` is the local window size $s$
        * `bias` is whether to have a bias parameter for transformations for $Q$, $K$ and $V$.
        """

        super().__init__()

        # Local window size $s$
        self.s = s
        # These transform the `query`, `key` and `value` vectors.
        self.query = nn.Linear(d_model, d_model, bias=bias)
        self.key = nn.Linear(d_model, d_model, bias=bias)
        self.value = nn.Linear(d_model, d_model, bias=bias)
        # Pair-wise positional biases $w \in \mathbb{R}^{T \times T}$
        self.pos_bias = nn.Parameter(torch.zeros(seq_len, seq_len), requires_grad=True)
        # Activation $\sigma$
        self.activation = nn.Sigmoid()
        # Output layer
        self.output = nn.Linear(d_model, d_model)

    def forward(self, *,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None):
        """
        `query`, `key` and `value` are the tensors that store
        collection of token embeddings for  *query*, *key* and *value*.
        They have shape `[seq_len, batch_size, d_model]`.

        `mask` should be `None`. We keep this parameter so that we can use this as an
         drop in replacement for [MHA](../mha.html).
        """

        # `query`, `key` and `value`  have shape `[seq_len, batch_size, d_model]`
        seq_len, _, _ = query.shape

        query = self.query(query)
        key = self.key(key)
        value = self.value(value)

        # We subtract $\max(K_{t'} + w_{t,t'})$ before calculating the exponents to stabilize
        # the softmax calculation.
        #
        # If $x_i$ is large $\exp(x_i)$ becomes huge and the computation of
        # $\frac{\sum\exp(x_i)y_i}{\sum\exp(x_i)}$becomes unstable.
        # Subtracting a constant before calculating the exponent from numerator and denominator will cancel out.
        # and can help stabilize the computation.
        # So we subtract $\max(x_i)$ to stabilize the computation.
        #
        # Here the maximum is the higher of $\max(K_{t'} + w_{t,t'})$ and $\max(K_{t'})$
        max_logit = torch.max(
            # $\max(K_{t'})$
            key.max(dim=0)[0],
            # $\max(K_{t'} + w_{t,t'})$
            (key + self.pos_bias[:seq_len, :seq_len].max(dim=0)[0].view(-1, 1, 1)).max(dim=0)[0]
        )[0]

        # \begin{align}
        # Y_t &= \sigma(Q_t) \odot
        #      \frac{\sum_{t'=1}^t \exp(K_{t'} + w_{t,t'}) \odot V_{t'}}
        #      {\sum_{t'=1}^t \exp(K_{t'} + w_{t,t'})} \\
        # &= \sigma(Q_t) \odot
        #      \frac{\sum_{t'=1}^{t-s} \exp(K_{t'}) \odot V_{t'} + \sum_{t'=t-s+1}^t \exp(K_{t'} + w_{t,t'}) \odot V_{t'}}
        #      {\sum_{t'=1}^{t-s} \exp(K_{t'}) + \sum_{t'=t-s+1}^t \exp(K_{t'} + w_{t,t'})} \\
        # \end{align}
        #
        # since
        # \begin{align}
        # w'_{t,t'} =
        # \begin{cases}
        # w_{t,t'},  & \text{for $\lvert t-t' \rvert \lt s$} \\
        # 0, & \text{otherwise}
        # \end{cases}
        # \end{align}
        #

        # The numerator part $\sum_{t'=1}^{t-s} \exp(K_{t'}) \odot V_{t'}$
        num = key.new_zeros(key.shape[1:])
        # The denominator part $\sum_{t'=1}^{t-s} \exp(K_{t'})$
        den = key.new_zeros(key.shape[1:])
        # Output $Y$
        y = key.new_zeros(key.shape)
        # Iterate $t \in [0, T]$
        for t in range(seq_len):
            # $t - s + 1$
            f = t - self.s + 1
            # This actually mean $t - s \ge 1$ since we are indexing from $1$ in the math equations and
            # indexing from $0$ in code
            if f >= 1:
                # $\exp(K_{t-s}$
                exp_l = torch.exp(key[f - 1] - max_logit)
                # Update numerator and denominator parts
                num = num + exp_l * value[f - 1]
                den = den + exp_l
            # Start from the beginning if the local window size falls beyond
            f = max(0, f)
            # $\exp(K_{t'} + w_{t,t'})$
            exp_l = torch.exp(key[f: t + 1] + self.pos_bias[t, f: t + 1].view(-1, 1, 1) - max_logit.squeeze(0))
            # Numerator
            # $$\sum_{t'=1}^{t-s} \exp(K_{t'}) \odot V_{t'} + \sum_{t'=t-s+1}^t \exp(K_{t'} + w_{t,t'}) \odot V_{t'}$$
            n = num + (exp_l * value[f: t + 1]).sum(dim=0)
            # Denominator
            # $$\sum_{t'=1}^{t-s} \exp(K_{t'}) + \sum_{t'=t-s+1}^t \exp(K_{t'} + w_{t,t'})$$
            d = den + exp_l.sum(dim=0)
            # $$Y_t = \sigma(Q_t) \odot
            #      \frac{\sum_{t'=1}^{t-s} \exp(K_{t'}) \odot V_{t'} + \sum_{t'=t-s+1}^t \exp(K_{t'} + w_{t,t'}) \odot V_{t'}}
            #      {\sum_{t'=1}^{t-s} \exp(K_{t'}) + \sum_{t'=t-s+1}^t \exp(K_{t'} + w_{t,t'})} $$
            y[t] = self.activation(query[t]) * n / d

        # Output layer
        return self.output(y)
