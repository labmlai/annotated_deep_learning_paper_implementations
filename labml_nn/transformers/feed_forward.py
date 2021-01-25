"""
---
title: Position-wise Feed-Forward Network (FFN)
summary: Documented reusable implementation of the position wise feedforward network.
---

# Position-wise Feed-Forward Network (FFN)

FFN consists of two fully connected layers.
Number of dimensions in the hidden layer $d_{ff}$, is generally set to around
four times that of the token embedding $d_{model}$.
So it is sometime also called the expand-and-contract network.

There is an activation at the hidden layer, which is
usually set to ReLU (Rectified Linear Unit) activation, $$\max(0, x)$$

That is, the FFN function is,
$$FFN(x, W_1, W_2, b_1, b_2) = \max(0, x W_1 + b_1) W_2 + b_2$$
where $W_1$, $W_2$, $b_1$ and $b_2$ are learnable parameters.

Sometimes the
GELU (Gaussian Error Linear Unit) activation is also used instead of ReLU.
$$x \Phi(x)$$ where $\Phi(x) = P(X \le x), X \sim \mathcal{N}(0,1)$
"""

import torch
from torch import nn as nn

from labml_helpers.module import Module


class FeedForward(Module):
    """
    ## Position-wise feed-forward network (FFN) module
    """

    def __init__(self, d_model: int, d_ff: int,
                 dropout: float = 0.1,
                 activation=nn.ReLU(),
                 is_gated: bool = False,
                 bias1: bool = True,
                 bias2: bool = True,
                 bias_gate: bool = True):
        """
        * `d_model` is the number of features in a token embedding
        * `d_ff` is the number of features in the hidden layer of the FFN
        * `dropout` is dropout probability for the hidden layer
        * `is_gated` specifies whether the hidden layer is gated
        * `bias1` specified whether the first fully connected layer should have a learnable bias
        * `bias2` specified whether the second fully connected layer should have a learnable bias
        * `bias_gate` specified whether the fully connected layer for the gate should have a learnable bias
        """
        super().__init__()
        self.layer1 = nn.Linear(d_model, d_ff, bias=bias1)
        self.layer2 = nn.Linear(d_ff, d_model, bias=bias2)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.is_gated = is_gated
        if is_gated:
            self.linear_v = nn.Linear(d_model, d_ff, bias=bias_gate)

    def __call__(self, x: torch.Tensor):
        g = self.activation(self.layer1(x))
        if self.is_gated:
            x = g * self.linear_v(x)
        else:
            x = g
        x = self.dropout(x)
        return self.layer2(x)
