"""
---
title: Recurrent Highway Networks
summary: A simple PyTorch implementation/tutorial of Recurrent Highway Networks.
---

# Recurrent Highway Networks

This is a [PyTorch](https://pytorch.org) implementation of [Recurrent Highway Networks](https://arxiv.org/abs/1607.03474).
"""
from typing import Optional

import torch
from torch import nn

from labml_helpers.module import Module


class RHNCell(Module):
    """
    ## Recurrent Highway Network Cell

    This implements equations $(6) - (9)$.

    $s_d^t = h_d^t \odot g_d^t + s_{d - 1}^t \odot c_d^t$

    where

    \begin{align}
    h_0^t &= \tanh(lin_{hx}(x) + lin_{hs}(s_D^{t-1})) \\
    g_0^t &= \sigma(lin_{gx}(x) + lin_{gs}^1(s_D^{t-1})) \\
    c_0^t &= \sigma(lin_{cx}(x) + lin_{cs}^1(s_D^{t-1}))
    \end{align}

    and for $0 < d < D$

    \begin{align}
    h_d^t &= \tanh(lin_{hs}^d(s_d^t)) \\
    g_d^t &= \sigma(lin_{gs}^d(s_d^t)) \\
    c_d^t &= \sigma(lin_{cs}^d(s_d^t))
    \end{align}

    $\odot$ stands for element-wise multiplication.

    Here we have made a couple of changes to notations from the paper.
    To avoid confusion with time, gate is represented with $g$,
    which was $t$ in the paper.
    To avoid confusion with multiple layers we use $d$ for depth and $D$ for
    total depth instead of $l$ and $L$ from the paper.

    We have also replaced the weight matrices and bias vectors from the equations with
    linear transforms, because that's how the implementation is going to look like.

    We implement weight tying, as described in paper, $c_d^t = 1 - g_d^t$.
    """

    def __init__(self, input_size: int, hidden_size: int, depth: int):
        """
        `input_size` is the feature length of the input and `hidden_size` is
        the feature length of the cell.
        `depth` is $D$.
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.depth = depth
        # We combine $lin_{hs}$ and $lin_{gs}$, with a single linear layer.
        # We can then split the results to get the $lin_{hs}$ and $lin_{gs}$ components.
        # This is the $lin_{hs}^d$ and $lin_{gs}^d$ for $0 \leq d < D$.
        self.hidden_lin = nn.ModuleList([nn.Linear(hidden_size, 2 * hidden_size) for _ in range(depth)])

        # Similarly we combine $lin_{hx}$ and $lin_{gx}$.
        self.input_lin = nn.Linear(input_size, 2 * hidden_size, bias=False)

    def __call__(self, x: torch.Tensor, s: torch.Tensor):
        """
        `x` has shape `[batch_size, input_size]` and
        `s` has shape `[batch_size, hidden_size]`.
        """

        # Iterate $0 \leq d < D$
        for d in range(self.depth):
            # We calculate the concatenation of linear transforms for $h$ and $g$
            if d == 0:
                # The input is used only when $d$ is $0$.
                hg = self.input_lin(x) + self.hidden_lin[d](s)
            else:
                hg = self.hidden_lin[d](s)

            # Use the first half of `hg` to get $h_d^t$
            # \begin{align}
            # h_0^t &= \tanh(lin_{hx}(x) + lin_{hs}(s_D^{t-1})) \\
            # h_d^t &= \tanh(lin_{hs}^d(s_d^t))
            # \end{align}
            h = torch.tanh(hg[:, :self.hidden_size])
            # Use the second half of `hg` to get $g_d^t$
            # \begin{align}
            # g_0^t &= \sigma(lin_{gx}(x) + lin_{gs}^1(s_D^{t-1})) \\
            # g_d^t &= \sigma(lin_{gs}^d(s_d^t))
            # \end{align}
            g = torch.sigmoid(hg[:, self.hidden_size:])

            s = h * g + s * (1 - g)

        return s


class RHN(Module):
    """
    ## Multilayer Recurrent Highway Network
    """

    def __init__(self, input_size: int, hidden_size: int, depth: int, n_layers: int):
        """
        Create a network of `n_layers` of recurrent highway network layers, each with depth `depth`, $D$.
        """

        super().__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        # Create cells for each layer. Note that only the first layer gets the input directly.
        # Rest of the layers get the input from the layer below
        self.cells = nn.ModuleList([RHNCell(input_size, hidden_size, depth)] +
                                   [RHNCell(hidden_size, hidden_size, depth) for _ in range(n_layers - 1)])

    def __call__(self, x: torch.Tensor, state: Optional[torch.Tensor] = None):
        """
        `x` has shape `[seq_len, batch_size, input_size]` and
        `state` has shape `[batch_size, hidden_size]`.
        """
        time_steps, batch_size = x.shape[:2]

        # Initialize the state if `None`
        if state is None:
            s = [x.new_zeros(batch_size, self.hidden_size) for _ in range(self.n_layers)]
        else:
            # Reverse stack the state to get the state of each layer <br />
            # 📝 You can just work with the tensor itself but this is easier to debug
            s = torch.unbind(state)

        # Array to collect the outputs of the final layer at each time step.
        out = []

        # Run through the network for each time step
        for t in range(time_steps):
            # Input to the first layer is the input itself
            inp = x[t]
            # Loop through the layers
            for layer in range(self.n_layers):
                # Get the state of the layer
                s[layer] = self.cells[layer](inp, s[layer])
                # Input to the next layer is the state of this layer
                inp = s[layer]
            # Collect the output of the final layer
            out.append(s[-1])

        # Stack the outputs and states
        out = torch.stack(out)
        s = torch.stack(s)

        return out, s
