"""
---
title: Long Short-Term Memory (LSTM)
summary: A simple PyTorch implementation/tutorial of Long Short-Term Memory (LSTM) modules.
---

# Long Short-Term Memory (LSTM)

This is a [PyTorch](https://pytorch.org) implementation of Long Short-Term Memory.
"""

from typing import Optional, Tuple

import torch
from torch import nn

from labml_helpers.module import Module


class LSTMCell(Module):
    """
    ## Long Short-Term Memory Cell

    LSTM Cell computes $c$, and $h$. $c$ is like the long-term memory,
    and $h$ is like the short term memory.
    We use the input $x$ and $h$ to update the long term memory.
    In the update, some features of $c$ are cleared with a forget gate $f$,
    and some features $i$ are added through a gate $g$.

    The new short term memory is the $\tanh$ of the long-term memory
    multiplied by the output gate $o$.

    Note that the cell doesn't look at long term memory $c$ when doing the update. It only modifies it.
    Also $c$ never goes through a linear transformation.
    This is what solves vanishing and exploding gradients.

    Here's the update rule.

    \begin{align}
    c_t &= \sigma(f_t) \odot c_{t-1} + \sigma(i_t) \odot \tanh(g_t) \\
    h_t &= \sigma(o_t) \odot \tanh(c_t)
    \end{align}

    $\odot$ stands for element-wise multiplication.

    Intermediate values and gates are computed as linear transformations of the hidden
    state and input.

    \begin{align}
    i_t &= lin_x^i(x_t) + lin_h^i(h_{t-1}) \\
    f_t &= lin_x^f(x_t) + lin_h^f(h_{t-1}) \\
    g_t &= lin_x^g(x_t) + lin_h^g(h_{t-1}) \\
    o_t &= lin_x^o(x_t) + lin_h^o(h_{t-1})
    \end{align}

    """

    def __init__(self, input_size: int, hidden_size: int, layer_norm: bool = False):
        super().__init__()

        # These are the linear layer to transform the `input` and `hidden` vectors.
        # One of them doesn't need a bias since we add the transformations.

        # This combines $lin_x^i$, $lin_x^f$, $lin_x^g$, and $lin_x^o$ transformations.
        self.hidden_lin = nn.Linear(hidden_size, 4 * hidden_size)
        # This combines $lin_h^i$, $lin_h^f$, $lin_h^g$, and $lin_h^o$ transformations.
        self.input_lin = nn.Linear(input_size, 4 * hidden_size, bias=False)

        # Whether to apply layer normalizations.
        #
        # Applying layer normalization gives better results.
        # $i$, $f$, $g$ and $o$ embeddings are normalized and $c_t$ is normalized in
        # $h_t = o_t \odot \tanh(\mathop{LN}(c_t))$
        if layer_norm:
            self.layer_norm = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(4)])
            self.layer_norm_c = nn.LayerNorm(hidden_size)
        else:
            self.layer_norm = nn.ModuleList([nn.Identity() for _ in range(4)])
            self.layer_norm_c = nn.Identity()

    def __call__(self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor):
        # We compute the linear transformations for $i_t$, $f_t$, $g_t$ and $o_t$
        # using the same linear layers.
        ifgo = self.hidden_lin(h) + self.input_lin(x)
        # Each layer produces an output of 4 times the `hidden_size` and we split them
        ifgo = ifgo.chunk(4, dim=-1)

        # Apply layer normalization (not in original paper, but gives better results)
        ifgo = [self.layer_norm[i](ifgo[i]) for i in range(4)]

        # $$i_t, f_t, g_t, o_t$$
        i, f, g, o = ifgo

        # $$c_t = \sigma(f_t) \odot c_{t-1} + \sigma(i_t) \odot \tanh(g_t) $$
        c_next = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)

        # $$h_t = \sigma(o_t) \odot \tanh(c_t)$$
        # Optionally, apply layer norm to $c_t$
        h_next = torch.sigmoid(o) * torch.tanh(self.layer_norm_c(c_next))

        return h_next, c_next


class LSTM(Module):
    """
    ## Multilayer LSTM
    """

    def __init__(self, input_size: int, hidden_size: int, n_layers: int):
        """
        Create a network of `n_layers` of LSTM.
        """

        super().__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        # Create cells for each layer. Note that only the first layer gets the input directly.
        # Rest of the layers get the input from the layer below
        self.cells = nn.ModuleList([LSTMCell(input_size, hidden_size)] +
                                   [LSTMCell(hidden_size, hidden_size) for _ in range(n_layers - 1)])

    def __call__(self, x: torch.Tensor, state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        """
        `x` has shape `[n_steps, batch_size, input_size]` and
        `state` is a tuple of $h$ and $c$, each with a shape of `[batch_size, hidden_size]`.
        """
        n_steps, batch_size = x.shape[:2]

        # Initialize the state if `None`
        if state is None:
            h = [x.new_zeros(batch_size, self.hidden_size) for _ in range(self.n_layers)]
            c = [x.new_zeros(batch_size, self.hidden_size) for _ in range(self.n_layers)]
        else:
            (h, c) = state
            # Reverse stack the tensors to get the states of each layer <br />
            # 📝 You can just work with the tensor itself but this is easier to debug
            h, c = list(torch.unbind(h)), list(torch.unbind(c))

        # Array to collect the outputs of the final layer at each time step.
        out = []
        for t in range(n_steps):
            # Input to the first layer is the input itself
            inp = x[t]
            # Loop through the layers
            for layer in range(self.n_layers):
                # Get the state of the layer
                h[layer], c[layer] = self.cells[layer](inp, h[layer], c[layer])
                # Input to the next layer is the state of this layer
                inp = h[layer]
            # Collect the output $h$ of the final layer
            out.append(h[-1])

        # Stack the outputs and states
        out = torch.stack(out)
        h = torch.stack(h)
        c = torch.stack(c)

        return out, (h, c)
