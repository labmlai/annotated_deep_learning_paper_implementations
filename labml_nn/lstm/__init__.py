"""
# Long Short-Term Memory (LSTM)
"""

from typing import Optional, Tuple

import torch
from labml_helpers.module import Module
from torch import nn


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

    Note that the cell doesn't look at long term memory $c$ when doing the update
    for the update. It only modifies it.
    Also $c$ never goes through a linear transformation.
    This is what solves vanishing and exploding gradients.

    Here's the update rule.

    \begin{align}
    c_t = f_t \odot c_{t-1} + i_t \odot g_t \\
    h_t = o_t \odot \tanh(c_t)
    \end{align}

    $\odot$ stands for element-wise multiplication.

    Here's how input intermediate values and gates are computed.

    \begin{align}
    i_t &= \sigma\big(lin_{xi}(x_t) + lin_{hi}(h_{t-1})\big) \\
    f_t &= \sigma\big(lin_{xf}(x_t) + lin_{hf}(h_{t-1})\big) \\
    g_t &= \tanh\big(lin_{xg}(x_t) + lin_{hg}(h_{t-1})\big) \\
    o_t &= \sigma\big(lin_{xo}(x_t) + lin_{ho}(h_{t-1})\big)
    \end{align}

    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()

        self.hidden_size = hidden_size

        # These are the linear layer to transform the `input` and `hidden` vectors.
        # One of them doesn't need a bias since we add the transformations.

        # This combines $lin_{xi}$, $lin_{xf}$, $lin_{xg}$, and $lin_{xo}$ transformations.
        self.hidden_lin = nn.Linear(hidden_size, 4 * hidden_size)
        # This combines $lin_{hi}$, $lin_{hf}$, $lin_{hg}$, and $lin_{ho}$ transformations.
        self.input_lin = nn.Linear(input_size, 4 * hidden_size, bias=False)

    def __call__(self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor):
        # We compute the linear transformations for $i_t$, $f_t$, $g_t$ and $o_t$
        # using the same linear layers.
        # Each layer produces an output of 4 times the `hidden_size` and we split them later
        ifgo = self.hidden_lin(h) + self.input_lin(x)

        # $$i_t = \sigma\big(lin_{xi}(x_t) + lin_{hi}(h_{t-1})\big)$$
        i = torch.sigmoid(ifgo[:, :self.hidden_size])
        # $$f_t = \sigma\big(lin_{xf}(x_t) + lin_{hf}(h_{t-1})\big)$$
        f = torch.sigmoid(ifgo[:, self.hidden_size:self.hidden_size * 2])
        # $$g_t = \tanh\big(lin_{xg}(x_t) + lin_{hg}(h_{t-1})\big)$$
        g = torch.tanh(ifgo[:, self.hidden_size * 2:self.hidden_size * 3])
        # $$o_t = \sigma\big(lin_{xo}(x_t) + lin_{ho}(h_{t-1})\big)$$
        o = torch.sigmoid(ifgo[:, self.hidden_size * 3:self.hidden_size * 4])

        # $$c_t = f_t \odot c_{t-1} + i_t \odot g_t$$
        c_next = f * c + i * g

        # $$h_t = o_t \odot \tanh(c_t)$$
        h_next = o * torch.tanh(c_next)

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
        `x` has shape `[seq_len, batch_size, input_size]` and
        `state` is a tuple of $h$ and $c$, each with a shape of `[batch_size, hidden_size]`.
        """
        time_steps, batch_size = x.shape[:2]

        # Initialize the state if `None`
        if state is None:
            h = [x.new_zeros(batch_size, self.hidden_size) for _ in range(self.n_layers)]
            c = [x.new_zeros(batch_size, self.hidden_size) for _ in range(self.n_layers)]
        else:
            (h, c) = state
            # Reverse stack the tensors to get the states of each layer <br />
            # 📝 You can just work with the tensor itself but this is easier to debug
            h, c = torch.unbind(h), torch.unbind(c)

        # Array to collect the outputs of the final layer at each time step.
        out = []
        for t in range(time_steps):
            # Input to the first layer is the input itself
            inp = x[t]
            # Loop through the layers
            for layer in range(self.n_layers):
                # Get the state of the first layer
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
