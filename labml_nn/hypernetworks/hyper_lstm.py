from typing import Optional, Tuple

import torch
from labml_helpers.module import Module
from torch import nn

from labml_nn.lstm import LSTMCell


class HyperLSTMCell(Module):
    def __init__(self, input_size: int, hidden_size: int, rhn_hidden_size: int, n_z: int):
        super().__init__()

        self.hidden_size = hidden_size

        # TODO: need layernorm
        self.rhn = LSTMCell(hidden_size + input_size, rhn_hidden_size)

        self.z_h = nn.Linear(rhn_hidden_size, 4 * n_z)
        self.z_x = nn.Linear(rhn_hidden_size, 4 * n_z)
        self.z_b = nn.Linear(rhn_hidden_size, 4 * n_z, bias=False)

        d_h = [nn.Linear(n_z, hidden_size, bias=False) for _ in range(4)]
        self.d_h = nn.ModuleList(d_h)
        d_x = [nn.Linear(n_z, hidden_size, bias=False) for _ in range(4)]
        self.d_x = nn.ModuleList(d_x)
        d_b = [nn.Linear(n_z, hidden_size) for _ in range(4)]
        self.d_b = nn.ModuleList(d_b)

        self.w_h = nn.ParameterList([nn.Parameter(torch.zeros(hidden_size, hidden_size)) for _ in range(4)])
        self.w_x = nn.ParameterList([nn.Parameter(torch.zeros(hidden_size, input_size)) for _ in range(4)])

        self.layer_norm = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(4)])

    def __call__(self, x: torch.Tensor,
                 h: torch.Tensor, c: torch.Tensor,
                 rhn_h: torch.Tensor, rhn_c: torch.Tensor):
        rhn_x = torch.cat((h, x), dim=-1)
        rhn_h, rhn_c = self.rhn(rhn_x, rhn_h, rhn_c)

        z_h = self.z_h(rhn_h).chunk(4, dim=-1)
        z_x = self.z_x(rhn_h).chunk(4, dim=-1)
        z_b = self.z_b(rhn_h).chunk(4, dim=-1)

        ifgo = []
        for i in range(4):
            d_h = self.d_h[i](z_h[i])
            w_h = torch.einsum('ij,bi->bij', self.w_h[i], d_h)
            d_x = self.d_x[i](z_x[i])
            w_x = torch.einsum('ij,bi->bij', self.w_x[i], d_x)
            b = self.d_b[i](z_b[i])

            g = torch.einsum('bij,bj->bi', w_h, h) + \
                torch.einsum('bij,bj->bi', w_x, x) + \
                b

            ifgo.append(self.layer_norm[i](g))

        # $$i_t = \sigma\big(lin_{xi}(x_t) + lin_{hi}(h_{t-1})\big)$$
        i = torch.sigmoid(ifgo[0])
        # $$f_t = \sigma\big(lin_{xf}(x_t) + lin_{hf}(h_{t-1})\big)$$
        f = torch.sigmoid(ifgo[1])
        # $$g_t = \tanh\big(lin_{xg}(x_t) + lin_{hg}(h_{t-1})\big)$$
        g = torch.tanh(ifgo[2])
        # $$o_t = \sigma\big(lin_{xo}(x_t) + lin_{ho}(h_{t-1})\big)$$
        o = torch.sigmoid(ifgo[3])

        # $$c_t = f_t \odot c_{t-1} + i_t \odot g_t$$
        c_next = f * c + i * g

        # $$h_t = o_t \odot \tanh(c_t)$$
        h_next = o * torch.tanh(c_next)

        return h_next, c_next, rhn_h, rhn_c


class HyperLSTM(Module):
    def __init__(self, input_size: int, hidden_size: int, rhn_hidden_size: int, n_z: int, n_layers: int):
        """
        Create a network of `n_layers` of LSTM.
        """

        super().__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.rhn_hidden_size = rhn_hidden_size
        # Create cells for each layer. Note that only the first layer gets the input directly.
        # Rest of the layers get the input from the layer below
        self.cells = nn.ModuleList([HyperLSTMCell(input_size, hidden_size, rhn_hidden_size, n_z)] +
                                   [HyperLSTMCell(hidden_size, hidden_size, rhn_hidden_size, n_z) for _ in
                                    range(n_layers - 1)])

    def __call__(self, x: torch.Tensor,
                 state: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None):
        """
        `x` has shape `[seq_len, batch_size, input_size]` and
        `state` is a tuple of $h$ and $c$, each with a shape of `[batch_size, hidden_size]`.
        """
        time_steps, batch_size = x.shape[:2]

        # Initialize the state if `None`
        if state is None:
            h = [x.new_zeros(batch_size, self.hidden_size) for _ in range(self.n_layers)]
            c = [x.new_zeros(batch_size, self.hidden_size) for _ in range(self.n_layers)]
            rhn_h = [x.new_zeros(batch_size, self.rhn_hidden_size) for _ in range(self.n_layers)]
            rhn_c = [x.new_zeros(batch_size, self.rhn_hidden_size) for _ in range(self.n_layers)]
        else:
            (h, c, rhn_h, rhn_c) = state
            # Reverse stack the tensors to get the states of each layer <br />
            # 📝 You can just work with the tensor itself but this is easier to debug
            h, c = list(torch.unbind(h)), list(torch.unbind(c))
            rhn_h, rhn_c = list(torch.unbind(rhn_h)), list(torch.unbind(rhn_c))

        # Array to collect the outputs of the final layer at each time step.
        out = []
        for t in range(time_steps):
            # Input to the first layer is the input itself
            inp = x[t]
            # Loop through the layers
            for layer in range(self.n_layers):
                # Get the state of the first layer
                h[layer], c[layer], rhn_h[layer], rhn_c[layer] = \
                    self.cells[layer](inp, h[layer], c[layer], rhn_h[layer], rhn_c[layer])
                # Input to the next layer is the state of this layer
                inp = h[layer]
            # Collect the output $h$ of the final layer
            out.append(h[-1])

        # Stack the outputs and states
        out = torch.stack(out)
        h = torch.stack(h)
        c = torch.stack(c)
        rhn_h = torch.stack(rhn_h)
        rhn_c = torch.stack(rhn_c)

        return out, (h, c, rhn_h, rhn_c)
