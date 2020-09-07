import torch
from torch import nn

from labml_helpers.module import Module


class RHNCell(Module):
    def __init__(self, input_size: int, hidden_size: int, depth: int):
        super().__init__()

        self.hidden_size = hidden_size
        self.depth = depth
        self.hidden_lin = nn.ModuleList([nn.Linear(hidden_size, 2 * hidden_size) for _ in range(depth)])
        self.input_lin = nn.Linear(input_size, 2 * hidden_size, bias=False)

    def __call__(self, x, s):
        for i in range(self.depth):
            if i == 0:
                ht = self.input_lin(x) + self.hidden_lin[i](s)
            else:
                ht = self.hidden_lin[i](s)

            h = torch.tanh(ht[:, :self.hidden_size])
            t = torch.sigmoid(ht[:, self.hidden_size:])

            s = s + (h - s) * t

        return s


class RHN(Module):
    def __init__(self, input_size: int, hidden_size: int, depth: int, n_layers: int):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.cells = nn.ModuleList([RHNCell(input_size, hidden_size, depth)] +
                                   [RHNCell(hidden_size, hidden_size, depth) for _ in range(n_layers - 1)])

    def __call__(self, x: torch.Tensor, state=None):
        # x [seq_len, batch, d_model]
        time_steps, batch_size = x.shape[:2]

        if state is None:
            s = [x.new_zeros(batch_size, self.hidden_size) for _ in range(self.n_layers)]
        else:
            s = torch.unbind(state)

        out = []
        for t in range(time_steps):
            inp = x[t]
            for i in range(self.n_layers):
                s[i] = self.cells[i](inp, s[i])
                inp = s[i]
            out.append(s[-1])

        out = torch.stack(out)
        s = torch.stack(s)
        return out, s
