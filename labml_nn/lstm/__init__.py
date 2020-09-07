import torch
from torch import nn

from labml_helpers.module import Module


class LSTMCell(Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()

        self.hidden_size = hidden_size
        self.hidden_lin = nn.Linear(hidden_size, 4 * hidden_size)
        self.input_lin = nn.Linear(input_size, 4 * hidden_size, bias=False)

    def __call__(self, x, h, c):
        ifgo = self.hidden_lin(h) + self.input_lin(x)

        i = torch.sigmoid(ifgo[:, :self.hidden_size])
        f = torch.sigmoid(ifgo[:, self.hidden_size:self.hidden_size * 2])
        g = torch.tanh(ifgo[:, self.hidden_size * 2:self.hidden_size * 3])
        o = torch.sigmoid(ifgo[:, self.hidden_size * 3:self.hidden_size * 4])
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next


class LSTM(Module):
    def __init__(self, input_size: int, hidden_size: int, n_layers: int):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.cells = nn.ModuleList([LSTMCell(input_size, hidden_size)] +
                                   [LSTMCell(hidden_size, hidden_size) for _ in range(n_layers - 1)])

    def __call__(self, x: torch.Tensor, state=None):
        time_steps, batch_size = x.shape[:2]

        if state is None:
            h = [x.new_zeros(batch_size, self.hidden_size) for _ in range(self.n_layers)]
            c = [x.new_zeros(batch_size, self.hidden_size) for _ in range(self.n_layers)]
        else:
            (h, c) = state
            h, c = torch.unbind(h), torch.unbind(c)

        out = []
        for t in range(time_steps):
            inp = x[t]
            for i in range(self.n_layers):
                h[i], c[i] = self.cells[i](inp, h[i], c[i])
                inp = h[i]
            out.append(h[-1])

        out = torch.stack(out)
        h = torch.stack(h)
        c = torch.stack(c)

        return out, (h, c)
