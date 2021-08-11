from typing import Tuple

import torch
from torch import nn

from labml_helpers.module import Module


class SimplePonderGRU(Module):
    def __init__(self, n_elems, n_hidden, max_steps):
        super().__init__()

        self.max_steps = max_steps
        self.n_hidden = n_hidden

        self.hidden_layer = nn.GRUCell(n_elems, n_hidden)
        self.output_layer = nn.Linear(n_hidden, 1)
        self.lambda_layer = nn.Linear(n_hidden, 1)

        self.hidden_activation = nn.Tanh()
        self.lambda_prob = nn.Sigmoid()

    def __call__(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h = x.new_zeros((x.shape[0], self.n_hidden))
        h = self.hidden_layer(x, h)

        return self.both(x, h)

    def both(self, x: torch.Tensor, h: torch.Tensor):
        p = []
        y = []
        un_halted_prob = None

        batch_size = h.shape[0]

        halted = h.new_zeros((batch_size,))
        p_sampled = h.new_zeros((batch_size,))
        y_sampled = h.new_zeros((batch_size,))

        for n in range(1, self.max_steps + 1):
            if n == self.max_steps:
                lambda_n = h.new_ones(h.shape[0])
            else:
                lambda_n = self.lambda_prob(self.lambda_layer(h))[:, 0]
            y_n = self.output_layer(h)[:, 0]
            if un_halted_prob is None:
                p_n = lambda_n
                un_halted_prob = (1 - lambda_n)
            else:
                p_n = un_halted_prob * lambda_n
                un_halted_prob = un_halted_prob * (1 - lambda_n)

            if n == self.max_steps:
                halt = (1 - halted)
            else:
                halt = torch.bernoulli(p_n) * (1 - halted)

            p.append(p_n)
            y.append(y_n)

            p_sampled = p_sampled * (1 - halt) + p_n * halt
            y_sampled = y_sampled * (1 - halt) + y_n * halt

            halted = halted + halt
            h = self.hidden_layer(x, h)

        return torch.stack(p), torch.stack(y), p_sampled[None, :], y_sampled[None, :]


class ReconstructionLoss(Module):
    def __init__(self, loss_func: nn.Module):
        super().__init__()
        self.loss_func = loss_func

    def __call__(self, p: torch.Tensor, y_hat: torch.Tensor, y: torch.Tensor):
        total_loss = None
        for i in range(p.shape[0]):
            loss = (p[i] * self.loss_func(y_hat[i], y)).mean()
            if total_loss is None:
                total_loss = loss
            else:
                total_loss = total_loss + loss

        return total_loss


class RegularizationLoss(Module):
    def __init__(self, lambda_p: float, max_steps: int = 1_000):
        super().__init__()
        self.lambda_p = lambda_p
        p_g = torch.zeros((max_steps,))
        not_halted = 1.
        for i in range(max_steps):
            p_g[i] = not_halted * lambda_p
            not_halted = not_halted * (1 - lambda_p)

        self.p_g = nn.Parameter(p_g, requires_grad=False)

        self.kl_div = nn.KLDivLoss(reduction='batchmean')

    def __call__(self, p: torch.Tensor):
        p = p.transpose(0, 1)
        p_g = self.p_g[None, :p.shape[1]].expand_as(p)

        return self.kl_div(p.log(), p_g)
