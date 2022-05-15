import torch
from torch import nn


class FTA(nn.Module):
    def __init__(self, lower_limit: float, upper_limit: float, delta: float, eta: float):
        super().__init__()
        self.c = torch.arange(lower_limit, upper_limit, delta)
        self.expansion_factor = len(self.c)
        self.delta = delta
        self.eta = eta

    def fuzzy_i_plus(self, x):
        return (x <= self.eta) * x + (x > self.eta)

    def forward(self, x: torch.Tensor):
        x = x.view(*x.shape, 1)

        x = 1. - self.fuzzy_i_plus(torch.clip(self.c - x, min=0.) + torch.clip(x - self.delta - self.c, min=0.))

        return x.view(*x.shape[:-2], -1)


def _test():
    from labml.logger import inspect

    a = FTA(-10, 10, 2., 0.5)
    inspect(a.c)

    features = torch.tensor([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9., 10., 11.])
    inspect(features)

    inspect(a(features))
    inspect(a.expansion_factor)


if __name__ == '__main__':
    _test()
