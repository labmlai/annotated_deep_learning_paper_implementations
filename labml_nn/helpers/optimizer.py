from typing import Tuple

import torch
from labml import tracker

from labml.configs import BaseConfigs, option, meta_config


class OptimizerConfigs(BaseConfigs):
    r"""
    This creates a configurable optimizer.

    Arguments:
        learning_rate (float): Learning rate of the optimizer. Defaults to ``0.01``.
        momentum (float): Momentum of the optimizer. Defaults to ``0.5``.
        parameters: Model parameters to optimize.
        d_model (int): Embedding size of the model (for Noam optimizer).
        betas (Tuple[float, float]): Betas for Adam optimizer. Defaults to ``(0.9, 0.999)``.
        eps (float): Epsilon for Adam/RMSProp optimizers. Defaults to ``1e-8``.
        step_factor (int): Step factor for Noam optimizer. Defaults to ``1024``.

    Also there is a better (more options) implementation in ``labml_nn``.
    `We recommend using that <https://nn.labml.ai/optimizers/configs.html>`_.
    """

    optimizer: torch.optim.Adam
    learning_rate: float = 0.01
    momentum: float = 0.5
    parameters: any
    d_model: int
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    step_factor: int = 1024

    def __init__(self):
        super().__init__(_primary='optimizer')


meta_config(OptimizerConfigs.parameters)


@option(OptimizerConfigs.optimizer, 'SGD')
def sgd_optimizer(c: OptimizerConfigs):
    return torch.optim.SGD(c.parameters, c.learning_rate, c.momentum)


@option(OptimizerConfigs.optimizer, 'Adam')
def adam_optimizer(c: OptimizerConfigs):
    return torch.optim.Adam(c.parameters, lr=c.learning_rate,
                            betas=c.betas, eps=c.eps)


class NoamOpt:
    def __init__(self, model_size: int, learning_rate: float, warmup: int, step_factor: int, optimizer):
        self.step_factor = step_factor
        self.optimizer = optimizer
        self.warmup = warmup
        self.learning_rate = learning_rate
        self.model_size = model_size
        self._rate = 0

    def step(self):
        rate = self.rate(tracker.get_global_step() / self.step_factor)
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step):
        factor = self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5))
        return self.learning_rate * factor

    def zero_grad(self):
        self.optimizer.zero_grad()


@option(OptimizerConfigs.optimizer, 'Noam')
def noam_optimizer(c: OptimizerConfigs):
    optimizer = torch.optim.Adam(c.parameters, lr=0.0, betas=c.betas, eps=c.eps)
    return NoamOpt(c.d_model, 1, 2000, c.step_factor, optimizer)


def _test_noam_optimizer():
    import matplotlib.pyplot as plt
    import numpy as np

    opts = [NoamOpt(512, 1, 4000, None),
            NoamOpt(512, 1, 8000, None),
            NoamOpt(2048, 1, 2000, None)]
    plt.plot(np.arange(1, 20000), [[opt.rate(i) for opt in opts] for i in range(1, 20000)])
    plt.legend(["512:4000", "512:8000", "256:4000"])
    plt.title("Optimizer")
    plt.show()


if __name__ == '__main__':
    _test_noam_optimizer()
