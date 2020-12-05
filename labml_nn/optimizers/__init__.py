"""
# Optimizers

* [Adam](adam.html)
"""

from typing import Dict, Tuple

import torch
from torch import nn
from torch.optim.optimizer import Optimizer


class GenericAdaptiveOptimizer(Optimizer):
    def __init__(self, params, defaults, lr: float, betas: Tuple[float, float], eps: float):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults.update(dict(lr=lr, betas=betas, eps=eps))
        super().__init__(params, defaults)

    def init_state(self, state: Dict[str, any], group: Dict[str, any], param: nn.Parameter):
        pass

    def step_param(self, state: Dict[str, any], group: Dict[str, any], grad: torch.Tensor, param: torch.Tensor):
        pass

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('AdaBelief does not support sparse gradients,'
                                       ' please consider SparseAdam instead')

                state = self.state[p]
                # Lazy state initialization
                if len(state) == 0:
                    self.init_state(state, group, p)

                self.step_param(state, group, grad, p)

        return loss


class WeightDecay:
    def __init__(self, weight_decay: float = 0., weight_decouple=True, absolute=False):
        self.absolute = absolute
        self.weight_decouple = weight_decouple
        self.weight_decay = weight_decay
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

    def defaults(self):
        return dict(weight_decay=self.weight_decay)

    def __call__(self, param: torch.nn.Parameter, group: Dict[str, any]):
        grad = param.grad.data

        if self.weight_decouple:
            if not self.absolute:
                param.data.mul_(1.0 - group['lr'] * group['weight_decay'])
            else:
                param.data.mul_(1.0 - group['weight_decay'])
        else:
            if group['weight_decay'] != 0:
                grad.add_(param.data, alpha=group['weight_decay'])
