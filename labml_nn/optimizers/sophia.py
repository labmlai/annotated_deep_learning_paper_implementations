"""
---
title: Sophia Optimizer
summary: A simple PyTorch implementation/tutorial of Sophia optimizer
---

# Sophia Optimizer

This is a [PyTorch](https://pytorch.org) implementation of *Sophia-G* from paper
 [Sophia: A Scalable Stochastic Second-order Optimizer for Language Model Pre-training](https://papers.labml.ai/paper/2305.14342).
"""

from typing import Dict, Any, Tuple, Optional

import torch
from torch import nn

from labml_nn.optimizers import GenericAdaptiveOptimizer, WeightDecay


class Sophia(GenericAdaptiveOptimizer):
    """
    ## Sophia-G Optimizer

    We extend the class `GenericAdaptiveOptimizer` defined in [`__init__.py`](index.html)
    to implement the Sophia optimizer.
    """

    def __init__(self, params,
                 lr: float = 1e-4, betas: Tuple[float, float] = (0.965, 0.99), eps: float = 1e-16,
                 rho: float = 0.04,
                 training_batch_tokens: int = None,
                 weight_decay: WeightDecay = WeightDecay(),
                 optimized_update: bool = True,
                 defaults: Optional[Dict[str, Any]] = None):
        """
        ### Initialize the optimizer

        * `params` is the list of parameters
        * `lr` is the learning rate $\alpha$
        * `betas` is a tuple of ($\beta_1$, $\beta_2$)
        * `eps` is $\epsilon$
        * `pho` is $\rho$
        * `weight_decay` is an instance of class `WeightDecay` defined in [`__init__.py`](index.html)
        * `optimized_update` is a flag whether to optimize the bias correction of the second moment
          by doing it after adding $\epsilon$
        * `defaults` is a dictionary of default for group values.
         This is useful when you want to extend the class `Adam`.
        """
        if training_batch_tokens is None:
            raise RuntimeError('Please set the number of tokens per training batch.')

        defaults = {} if defaults is None else defaults
        defaults.update(weight_decay.defaults())
        defaults.update(dict(rho=rho, training_batch_tokens=training_batch_tokens))
        super().__init__(params, defaults, lr, betas, eps)

        self.weight_decay = weight_decay
        self.optimized_update = optimized_update

    def init_state(self, state: Dict[str, any], group: Dict[str, any], param: nn.Parameter):
        """
        ### Initialize a parameter state

        * `state` is the optimizer state of the parameter (tensor)
        * `group` stores optimizer attributes of the parameter group
        * `param` is the parameter tensor $\theta_{t-1}$
        """

        # This is the number of optimizer steps taken on the parameter, $t$
        state['step'] = 0
        # state['hessian_updates']
        # Exponential moving average of gradients, $m_t$
        state['exp_avg'] = torch.zeros_like(param, memory_format=torch.preserve_format)
        # Exponential moving average of Hessian
        state['hessian'] = torch.zeros_like(param, memory_format=torch.preserve_format)

    def update_hessian(self, batch_size):
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]

                if len(state) == 0:
                    self.init_state(state, group, p)

                state['hessian'].mul_(beta2).addcmul_(p.grad, p.grad, value=(1 - beta2) * batch_size)

    def step_param(self, state: Dict[str, any], group: Dict[str, any], grad: torch.Tensor, param: torch.nn.Parameter):
        """
        ### Take an update step for a given parameter tensor

        * `state` is the optimizer state of the parameter (tensor)
        * `group` stores optimizer attributes of the parameter group
        * `grad` is the current gradient tensor  $g_t$ for the parameter $\theta_{t-1}$
        * `param` is the parameter tensor $\theta_{t-1}$
        """

        # Calculate weight decay
        grad = self.weight_decay(param, grad, group)

        # Get $\beta_1$ and $\beta_2$
        beta1, beta2 = group['betas']

        rho = group['rho']

        # Get $m_{t-1}$ and $v_{t-1}$
        m, hessian = state['exp_avg'], state['hessain']

        # In-place calculation of $m_t$
        # $$m_t \leftarrow \beta_1 m_{t-1} + (1 - \beta_1) \cdot g_t$$
        m.mul_(beta1).add_(grad, alpha=1 - beta1)

        # Increment $t$ the number of optimizer steps
        state['step'] += 1

        # Get learning rate
        lr = group['lr']

        ratio = (m.abs() / (rho * hessian + group['training_batch_tokens'] * group['eps'])).clamp(None, 1)

        param.data.addcmul_(m.sign(), ratio, value=-lr)
