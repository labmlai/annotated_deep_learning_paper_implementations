"""
---
title: Rectified Adam (RAdam) optimizer
summary: A simple PyTorch implementation/tutorial of RAdam optimizer.
---

# Rectified Adam (RAdam) optimizer

This implementation is based on
[the official implementation](https://github.com/LiyuanLucasLiu/RAdam)
of the paper
[On the Variance of the Adaptive Learning Rate and Beyond](https://arxiv.org/abs/1908.03265).

We have implemented it as an extension to [our AMSGrad implementation](amsgrad.html)
thus requiring only the modifications to be implemented.

Adam optimizer sometimes converges to a bad local optima during the initial stages of the training;
especially when training transformers.
Researches use warmups to counter this; for the the initial training steps (warm-up stage)
they use a low learning rate.
This paper identifies the problem to be the high variance of adaptive learning rate
during initial stages of training, and counters it using a new rectification term to
reduce variance.


"""

import math
from typing import Dict

import torch

from labml_nn.optimizers import WeightDecay
from labml_nn.optimizers.amsgrad import AMSGrad


class RAdam(AMSGrad):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay: WeightDecay = WeightDecay(), amsgrad=False,
                 degenerated_to_sgd=True, defaults=None):
        self.degenerated_to_sgd = degenerated_to_sgd
        super().__init__(params, lr, betas, eps, weight_decay, False, amsgrad, defaults)

    def step_param(self, state: Dict[str, any], group: Dict[str, any], grad: torch.Tensor, param: torch.nn.Parameter):
        grad = self.weight_decay(param, grad, group)

        m, v = self.get_mv(state, group, grad)
        state['step'] += 1

        self.r_adam_update(state, group, param, m, v)

    def r_adam_update(self, state: Dict[str, any], group: Dict[str, any], param: torch.nn.Parameter,
                      m: torch.Tensor, v: torch.Tensor):
        beta1, beta2 = group['betas']
        bias_correction1 = 1 - beta1 ** state['step']
        bias_correction2 = 1 - beta2 ** state['step']

        beta2_t = beta2 ** state['step']
        rho_inf = 2 / (1 - beta2) - 1
        rho = rho_inf - 2 * state['step'] * beta2_t / (1 - beta2_t)

        # more conservative since it's an approximated value
        if rho >= 5:
            r2 = (rho - 4) / (rho_inf - 4) * (rho - 2) / rho * rho_inf / (rho_inf - 2)
            denominator = (v.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
            step_size = self.get_lr(state, group) * math.sqrt(r2) / bias_correction1
            param.data.addcdiv_(m, denominator, value=-step_size)
        elif self.degenerated_to_sgd:
            step_size = self.get_lr(state, group) / bias_correction1
            param.data.add_(m, alpha=-step_size)
