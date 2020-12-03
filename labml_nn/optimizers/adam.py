import math
from typing import Dict, Any

import torch
from torch import nn

from labml_nn.optimizers import GenericAdaptiveOptimizer, WeightDecay


class Adam(GenericAdaptiveOptimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-16,
                 weight_decay: WeightDecay = WeightDecay(), defaults=None):
        defaults = {} if defaults is None else defaults
        defaults.update(weight_decay.defaults())
        super().__init__(params, defaults, lr, betas, eps)

        self.weight_decay = weight_decay

    def init_state(self, state: Dict[str, any], group: Dict[str, any], p: nn.Parameter):
        state['step'] = 0
        # Exponential moving average of gradient values
        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
        # Exponential moving average of squared gradient values
        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

    def get_mv(self, state: Dict[str, Any], group: Dict[str, Any], grad: torch.Tensor):
        beta1, beta2 = group['betas']

        # get current state variable
        m, v = state['exp_avg'], state['exp_avg_sq']

        # Update first and second moment running average
        m.mul_(beta1).add_(grad, alpha=1 - beta1)
        v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        return m, v

    def get_lr(self, state: Dict[str, any], group: Dict[str, any]):
        return group['lr']

    def adam_update(self, state: Dict[str, any], group: Dict[str, any], param: torch.nn.Parameter,
             m: torch.Tensor, v: torch.Tensor):
        beta1, beta2 = group['betas']
        bias_correction1 = 1 - beta1 ** state['step']
        bias_correction2 = 1 - beta2 ** state['step']

        denominator = (v.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
        step_size = self.get_lr(state, group) / bias_correction1
        param.data.addcdiv_(m, denominator, value=-step_size)

    def calculate(self, state: Dict[str, any], group: Dict[str, any], grad: torch.Tensor, param: torch.nn.Parameter):
        self.weight_decay(param, group)

        m, v = self.get_mv(state, group, grad)

        state['step'] += 1

        self.adam_update(state, group, param, m, v)

