import math
from typing import Dict

import torch
from torch import nn

from labml_nn.optimizers import GenericAdaptiveOptimizer, WeightDecay


class Adam(GenericAdaptiveOptimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-16,
                 amsgrad=False,
                 weight_decay: WeightDecay = WeightDecay()):
        defaults = dict(amsgrad=amsgrad,
                        buffer=[[None, None, None] for _ in range(10)])
        defaults.update(weight_decay.defaults())
        super().__init__(params, defaults, lr, betas, eps)

        self.weight_decay = weight_decay

    def init_state(self, state: Dict[str, any], group: Dict[str, any], p: nn.Parameter):
        state['step'] = 0
        # Exponential moving average of gradient values
        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
        # Exponential moving average of squared gradient values
        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

        if group['amsgrad']:
            # Maintains max of all exp. moving avg. of sq. grad. values
            state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

    def calculate(self, state: Dict[str, any], group: Dict[str, any], grad: torch.Tensor, param: torch.nn.Parameter):
        self.weight_decay(param, group)

        beta1, beta2 = group['betas']

        # get current state variable
        m, v = state['exp_avg'], state['exp_avg_sq']

        state['step'] += 1
        bias_correction1 = 1 - beta1 ** state['step']
        bias_correction2 = 1 - beta2 ** state['step']

        # Update first and second moment running average
        m.mul_(beta1).add_(grad, alpha=1 - beta1)
        v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        if group['amsgrad']:
            v_max = state['max_exp_avg_sq']
            torch.maximum(v_max, v, out=v_max)
            denominator = (v_max.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
        else:
            denominator = (v.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

        param.data.addcdiv_(m, denominator, value=-group['lr'] / bias_correction1)
