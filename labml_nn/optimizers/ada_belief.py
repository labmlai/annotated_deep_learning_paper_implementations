"""
---
title: AdaBelief optimizer
summary: A simple PyTorch implementation/tutorial of AdaBelief optimizer.
---

This is based from AdaBelief official implementation
https://github.com/juntang-zhuang/Adabelief-Optimizer
"""
from typing import Dict, Any

import torch
from torch import nn

from labml_nn.optimizers import WeightDecay
from labml_nn.optimizers.radam import RAdam


class AdaBelief(RAdam):
    r"""Implements AdaBelief algorithm. Modified from Adam in PyTorch
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-16)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)
        weight_decouple (boolean, optional): ( default: True) If set as True, then
            the optimizer uses decoupled weight decay as in AdamW
        fixed_decay (boolean, optional): (default: False) This is used when weight_decouple
            is set as True.
            When fixed_decay == True, the weight decay is performed as
            $W_{new} = W_{old} - W_{old} \times decay$.
            When fixed_decay == False, the weight decay is performed as
            $W_{new} = W_{old} - W_{old} \times decay \times lr$. Note that in this case, the
            weight decay ratio decreases with learning rate (lr).
        rectify (boolean, optional): (default: True) If set as True, then perform the rectified
            update similar to RAdam
        degenerated_to_sgd (boolean, optional) (default:True) If set as True, then perform SGD update
            when variance of gradient is high
    reference: AdaBelief Optimizer, adapting stepsizes by the belief in observed gradients, NeurIPS 2020
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-16,
                 weight_decay: WeightDecay = WeightDecay(), amsgrad=False,
                 degenerated_to_sgd=True,
                 rectify=True, defaults=None):

        defaults = {} if defaults is None else defaults
        super().__init__(params, lr, betas, eps, weight_decay, amsgrad, degenerated_to_sgd, defaults)
        self.rectify = rectify

    def init_state(self, state: Dict[str, any], group: Dict[str, any], param: nn.Parameter):
        state['step'] = 0
        # Exponential moving average of gradient values
        state['exp_avg'] = torch.zeros_like(param, memory_format=torch.preserve_format)
        # Exponential moving average of squared gradient values
        state['exp_avg_var'] = torch.zeros_like(param, memory_format=torch.preserve_format)

        if group['amsgrad']:
            # Maintains max of all exp. moving avg. of sq. grad. values
            state['max_exp_avg_var'] = torch.zeros_like(param, memory_format=torch.preserve_format)

    def get_mv(self, state: Dict[str, Any], group: Dict[str, Any], grad: torch.Tensor):
        beta1, beta2 = group['betas']

        # get current state variable
        m, v = state['exp_avg'], state['exp_avg_var']

        # Update first and second moment running average
        m.mul_(beta1).add_(grad, alpha=1 - beta1)
        grad_residual = grad - m
        v.mul_(beta2).addcmul_(grad_residual, grad_residual, value=1 - beta2)

        if group['amsgrad']:
            v_max = state['max_exp_avg_var']
            torch.maximum(v_max, v, out=v_max)

            return m, v_max
        else:
            return m, v

    def step_param(self, state: Dict[str, any], group: Dict[str, any], grad: torch.Tensor, param: torch.nn.Parameter):
        grad = self.weight_decay(param, grad, group)
        m, v = self.get_mv(state, group, grad)
        state['step'] += 1

        if not self.rectify:
            self.adam_update(state, group, param, m, v)
        else:  # Rectified update, forked from RAdam
            self.r_adam_update(state, group, param, m, v)
