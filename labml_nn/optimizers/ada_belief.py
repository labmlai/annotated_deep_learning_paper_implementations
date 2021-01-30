"""
---
title: AdaBelief optimizer
summary: A simple PyTorch implementation/tutorial of AdaBelief optimizer.
---

# AdaBelief Optimizer

This is based from AdaBelief
[official implementation](https://github.com/juntang-zhuang/Adabelief-Optimizer)
of the paper
[AdaBelief Optimizer: Adapting Stepsizes by the Belief in Observed Gradients](https://arxiv.org/abs/2010.07468).

This is implemented in [PyTorch](https://pytorch.org) as an extension to [RAdam](radam.html).

The main difference between Adam optimizer and AdaBelief is that,
how it calculates the adaptive learning rate;
instead of dividing by the exponential moving average of square of the gradients,
AdaBelief divides by the exponential mean of variance.

\begin{align}
m_t &\leftarrow \beta_1 m_{t-1} + (1 - \beta_1) \cdot g_t \\
\color{cyan}{s_t} &\color{cyan}{\leftarrow} \color{cyan}{\beta_2 s_{t-1} + (1 - \beta_2) \cdot (g_t - m_t)^2} \\
\hat{m}_t &\leftarrow \frac{m_t}{1-\beta_1^t} \\
\color{cyan}{\hat{s}_t} &\color{cyan}{\leftarrow} \frac{\color{cyan}{s_t} + \color{red}{\epsilon}}{\color{cyan}{1-\beta_2^t}} \\
\theta_t &\leftarrow \theta_{t-1} - \alpha \cdot \frac{\hat{m}_t}{\sqrt{\color{cyan}{\hat{s}_t}} + \epsilon}
\end{align}

🤔 The paper calculates variance as $(g_t - m_t)^2$,
but I feel it should use the bias corrected momentum
$(g_t - \color{orange}{\hat{m}_t})^2$.
I guess this doesn't affect things much because
bias correction is $\approx 1$ after the initial training steps.
"""

from typing import Dict, Any

import torch
from torch import nn

from labml_nn.optimizers import WeightDecay
from labml_nn.optimizers.radam import RAdam


class AdaBelief(RAdam):
    """
    ## AdaBelief Optimizer

    This class extends from RAdam optimizer defined in [`radam.py`](radam.html).
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-16,
                 weight_decay: WeightDecay = WeightDecay(), amsgrad=False,
                 degenerate_to_sgd=True,
                 rectify=True, defaults=None):
        """
        ### Initialize the optimizer

        * `params` is the list of parameters
        * `lr` is the learning rate $\alpha$
        * `betas` is a tuple of ($\beta_1$, $\beta_2$)
        * `eps` is $\hat{\epsilon}$ or $\epsilon$ based on `optimized_update`
        * `weight_decay` is an instance of class `WeightDecay` defined in [`__init__.py`](index.html)
        * 'optimized_update' is a flag whether to optimize the bias correction of the second moment
          by doing it after adding $\epsilon$
        * `amsgrad` is a flag indicating whether to use AMSGrad or fallback to plain Adam
        * `degenerate_to_sgd` whether to use sgd when the rectification term $r_t is intractable
        * 'rectify' is whether to use RAdam update
        * `defaults` is a dictionary of default for group values.
         This is useful when you want to extend the class `AdaBelief`.
        """

        defaults = {} if defaults is None else defaults
        super().__init__(params, lr, betas, eps, weight_decay, amsgrad, degenerate_to_sgd, defaults)
        self.rectify = rectify

    def init_state(self, state: Dict[str, any], group: Dict[str, any], param: nn.Parameter):
        """
        ### Initialize a parameter state

        * `state` is the optimizer state of the parameter (tensor)
        * `group` stores optimizer attributes of the parameter group
        * `param` is the parameter tensor $\theta_{t-1}$
        """
        state['step'] = 0
        # Exponential moving average of gradient values
        state['exp_avg'] = torch.zeros_like(param, memory_format=torch.preserve_format)
        # Exponential moving average of variance
        state['exp_avg_var'] = torch.zeros_like(param, memory_format=torch.preserve_format)

        # If `amsgrad` flag is `True` for this parameter group, we maintain the maximum of
        # exponential moving average of variance
        if group['amsgrad']:
            # Maintains max of all exp. moving avg. of sq. grad. values
            state['max_exp_avg_var'] = torch.zeros_like(param, memory_format=torch.preserve_format)

    def get_ms(self, state: Dict[str, Any], group: Dict[str, Any], grad: torch.Tensor):
        """
        ### Calculate $m_t$ and $s_t$ or $\max(s_1, s_2, ..., s_{t-1}, s_t)$

        * `state` is the optimizer state of the parameter (tensor)
        * `group` stores optimizer attributes of the parameter group
        * `grad` is the current gradient tensor $g_t$ for the parameter $\theta_{t-1}$
        """

        # Get $\beta_1$ and $\beta_2$
        beta1, beta2 = group['betas']

        # Get $m_{t-1}$ and $s_{t-1}$
        m, s = state['exp_avg'], state['exp_avg_var']

        # In-place calculation of $m_t$
        # $$m_t \leftarrow \beta_1 m_{t-1} + (1 - \beta_1) \cdot g_t$$
        m.mul_(beta1).add_(grad, alpha=1 - beta1)
        # Difference between gradient and momentum
        grad_residual = grad - m
        # In-place calculation of $s_t$
        # $$s_t \leftarrow \beta_2 s_{t-1} + (1 - \beta_2) \cdot (g_t - m_t)^2$$
        s.mul_(beta2).addcmul_(grad_residual, grad_residual, value=1 - beta2)

        # If this parameter group is using `amsgrad`
        if group['amsgrad']:
            # Get $\max(s_1, s_2, ..., s_{t-1})$.
            s_max = state['max_exp_avg_var']
            # Calculate $\max(s_1, s_2, ..., s_{t-1}, s_t)$.
            torch.maximum(s_max, s, out=s_max)

            return m, s_max
        else:
            # $m_t$ and $s_t$ otherwise
            return m, s

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

        # Get $m_t$ and $v_t$
        m, s = self.get_ms(state, group, grad)

        # Increment $t$ the number of optimizer steps
        state['step'] += 1

        if not self.rectify:
            # Perform *Adam* update, defined in [`adam.py`](adam.html), with
            # $\color{cyan}{s_t} + \color{red}{\epsilon}$ in place of $v_t$.
            self.adam_update(state, group, param, m, s + group['eps'])
        else:
            # Perform *Rectified Adam* update defined in [`radam.py`](radam.html), with
            # $\color{cyan}{s_t} + \color{red}{\epsilon}$ in place of $v_t$.
            self.r_adam_update(state, group, param, m, s + group['eps'])
