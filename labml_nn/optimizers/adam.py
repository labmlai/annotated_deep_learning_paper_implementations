"""
---
title: Adam Optimizer
summary: A simple PyTorch implementation/tutorial of Adam optimizer
---

# Adam Optimizer

This is a [PyTorch](https://pytorch.org) implementation of popular optimizer *Adam* from paper
 [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980v9).

*Adam* update is,

\begin{align}
m_t &\leftarrow \beta_1 m_{t-1} + (1 - \beta_1) \cdot g_t \\
v_t &\leftarrow \beta_2 v_{t-1} + (1 - \beta_2) \cdot g_t^2 \\
\hat{m}_t &\leftarrow \frac{m_t}{1-\beta_1^t} \\
\hat{v}_t &\leftarrow \frac{v_t}{1-\beta_2^t} \\
\theta_t &\leftarrow \theta_{t-1} - \alpha \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
\end{align}

where $\alpha$, $\beta_1$, $\beta_2$ and $\epsilon$ are scalar hyper parameters.
$m_t$ and $v_t$ are first and second order moments.
$\hat{m}_t$  and $\hat{v}_t$ are biased corrected moments.
$\epsilon$ is used as a fix for division by zero error, but also acts as a form of a hyper-parameter
that acts against variance in gradients.

Effective step taken assuming $\epsilon = 0$ is,
$$\Delta t = \alpha \cdot \frac{\hat{m}_t}{\hat{v}_t}$$
This is bounded by,
$$\vert \Delta t \vert \le \alpha \cdot \frac{1 - \beta_1}{\sqrt{1-\beta_2}}$$
when $1-\beta_1 \gt \sqrt{1-\beta_2}$
and
$$\vert \Delta t\vert  \le \alpha$$
otherwise.
And in most common scenarios,
$$\vert \Delta t \vert \approx \alpha$$
"""

import math
from typing import Dict, Any, Tuple, Optional

import torch
from labml import tracker
from torch import nn

from labml_nn.optimizers import GenericAdaptiveOptimizer, WeightDecay


class Adam(GenericAdaptiveOptimizer):
    """
    ## Adam Optimizer

    We extend the class `GenericAdaptiveOptimizer` defined in [`__init__.py`](index.html)
    to implement the Adam optimizer.
    """

    def __init__(self, params,
                 lr: float = 1e-3, betas: Tuple[float, float] = (0.9, 0.999), eps: float = 1e-16,
                 weight_decay: WeightDecay = WeightDecay(),
                 optimized_update: bool = True,
                 defaults: Optional[Dict[str, Any]] = None):
        """
        ### Initialize the optimizer

        * `params` is the list of parameters
        * `lr` is the learning rate $\alpha$
        * `betas` is a tuple of ($\beta_1$, $\beta_2$)
        * `eps` is $\hat{\epsilon}$ or $\epsilon$ based on `optimized_update`
        * `weight_decay` is an instance of class `WeightDecay` defined in [`__init__.py`](index.html)
        * `optimized_update` is a flag whether to optimize the bias correction of the second moment
          by doing it after adding $\epsilon$
        * `defaults` is a dictionary of default for group values.
         This is useful when you want to extend the class `Adam`.
        """
        defaults = {} if defaults is None else defaults
        defaults.update(weight_decay.defaults())
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
        # Exponential moving average of gradients, $m_t$
        state['exp_avg'] = torch.zeros_like(param, memory_format=torch.preserve_format)
        # Exponential moving average of squared gradient values, $v_t$
        state['exp_avg_sq'] = torch.zeros_like(param, memory_format=torch.preserve_format)

    def get_mv(self, state: Dict[str, Any], group: Dict[str, Any], grad: torch.Tensor):
        """
        ### Calculate $m_t$ and and $v_t$

        * `state` is the optimizer state of the parameter (tensor)
        * `group` stores optimizer attributes of the parameter group
        * `grad` is the current gradient tensor $g_t$ for the parameter $\theta_{t-1}$
        """

        # Get $\beta_1$ and $\beta_2$
        beta1, beta2 = group['betas']

        # Get $m_{t-1}$ and $v_{t-1}$
        m, v = state['exp_avg'], state['exp_avg_sq']

        # In-place calculation of $m_t$
        # $$m_t \leftarrow \beta_1 m_{t-1} + (1 - \beta_1) \cdot g_t$$
        m.mul_(beta1).add_(grad, alpha=1 - beta1)
        # In-place calculation of $v_t$
        # $$v_t \leftarrow \beta_2 v_{t-1} + (1 - \beta_2) \cdot g_t^2$$
        v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        return m, v

    def get_lr(self, state: Dict[str, any], group: Dict[str, any]):
        """
        ### Get learning-rate

        This returns the modified learning rate based on the state.
        For *Adam* this is just the specified learning rate for the parameter group,
        $\alpha$.
        """
        return group['lr']

    def adam_update(self, state: Dict[str, any], group: Dict[str, any], param: torch.nn.Parameter,
                    m: torch.Tensor, v: torch.Tensor):
        """
        ### Do the *Adam* parameter update

        * `state` is the optimizer state of the parameter (tensor)
        * `group` stores optimizer attributes of the parameter group
        * `param` is the parameter tensor $\theta_{t-1}$
        * `m` and `v` are the uncorrected first and second moments $m_t$ and $v_t$.

        This computes the following

        \begin{align}
        \theta_t &\leftarrow \theta_{t-1} - \alpha \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
        \end{align}

        Since $\alpha$, $\beta_1$, $\beta_2$ and $\epsilon$ are scalars and others are tensors
        we modify this calculation to optimize the computation.

        \begin{align}
        \theta_t &\leftarrow \theta_{t-1} - \alpha \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} \\
        \theta_t &\leftarrow \theta_{t-1} - \alpha \cdot
                \frac{m_t / (1-\beta_1^t)}{\sqrt{v_t/(1-\beta_2^t)} + \epsilon} \\
        \theta_t &\leftarrow \theta_{t-1} - \alpha \frac{\sqrt{1-\beta_2^t}}{1-\beta_1^t} \cdot
                \frac{m_t}{\sqrt{v_t} + \hat{\epsilon}} \\
        \end{align}

        where
        $$\hat{\epsilon} = (1-\beta_2^t) \epsilon$$
        is what we should specify as the hyper-parameter.
        """

        # Get $\beta_1$ and $\beta_2$
        beta1, beta2 = group['betas']
        # Bias correction term for $\hat{m}_t$, $1 - \beta_1^t$
        bias_correction1 = 1 - beta1 ** state['step']
        # Bias correction term for $\hat{v}_t$, $1 - \beta_2^t$
        bias_correction2 = 1 - beta2 ** state['step']

        # Get learning rate
        lr = self.get_lr(state, group)

        # Whether to optimize the computation
        if self.optimized_update:
            # $\sqrt{v_t} + \hat{\epsilon}$
            denominator = v.sqrt().add_(group['eps'])
            # $\alpha \frac{\sqrt{1-\beta_2^t}}{1-\beta_1^t}$
            step_size = lr * math.sqrt(bias_correction2) / bias_correction1
            # $\theta_t \leftarrow \theta_{t-1} - \alpha \frac{\sqrt{1-\beta_2^t}}{1-\beta_1^t} \cdot
            #  \frac{m_t}{\sqrt{v_t} + \hat{\epsilon}}$
            param.data.addcdiv_(m, denominator, value=-step_size)
        # Computation without optimization
        else:
            # $\frac{\sqrt{v_t}}{\sqrt{1-\beta_2^t}} + \epsilon$
            denominator = (v.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
            # $\frac{\alpha}{1-\beta_1^t}$
            step_size = lr / bias_correction1
            # $\theta_t \leftarrow \theta_{t-1} - \alpha \cdot
            # \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$
            param.data.addcdiv_(m, denominator, value=-step_size)

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
        m, v = self.get_mv(state, group, grad)

        # Increment $t$ the number of optimizer steps
        state['step'] += 1

        # Perform *Adam* update
        self.adam_update(state, group, param, m, v)
