"""
---
title: Sophia Optimizer
summary: A simple PyTorch implementation/tutorial of Sophia optimizer
---

# Sophia Optimizer

This is a [PyTorch](https://pytorch.org) implementation of *Sophia-G* from paper
 [Sophia: A Scalable Stochastic Second-order Optimizer for Language Model Pre-training](https://papers.labml.ai/paper/2305.14342).
Official implementation is available at [Liuhong99/Sophia](https://github.com/Liuhong99/Sophia).

Sophia is more adaptive to heterogeneous curvatures than Adam, more resistant
to non-convexity and rapid change of Hessian than Newton’s method, and also uses a low-cost
pre-conditioner.

Sophia keeps diagonal Hessian estimates with EMA across iterations.
The diagonal Hessian $\hat{h}_t$ is calculated every $k$ steps.

\begin{align}
h_t = \beta_2 h_{t-k} + (1 - \beta_2) \hat{h}_t \ \ \ \ \text{ if } t \text{ mod } k = 1; \text{ else }  h_t = h_{t-1}
\end{align}

Sophia uses EMA of gradients $m_t$, only considers positive entries of
 the diagonal Hessian and does per-coordinate clipping to the update.

\begin{align}
m_t &\leftarrow \beta_1 m_{t-1} + (1 - \beta_1)g_t \\
\theta_{t + 1} &\leftarrow \theta_t - \eta \cdot \operatorname{clip} \bigg(\frac{m_t}{ \max \{h_t, \epsilon \} }, \rho \bigg)
\end{align}

where $\epsilon$ is a very small value to prevent division by $0$.

### Gauss-Newton-Bartlett (GNB) estimator

\begin{align}
\hat{L}(\theta) &= \frac{1}{B} \sum^{B}_{b=1} \ell_{CE} \big( f(\theta, x_b), \hat{y}_b \big) \\
\hat{h}_t &= B \cdot \nabla_\theta \hat{L} (\theta) \odot \nabla_\theta \hat{L} (\theta)
\end{align}

where $x_b$ are the inputs,
$B$ is the batch size (number of inputs/tokens),
$\ell_{CE}$ is cross entropy loss, and
$\hat{y}_b$ are sampled from the logits $f(\theta, x_b)$.

Note that this hessian estimate is always positive and therefore we
can replace $\max \{h_t, \epsilon \}$ with $h_t + \epsilon$.

Sophia with Gauss-Newton-Bartlett (GNB) estimator is **Sophia-G**

Here is an [experiment](../transformers/basic/with_sophia.html) that uses Sophia-G to train a transformer.
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
                 lr: float = 1e-4, betas: Tuple[float, float] = (0.9, 0.95), eps: float = 1e-12,
                 rho: float = 0.03,
                 weight_decay: WeightDecay = WeightDecay(),
                 defaults: Optional[Dict[str, Any]] = None):
        """
        ### Initialize the optimizer

        * `params` is the list of parameters
        * `lr` is the maximum learning rate $\eta \rho$
        * `betas` is a tuple of ($\beta_1$, $\beta_2$)
        * `eps` is $\epsilon$
        * `pho` is $\rho$
        * `weight_decay` is an instance of class `WeightDecay` defined in [`__init__.py`](index.html)
        * `defaults` is a dictionary of default for group values.
         This is useful when you want to extend the class `Adam`.
        """
        defaults = {} if defaults is None else defaults
        defaults.update(weight_decay.defaults())
        defaults.update(dict(rho=rho))
        super().__init__(params, defaults, lr, betas, eps)

        self.weight_decay = weight_decay

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
        # Exponential moving average of Hessian diagonal, $h_t$
        state['hessian'] = torch.zeros_like(param, memory_format=torch.preserve_format)

    def update_hessian(self, n_tokens_training_batch):
        """
        ### Update the EMA of Hessian diagonal $h_t$

        * `n_tokens_training_batch` is the number of tokens/inputs in the batch $B$

        \begin{align}
        \hat{h}_t &= B \cdot \nabla_\theta \hat{L} (\theta) \odot \nabla_\theta \hat{L} (\theta) \\
        h_t &= \beta_2 h_{t-k} + (1 - \beta_2) \hat{h}_t
        \end{align}
        """

        # Iterate through parameter groups
        for group in self.param_groups:
            # $\beta_2$
            _, beta2 = group['betas']
            # Iterate through parameters
            for p in group['params']:
                # Skip parameters without gradients
                if p.grad is None:
                    continue

                # Get optimizer state
                state = self.state[p]

                # Initialize state if empty
                if len(state) == 0:
                    self.init_state(state, group, p)

                # Update EMA Hessian diagonal
                #
                # \begin{align}
                # \hat{h}_t &= B \cdot \nabla_\theta \hat{L} (\theta) \odot \nabla_\theta \hat{L} (\theta) \\
                # h_t &= \beta_2 h_{t-k} + (1 - \beta_2) \hat{h}_t
                # \end{align}
                state['hessian'].mul_(beta2).addcmul_(p.grad, p.grad, value=(1 - beta2) * n_tokens_training_batch)

    def step_param(self, state: Dict[str, any], group: Dict[str, any], grad: torch.Tensor, param: torch.nn.Parameter):
        """
        ### Take an update step for a given parameter tensor

        * `state` is the optimizer state of the parameter (tensor)
        * `group` stores optimizer attributes of the parameter group
        * `grad` is the current gradient tensor  $g_t$ for the parameter $\theta_{t-1}$
        * `param` is the parameter tensor $\theta_{t-1}$

        We do the following parameter update,

        \begin{align}
        \theta_{t + 1} &\leftarrow \theta_t - \eta \cdot \operatorname{clip} \bigg(\frac{m_t}{h_t + \epsilon}, \rho \bigg)
        \end{align}
        """

        # Calculate weight decay
        grad = self.weight_decay(param, grad, group)

        # Get $\beta_1$ and $\beta_2$
        beta1, beta2 = group['betas']
        # Get $\rho$
        rho = group['rho']

        # Get $m_{t-1}$ and $h_{t}$
        m, hessian = state['exp_avg'], state['hessian']

        # In-place calculation of $m_t$
        # $$m_t \leftarrow \beta_1 m_{t-1} + (1 - \beta_1) \cdot g_t$$
        m.mul_(beta1).add_(grad, alpha=1 - beta1)

        # Increment $t$ the number of optimizer steps
        state['step'] += 1

        # Get maximum learning rate $\eta \rho$
        lr = group['lr']

        # $\eta$
        eta = lr / rho

        # $$\operatorname{clip} \bigg(\frac{m_t}{h_t + \epsilon}, \rho \bigg)$$
        ratio = (m / (hessian + group['eps'])).clamp(-rho, rho)

        # $$\theta_{t + 1} \leftarrow \theta_t - \eta \cdot \operatorname{clip} \bigg(\frac{m_t}{h_t + \epsilon}, \rho \bigg)$$
        param.data.add_(ratio, alpha=-eta)
