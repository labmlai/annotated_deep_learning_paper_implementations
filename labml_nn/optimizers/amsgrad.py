"""
---
title: AMSGrad Optimizer
summary: A simple PyTorch implementation/tutorial of AMSGrad optimizer.
---

# AMSGrad

This is a [PyTorch](https://pytorch.org) implementation of the paper
[On the Convergence of Adam and Beyond](https://arxiv.org/abs/1904.09237).

We implement this as an extension to our [Adam optimizer implementation](adam.html).
The implementation it self is really small since it's very similar to Adam.

We also have an implementation of the synthetic example described in the paper where Adam fails to converge.
"""

from typing import Dict

import torch
from torch import nn

from labml_nn.optimizers import WeightDecay
from labml_nn.optimizers.adam import Adam


class AMSGrad(Adam):
    """
    ## AMSGrad Optimizer

    This class extends from Adam optimizer defined in [`adam.py`](adam.html).
    Adam optimizer is extending the class `GenericAdaptiveOptimizer`
    defined in [`__init__.py`](index.html).
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-16,
                 weight_decay: WeightDecay = WeightDecay(),
                 optimized_update: bool = True,
                 amsgrad=True, defaults=None):
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
        * `defaults` is a dictionary of default for group values.
         This is useful when you want to extend the class `Adam`.
        """
        defaults = {} if defaults is None else defaults
        defaults.update(dict(amsgrad=amsgrad))

        super().__init__(params, lr, betas, eps, weight_decay, optimized_update, defaults)

    def init_state(self, state: Dict[str, any], group: Dict[str, any], param: nn.Parameter):
        """
        ### Initialize a parameter state

        * `state` is the optimizer state of the parameter (tensor)
        * `group` stores optimizer attributes of the parameter group
        * `param` is the parameter tensor $\theta_{t-1}$
        """

        # Call `init_state` of Adam optimizer which we are extending
        super().init_state(state, group, param)

        # If `amsgrad` flag is `True` for this parameter group, we maintain the maximum of
        # exponential moving average of squared gradient
        if group['amsgrad']:
            state['max_exp_avg_sq'] = torch.zeros_like(param, memory_format=torch.preserve_format)

    def get_mv(self, state: Dict[str, any], group: Dict[str, any], grad: torch.Tensor):
        """
        ### Calculate $m_t$ and and $v_t$ or $\max(v_1, v_2, ..., v_{t-1}, v_t)$

        * `state` is the optimizer state of the parameter (tensor)
        * `group` stores optimizer attributes of the parameter group
        * `grad` is the current gradient tensor $g_t$ for the parameter $\theta_{t-1}$
        """

        # Get $m_t$ and $v_t$ from *Adam*
        m, v = super().get_mv(state, group, grad)

        # If this parameter group is using `amsgrad`
        if group['amsgrad']:
            # Get $\max(v_1, v_2, ..., v_{t-1})$.
            #
            # 🗒 The paper uses the notation $\hat{v}_t$ for this, which we don't use
            # that here because it confuses with the Adam's usage of the same notation
            # for bias corrected exponential moving average.
            v_max = state['max_exp_avg_sq']
            # Calculate $\max(v_1, v_2, ..., v_{t-1}, v_t)$.
            #
            # 🤔 I feel you should be taking / maintaining the max of the bias corrected
            # second exponential average of squared gradient.
            # But this is how it's
            # [implemented in PyTorch also](https://github.com/pytorch/pytorch/blob/19f4c5110e8bcad5e7e75375194262fca0a6293a/torch/optim/functional.py#L90).
            # I guess it doesn't really matter since bias correction only increases the value
            # and it only makes an actual difference during the early few steps of the training.
            torch.maximum(v_max, v, out=v_max)

            return m, v_max
        else:
            # Fall back to *Adam* if the parameter group is not using `amsgrad`
            return m, v


def _synthetic_experiment(is_adam: bool):
    """
    ## Synthetic Experiment

    This is the synthetic experiment described in the paper,
    that shows a scenario where *Adam* fails.

    The paper (and Adam) formulates the problem of optimizing as
    minimizing the expected value of a function, $\mathbb{E}[f(\theta)]$
    with respect to the parameters $\theta$.
    In the stochastic training setting we do not get hold of the function $f$
    it self; that is,
    when you are optimizing a NN $f$ would be the function on  entire
    batch of data.
    What we actually evaluate is a mini-batch so the actual function is
    realization of the stochastic $f$.
    This is why we are talking about an expected value.
    So let the function realizations be $f_1, f_2, ..., f_T$ for each time step
    of training.

    We measure the performance of the optimizer as the regret,
    $$R(T) = \sum_{t=1}^T \big[ f_t(\theta_t) - f_t(\theta^*) \big]$$
    where $theta_t$ is the parameters at time step $t$, and  $\theta^*$ is the
    optimal parameters that minimize $\mathbb{E}[f(\theta)]$.

    Now lets define the synthetic problem,
    \begin{align}
    f_t(x) =
    \begin{cases}
    1010 x,  & \text{for $t \mod 101 = 1$} \\
    -10  x, & \text{otherwise}
    \end{cases}
    \end{align}
    where $-1 \le x \le +1$.
    The optimal solution is $x = -1$.

    This code will try running *Adam* and *AMSGrad* on this problem.
    """

    # Define $x$ parameter
    x = nn.Parameter(torch.tensor([.0]))
    # Optimal, $x^* = -1$
    x_star = nn.Parameter(torch.tensor([-1]), requires_grad=False)

    def func(t: int, x_: nn.Parameter):
        """
        ### $f_t(x)$
        """
        if t % 101 == 1:
            return (1010 * x_).sum()
        else:
            return (-10 * x_).sum()

    # Initialize the relevant optimizer
    if is_adam:
        optimizer = Adam([x], lr=1e-2, betas=(0.9, 0.99))
    else:
        optimizer = AMSGrad([x], lr=1e-2, betas=(0.9, 0.99))
    # $R(T)$
    total_regret = 0

    from labml import monit, tracker, experiment

    # Create experiment to record results
    with experiment.record(name='synthetic', comment='Adam' if is_adam else 'AMSGrad'):
        # Run for $10^7$ steps
        for step in monit.loop(10_000_000):
            # $f_t(\theta_t) - f_t(\theta^*)$
            regret = func(step, x) - func(step, x_star)
            # $R(T) = \sum_{t=1}^T \big[ f_t(\theta_t) - f_t(\theta^*) \big]$
            total_regret += regret.item()
            # Track results every 1,000 steps
            if (step + 1) % 1000 == 0:
                tracker.save(loss=regret, x=x, regret=total_regret / (step + 1))
            # Calculate gradients
            regret.backward()
            # Optimize
            optimizer.step()
            # Clear gradients
            optimizer.zero_grad()

            # Make sure $-1 \le x \le +1$
            x.data.clamp_(-1., +1.)


if __name__ == '__main__':
    # Run the synthetic experiment is *Adam*.
    # [Here are the results](https://web.lab-ml.com/metrics?uuid=61ebfdaa384411eb94d8acde48001122).
    # You can see that Adam converges at $x = +1$
    _synthetic_experiment(True)
    # Run the synthetic experiment is *AMSGrad*
    # [Here are the results](https://web.lab-ml.com/metrics?uuid=bc06405c384411eb8b82acde48001122).
    # You can see that AMSGrad converges to true optimal $x = -1$
    _synthetic_experiment(False)
