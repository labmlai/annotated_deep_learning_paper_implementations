"""
# AMSGrad

This is an implementation of the paper
[On the Convergence of Adam and Beyond](https://arxiv.org/abs/1904.09237).

We implement this as an extention to our [Adam optimizer implementation](adam.html).
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

    This class extends from Adam optimizer
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-16,
                 weight_decay: WeightDecay = WeightDecay(), amsgrad=True, defaults=None):
        """
        ### Initialize the optimizer

        * `params` is the list of parameters
        * 'lr' is the learning rate $\alpha$
        * `betas` is a tuple of ($\beta_1$, $\beta_2$)
        * `eps` is $\hat{\epsilon}$
        * `weight_decay` is an instance of class `WeightDecay` defined in [__init__.py](index.html)
        * `amsgrad` is a flag indicating whether to use AMSGrad or fallback to plain Adam
        * `defaults` is a dictionary of default for group values.
         This is useful when you want to extend the class `Adam`.
        """
        defaults = {} if defaults is None else defaults
        defaults.update(dict(amsgrad=amsgrad))

        super().__init__(params, lr, betas, eps, weight_decay, defaults)

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
        ### Calculate $m_t$ and and $v_t$ or $$

        * `state` is the optimizer state of the parameter (tensor)
        * `group` stores optimizer attributes of the parameter group
        * `grad` is the current gradient tensor $g_t$ for the parameter $\theta_{t-1}$
        """
        m, v = super().get_mv(state, group, grad)
        if group['amsgrad']:
            v_max = state['max_exp_avg_sq']
            torch.maximum(v_max, v, out=v_max)

            return m, v_max
        else:
            return m, v


def _synthetic_experiment(is_adam: bool):
    x = nn.Parameter(torch.tensor([.0]))

    def func(t: int):
        if t % 101 == 1:
            return (1010 * x).sum()
        else:
            return (-10 * x).sum()

    from labml import monit, tracker, experiment

    if is_adam:
        optimizer = Adam([x], lr=1e-2, betas=(0.9, 0.99))
    else:
        optimizer = AMSGrad([x], lr=1e-2, betas=(0.9, 0.99))
    total_loss = 0
    with experiment.record(name='synthetic', comment='Adam' if is_adam else 'AMSGrad'):
        for i in monit.loop(10_000):
            loss = func(i) - (-1010 + 10 * 100) / 101.
            total_loss += loss.item()
            if (i + 1) % 1000 == 0:
                tracker.save(loss=loss, x=x, regret=total_loss / (i + 1))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            x.data.clamp_(-1., +1.)


if __name__ == '__main__':
    _synthetic_experiment(True)
    _synthetic_experiment(False)
