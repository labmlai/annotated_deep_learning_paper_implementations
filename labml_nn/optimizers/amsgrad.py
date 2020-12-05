from typing import Dict

import torch
from torch import nn

from labml_nn.optimizers import WeightDecay
from labml_nn.optimizers.adam import Adam


class AMSGrad(Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-16,
                 weight_decay: WeightDecay = WeightDecay(), amsgrad=True, defaults=None):
        defaults = {} if defaults is None else defaults
        defaults.update(dict(amsgrad=amsgrad))

        super().__init__(params, lr, betas, eps, weight_decay, defaults)

    def init_state(self, state: Dict[str, any], group: Dict[str, any], param: nn.Parameter):
        super().init_state(state, group, param)
        # Maintains max of all exp. moving avg. of sq. grad. values
        if group['amsgrad']:
            state['max_exp_avg_sq'] = torch.zeros_like(param, memory_format=torch.preserve_format)

    def get_mv(self, state: Dict[str, any], group: Dict[str, any], grad: torch.Tensor):
        m, v = super().get_mv(state, group, grad)
        if group['amsgrad']:
            v_max = state['max_exp_avg_sq']
            torch.maximum(v_max, v, out=v_max)

            return m, v_max
        else:
            return m, v
