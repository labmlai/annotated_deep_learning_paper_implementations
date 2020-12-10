"""
---
title: Adam optimizer with warm-up
summary: A simple PyTorch implementation/tutorial of Adam optimizer with warm-up.
---
"""

from typing import Dict

from labml_nn.optimizers import WeightDecay
from labml_nn.optimizers.amsgrad import AMSGrad


class AdamWarmup(AMSGrad):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-16,
                 weight_decay: WeightDecay = WeightDecay(),
                 optimized_update: bool = True,
                 amsgrad=False, warmup=0, defaults=None):
        defaults = {} if defaults is None else defaults
        defaults.update(dict(warmup=warmup))
        super().__init__(params, lr, betas, eps, weight_decay, optimized_update, amsgrad, defaults)

    def get_lr(self, state: Dict[str, any], group: Dict[str, any]):
        if group['warmup'] > state['step']:
            return 1e-8 + state['step'] * group['lr'] / group['warmup']
        else:
            return group['lr']
