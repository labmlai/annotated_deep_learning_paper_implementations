"""
---
title: Noam optimizer from Attention is All You Need paper
summary: >
  This is a tutorial/implementation of Noam optimizer.
  Noam optimizer has a warm-up period and then an exponentially decaying learning rate.
---
"""
from typing import Dict

from labml_nn.optimizers import WeightDecay
from labml_nn.optimizers.amsgrad import AMSGrad


class Noam(AMSGrad):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-16,
                 weight_decay: WeightDecay = WeightDecay(),
                 optimized_update: bool = True,
                 amsgrad=False,
                 warmup=0, d_model=512, defaults=None):
        defaults = {} if defaults is None else defaults
        defaults.update(dict(warmup=warmup))
        super().__init__(params, lr, betas, eps, weight_decay, optimized_update, amsgrad, defaults)
        self.d_model = d_model

    def get_lr(self, state: Dict[str, any], group: Dict[str, any]):
        step = state['step']
        factor = self.d_model ** (-0.5) * min(step ** (-0.5), step * group['warmup'] ** (-1.5))
        return group['lr'] * factor


def _test_noam_optimizer():
    import matplotlib.pyplot as plt
    import numpy as np
    from torch import nn

    model = nn.Linear(10, 10)
    opts = [Noam(model.parameters(), d_model=512, warmup=4000),
            Noam(model.parameters(), d_model=512, warmup=8000),
            Noam(model.parameters(), d_model=2048, warmup=2000)]
    plt.plot(np.arange(1, 20000), [[opt.get_lr({'step': i}, opt.defaults) for opt in opts] for i in range(1, 20000)])
    plt.legend(["512:4000", "512:8000", "256:4000"])
    plt.title("Optimizer")
    plt.show()


if __name__ == '__main__':
    _test_noam_optimizer()
