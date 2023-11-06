"""
---
title: Noam optimizer from Attention is All You Need paper
summary: >
  This is a tutorial/implementation of Noam optimizer.
  Noam optimizer has a warm-up period and then an exponentially decaying learning rate.
---

# Noam Optimizer

This is the [PyTorch](https://pytorch.org) implementation of optimizer introduced in the paper
[Attention Is All You Need](https://arxiv.org/abs/1706.03762).
"""
from typing import Dict

from labml_nn.optimizers import WeightDecay
from labml_nn.optimizers.amsgrad import AMSGrad


class Noam(AMSGrad):
    """
    ## Noam Optimizer

    This class extends from Adam optimizer defined in [`adam.py`](adam.html).
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-16,
                 weight_decay: WeightDecay = WeightDecay(),
                 optimized_update: bool = True,
                 amsgrad=False,
                 warmup=0, d_model=512, defaults=None):
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
        * `warmup` number of warmup steps
        * `d_model` model size; i.e. number of dimensions in the transformer
        * `defaults` is a dictionary of default for group values.
         This is useful when you want to extend the class `AdamWarmup`.
        """

        defaults = {} if defaults is None else defaults
        defaults.update(dict(warmup=warmup))
        super().__init__(params, lr, betas, eps, weight_decay, optimized_update, amsgrad, defaults)
        self.d_model = d_model

    def get_lr(self, state: Dict[str, any], group: Dict[str, any]):
        """
        ### Get learning-rate

        $$\alpha \frac{1}{\sqrt{d_{model}}} \min \bigg(\frac{1}{\sqrt{t}}, \frac{t}{w^{3/2}}\bigg)$$
        where $w$ is the number of warmup steps.
        """
        # $$\min \bigg(\frac{1}{\sqrt{t}}, \frac{t}{w^{3/2}}\bigg)$$
        factor = min(state['step'] ** (-0.5), state['step'] * group['warmup'] ** (-1.5))
        # $$\alpha \frac{1}{\sqrt{d_{model}}} \min \bigg(\frac{1}{\sqrt{t}}, \frac{t}{w^{3/2}}\bigg)$$
        return group['lr'] * self.d_model ** (-0.5) * factor


def _test_noam_lr():
    """
    ### Plot learning rate for different warmups and model sizes

    ![Plot of learning rate](noam_lr.png)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from torch import nn

    model = nn.Linear(10, 10)
    opts = [Noam(model.parameters(), d_model=512, warmup=4000, lr=1),
            Noam(model.parameters(), d_model=512, warmup=8000, lr=1),
            Noam(model.parameters(), d_model=2048, warmup=2000, lr=1)]
    plt.plot(np.arange(1, 20000), [[opt.get_lr({'step': i}, opt.defaults) for opt in opts] for i in range(1, 20000)])
    plt.legend(["512:4000", "512:8000", "2048:2000"])
    plt.title("Learning Rate")
    plt.show()


if __name__ == '__main__':
    _test_noam_lr()
