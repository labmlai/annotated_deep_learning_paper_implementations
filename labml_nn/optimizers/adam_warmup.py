"""
---
title: Adam optimizer with warm-up
summary: A simple PyTorch implementation/tutorial of Adam optimizer with warm-up.
---

# Adam Optimizer with Warmup

This extends [AMSGrad optimizer](adam.html) and adds a warmup stage.
"""

from typing import Dict

from labml_nn.optimizers import WeightDecay
from labml_nn.optimizers.amsgrad import AMSGrad


class AdamWarmup(AMSGrad):
    """
    ## Adam Optimizer with Warmup

    This class extends from Adam optimizer defined in [`adam.py`](adam.html).
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-16,
                 weight_decay: WeightDecay = WeightDecay(),
                 optimized_update: bool = True,
                 amsgrad=False, warmup=0, defaults=None):
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
        * `defaults` is a dictionary of default for group values.
         This is useful when you want to extend the class `AdamWarmup`.
        """

        defaults = {} if defaults is None else defaults
        defaults.update(dict(warmup=warmup))
        super().__init__(params, lr, betas, eps, weight_decay, optimized_update, amsgrad, defaults)

    def get_lr(self, state: Dict[str, any], group: Dict[str, any]):
        """
        ### Get learning-rate

        $$\alpha \min \bigg(1, \frac{t}{w}\bigg)$$
        where $w$ is the number of warmup steps.
        """
        # If we are in warmup stage
        if group['warmup'] > state['step']:
            # A linearly increasing learning rate from $0$ to $\alpha$
            return 1e-8 + state['step'] * group['lr'] / group['warmup']
        else:
            # Constant learning rate $\alpha$
            return group['lr']
