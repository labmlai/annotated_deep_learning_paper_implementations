"""
---
title: Adam optimizer with warm-up and cosine decay
summary: A PyTorch implementation/tutorial of Adam optimizer with warm-up and cosine decay for GPT.
---

# Adam Optimizer with Warmup and Cosine Decay

This extends [AMSGrad optimizer](adam.html) and adds a warmup stage.
"""
import math
from typing import Dict

from labml_nn.optimizers import WeightDecay
from labml_nn.optimizers.amsgrad import AMSGrad


class AdamWarmupCosineDecay(AMSGrad):
    """
    <a id="EmbeddingsWithPositionalEncoding">
    ## Adam Optimizer with Warmup and Cosine Decay
    </a>

    This class extends from AMSGrad optimizer defined in [`amsgrad.py`](amsgrad.html).
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-16,
                 weight_decay: WeightDecay = WeightDecay(),
                 optimized_update: bool = True,
                 amsgrad=False, warmup=0, total_steps=1e10, defaults=None):
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
        * `total_steps` total number of steps. Cosine decay reaches 0 at this,
        but stays at 10% of `lr` because we take $\alpha * \max(0.1, decay)$
        * `defaults` is a dictionary of default for group values.
         This is useful when you want to extend the class `AdamWarmup`.
        """

        defaults = {} if defaults is None else defaults
        defaults.update(dict(warmup=warmup, total_steps=total_steps))
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
            progress = (state['step'] - group['warmup']) / max(1, group['total_steps'] - group['warmup'])
            return group['lr'] * max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))


def _test_lr():
    """
    ### Plot learning rate for different warmups and model sizes

    ![Plot of learning rate](noam_lr.png)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from torch import nn

    model = nn.Linear(10, 10)
    opt = AdamWarmupCosineDecay(model.parameters(), warmup=5000, lr=1e-4, total_steps=4e6)
    steps = 20_000
    plt.plot(np.arange(1, steps), [opt.get_lr({'step': i}, opt.defaults) for i in range(1, steps)])
    plt.legend(["5000:4e6", "5000:2e6", "5000:1e6"])
    plt.title("Learning Rate")
    plt.show()

    steps = int(6e6)
    step_size = 1000
    plt.plot(np.arange(1, steps, step_size), [opt.get_lr({'step': i}, opt.defaults) for i in range(1, steps, step_size)])
    plt.legend(["5000:4e6", "5000:2e6", "5000:1e6"])
    plt.title("Learning Rate")
    plt.show()


if __name__ == '__main__':
    _test_lr()
