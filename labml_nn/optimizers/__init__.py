"""
---
title: Optimizers
summary: >
 A set of PyTorch implementations/tutorials of popular gradient descent based optimizers.
 Currently includes Adam, AMSGrad and RAdam optimizers.
---

# Optimizers

## Optimizer Implementations
* [Adam Optimizer](adam.html)
* [AMSGrad Optimizer](amsgrad.html)
* [Adam Optimizer with warmup](adam_warmup.html)
* [Noam Optimizer](noam.html)
* [Rectified Adam Optimizer](radam.html)
* [AdaBelief Optimizer](ada_belief.html)

This [MNIST example](mnist_experiment.html) uses these optimizers.

## Generic Adaptive Optimizer Base class and Weight Decay
This file defines a common base class for *Adam* and extensions of it.
The base class helps use implement other optimizers with minimal code
because of re-usability.

We also define a special class for L2 weight decay, so that we don't
have to implement it inside each of the optimizers,
and can easily extend to other weight decays like L1 without
changing the optimizers.

Here are some concepts on PyTorch optimizers:

### Parameter groups
PyTorch optimizers group parameters into sets called groups.
Each group can have it's own hyper-parameters like learning rates.

In most common cases there will be only one group.
This is when you initialize your optimizer with,
```python
Optimizer(model.parameters())
```

You can define multiple parameter groups when initializing the optimizer:
```python
Optimizer([{'params': model1.parameters()}, {'params': model2.parameters(), 'lr': 2}])
```
Here we pass a list of groups. Each group is a dictionary with it's parameters under the key 'params'.
You specify any hyper-parameters as well. If the hyper parameters are not defined they will default
to the optimizer level defaults.

You can access (and even change) these groups, and their hyper-parameters with `optimizer.param_groups`.
Most learning rate schedule implementations I've come across do access this and change 'lr'.

### States
Optimizer maintains states (a dictionary) for each parameter (a tensor), in a dictionary `optimizer.state`.
This is where the optimizer maintains things like exponential averages.
"""

from typing import Dict, Tuple, Any

import torch
from torch import nn
from torch.optim.optimizer import Optimizer


class GenericAdaptiveOptimizer(Optimizer):
    """
    ## Base class for *Adam* and extensions
    """

    def __init__(self, params, defaults: Dict[str, Any], lr: float, betas: Tuple[float, float], eps: float):
        """
        ### Initialize

        * `params` is the collection of parameters or set of parameter groups.
        * `defaults` a dictionary of default hyper-parameters
        * 'lr` is the learning rate, $\alpha$
        * `betas` is the tuple $(\beta_1, \beta_2)$
        * `eps` is $\epsilon$
        """

        # Check the hyper-parameters
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")

        # Add the hyper-parameters to the defaults
        defaults.update(dict(lr=lr, betas=betas, eps=eps))
        # Initialize the PyTorch optimizer.
        # This will create parameter groups with the default hyper-parameters
        super().__init__(params, defaults)

    def init_state(self, state: Dict[str, any], group: Dict[str, any], param: nn.Parameter):
        """
        ### Initialize state for a given parameter tensor

        This should be overridden with code to initialize `state` for parameters `param`.
        `group` is the parameter group dictionary to which `param` belongs.
        """
        pass

    def step_param(self, state: Dict[str, any], group: Dict[str, any], grad: torch.Tensor, param: torch.Tensor):
        """
        ### Take optimizer step on a parameter tensor

        This should be overridden and take the optimization step on `param` tensor $\theta$,
        where `grad` is the gradient for that parameter, $g_t$,
        `state` is the optimizer state dictionary for that parameter, and
        `group` is the parameter group dictionary `param` belongs to.
        """
        pass

    @torch.no_grad()
    def step(self, closure=None):
        """
        ### Optimizer step

        We have created a template method that does the common stuff every *Adam* based optimizer needs.
        """
        # Calculate loss.
        #
        # 🤔 I'm not sure when you need this. I guess it's if you define a function that
        # calculates the loss, does `loss.backward` and return the loss, instead of calling
        # it on your own you could pass it to `optimizer.step`. 🤷‍♂️
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Iterate through the parameter groups
        for group in self.param_groups:
            # Iterate through the parameters in the parameter group
            for param in group['params']:
                # Skip if the parameter has no gradient
                if param.grad is None:
                    continue
                # Get the gradient tensor
                grad = param.grad.data
                # We don't handle sparse gradients
                if grad.is_sparse:
                    raise RuntimeError('GenericAdaptiveOptimizer does not support sparse gradients,'
                                       ' please consider SparseAdam instead')

                # Get the state for the parameter
                state = self.state[param]

                # Initialize the state if state is uninitialized
                if len(state) == 0:
                    self.init_state(state, group, param)

                # Take the optimization step on the parameter
                self.step_param(state, group, grad, param)

        # Return the loss, calculated from closure
        return loss


class WeightDecay:
    """
    ## L2 Weight decay
    """

    def __init__(self, weight_decay: float = 0., weight_decouple: bool = True, absolute: bool = False):
        """
        ### Initialize weight decay

        * `weight_decay` is the decay coefficient
        * `weight_decouple` is a flag indicating whether to add the weight decay to the gradient or directly
        decay from the parameter. If added to the  gradient it will go through the normal optimizer update.
        * `absolute` this flag indicates whether the weight decay coefficient is absolute. This is applicable
        when the decay is performed directly on the parameter. If this is false the actual decay is
        `weight_decay` * `learning_rate`.
        """
        # Check hyper-parameters
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        self.absolute = absolute
        self.weight_decouple = weight_decouple
        self.weight_decay = weight_decay

    def defaults(self):
        """
        Return defaults for parameter groups
        """
        return dict(weight_decay=self.weight_decay)

    def __call__(self, param: torch.nn.Parameter, grad: torch.Tensor, group: Dict[str, any]):
        """
        ### Perform weight decay and return the gradient
        """

        # If we are doing the decay on the parameter directly
        if self.weight_decouple:
            # If the weight decay coefficient is absolute
            if self.absolute:
                param.data.mul_(1.0 - group['weight_decay'])
            # Otherwise,
            else:
                param.data.mul_(1.0 - group['lr'] * group['weight_decay'])
            # Return the unmodified gradient
            return grad
        else:
            if group['weight_decay'] != 0:
                # Add the weight decay to the gradient and return the modified gradient
                return grad.add(param.data, alpha=group['weight_decay'])
            else:
                return grad
