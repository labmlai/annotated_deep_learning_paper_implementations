"""
---
title: Fuzzy Tiling Activations
summary: >
  PyTorch implementation and tutorial of Fuzzy Tiling Activations from the
  paper Fuzzy Tiling Activations: A Simple Approach to Learning Sparse Representations Online.
---

# Fuzzy Tiling Activations (FTA)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/activations/fta/experiment.ipynb)

This is a [PyTorch](https://pytorch.org) implementation/tutorial of
[Fuzzy Tiling Activations: A Simple Approach to Learning Sparse Representations Online](https://papers.labml.ai/paper/1911.08068).

Fuzzy tiling activations are a form of sparse activations based on binning.

Binning is classification of a scalar value into a bin based on intervals.
One problem with binning is that it gives zero gradients for most values (except at the boundary of bins).
The other is that binning loses precision if the bin intervals are large.

FTA overcomes these disadvantages.
Instead of hard boundaries like in Tiling Activations, FTA uses soft boundaries
between bins.
This gives non-zero gradients for all or a wide range of values.
And also doesn't lose precision since it's captured in partial values.

#### Tiling Activations

$\mathbf{c}$ is the tiling vector,

$$\mathbf{c} = (l, l + \delta, l + 2 \delta, \dots, u - 2 \delta, u - \delta)$$

where $[l, u]$ is the input range, $\delta$ is the bin size, and $u - l$ is divisible by $\delta$.

Tiling activation is,

$$\phi(z) = 1 - I_+ \big( \max(\mathbf{c} - z, 0) + \max(z - \delta - \mathbf{c}) \big)$$

where $I_+(\cdot)$ is the indicator function which gives $1$ if the input is positive and $0$ otherwise.

Note that tiling activation gives zero gradients because it has hard boundaries.

#### Fuzzy Tiling Activations

The fuzzy indicator function,

$$I_{\eta,+}(x) = I_+(\eta - x) x + I_+ (x - \eta)$$

which increases linearly from $0$ to $1$ when $0 \le x \lt \eta$
and is equal to $1$ for $\eta \le x$.
$\eta$ is a hyper-parameter.

FTA uses this to create soft boundaries between bins.

$$\phi_\eta(z) = 1 - I_{\eta,+} \big( \max(\mathbf{c} - z, 0) + \max(z - \delta - \mathbf{c}, 0) \big)$$

[Here's a simple experiment](experiment.html) that uses FTA in a transformer.
"""

import torch
from torch import nn


class FTA(nn.Module):
    """
    ### Fuzzy Tiling Activations (FTA)
    """

    def __init__(self, lower_limit: float, upper_limit: float, delta: float, eta: float):
        """
        :param lower_limit: is the lower limit $l$
        :param upper_limit: is the upper limit $u$
        :param delta: is the bin size $\delta$
        :param eta: is the parameter $\eta$ that detemines the softness of the boundaries.
        """
        super().__init__()
        # Initialize tiling vector
        # $$\mathbf{c} = (l, l + \delta, l + 2 \delta, \dots, u - 2 \delta, u - \delta)$$
        self.c = nn.Parameter(torch.arange(lower_limit, upper_limit, delta), requires_grad=False)
        # The input vector expands by a factor equal to the number of bins $\frac{u - l}{\delta}$
        self.expansion_factor = len(self.c)
        # $\delta$
        self.delta = delta
        # $\eta$
        self.eta = eta

    def fuzzy_i_plus(self, x: torch.Tensor):
        """
        #### Fuzzy indicator function

        $$I_{\eta,+}(x) = I_+(\eta - x) x + I_+ (x - \eta)$$
        """
        return (x <= self.eta) * x + (x > self.eta)

    def forward(self, z: torch.Tensor):
        # Add another dimension of size $1$.
        # We will expand this into bins.
        z = z.view(*z.shape, 1)

        # $$\phi_\eta(z) = 1 - I_{\eta,+} \big( \max(\mathbf{c} - z, 0) + \max(z - \delta - \mathbf{c}, 0) \big)$$
        z = 1. - self.fuzzy_i_plus(torch.clip(self.c - z, min=0.) + torch.clip(z - self.delta - self.c, min=0.))

        # Reshape back to original number of dimensions.
        # The last dimension size gets expanded by the number of bins, $\frac{u - l}{\delta}$.
        return z.view(*z.shape[:-2], -1)


def _test():
    """
    #### Code to test the FTA module
    """
    from labml.logger import inspect

    # Initialize
    a = FTA(-10, 10, 2., 0.5)
    # Print $\mathbf{c}$
    inspect(a.c)
    # Print number of bins $\frac{u - l}{\delta}$
    inspect(a.expansion_factor)

    # Input $z$
    z = torch.tensor([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9., 10., 11.])
    # Print $z$
    inspect(z)
    # Print $\phi_\eta(z)$
    inspect(a(z))


if __name__ == '__main__':
    _test()
