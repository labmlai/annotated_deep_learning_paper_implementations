r"""
---
title: Wasserstein GAN (WGAN)
summary: A simple PyTorch implementation/tutorial of Wasserstein Generative Adversarial Networks (WGAN) loss functions.
---

# Wasserstein GAN (WGAN)

This is an implementation of
[Wasserstein GAN](https://arxiv.org/abs/1701.07875).

The original GAN loss is based on Jensen-Shannon (JS) divergence
between the real distribution $\mathbb{P}_r$ and generated distribution $\mathbb{P}_g$.
The Wasserstein GAN is based on Earth Mover distance between these distributions.

$$
W(\mathbb{P}_r, \mathbb{P}_g) =
 \underset{\gamma \in \Pi(\mathbb{P}_r, \mathbb{P}_g)} {\mathrm{inf}}
 \mathbb{E}_{(x,y) \sim \gamma}
 \Vert x - y \Vert
$$

$\Pi(\mathbb{P}_r, \mathbb{P}_g)$ is the set of all joint distributions, whose
marginal probabilities are $\gamma(x, y)$.

$\mathbb{E}_{(x,y) \sim \gamma} \Vert x - y \Vert$ is the earth mover distance for
a given joint distribution ($x$ and $y$ are probabilities).

So $W(\mathbb{P}_r, \mathbb{P}g)$ is equal to the least earth mover distance for
any joint distribution between the real distribution $\mathbb{P}_r$ and generated distribution $\mathbb{P}_g$.

The paper shows that Jensen-Shannon (JS) divergence and other measures for the difference between two probability
distributions are not smooth. And therefore if we are doing gradient descent on one of the probability
distributions (parameterized) it will not converge.

Based on Kantorovich-Rubinstein duality,
$$
W(\mathbb{P}_r, \mathbb{P}_g) =
 \underset{\Vert f \Vert_L \le 1} {\mathrm{sup}}
 \mathbb{E}_{x \sim \mathbb{P}_r} [f(x)]- \mathbb{E}_{x \sim \mathbb{P}_g} [f(x)]
$$

where $\Vert f \Vert_L \le 1$ are all 1-Lipschitz functions.

That is, it is equal to the greatest difference
$$\mathbb{E}_{x \sim \mathbb{P}_r} [f(x)] - \mathbb{E}_{x \sim \mathbb{P}_g} [f(x)]$$
among all 1-Lipschitz functions.

For $K$-Lipschitz functions,
$$
W(\mathbb{P}_r, \mathbb{P}_g) =
 \underset{\Vert f \Vert_L \le K} {\mathrm{sup}}
 \mathbb{E}_{x \sim \mathbb{P}_r} \Bigg[\frac{1}{K} f(x) \Bigg]
  - \mathbb{E}_{x \sim \mathbb{P}_g} \Bigg[\frac{1}{K} f(x) \Bigg]
$$

If all $K$-Lipschitz functions can be represented as $f_w$ where $f$ is parameterized by
$w \in \mathcal{W}$,

$$
K \cdot W(\mathbb{P}_r, \mathbb{P}_g) =
 \max_{w \in \mathcal{W}}
 \mathbb{E}_{x \sim \mathbb{P}_r} [f_w(x)]- \mathbb{E}_{x \sim \mathbb{P}_g} [f_w(x)]
$$

If $(\mathbb{P}_{g})$ is represented by a generator $$g_\theta (z)$$ and $z$ is from a known
distribution $z \sim p(z)$,

$$
K \ cdot W(\mathbb{P}_r, \mathbb{P}_\theta) =
 \max_{w \in \mathcal{W}}
 \mathbb{E}_{x \sim \mathbb{P}_r} [f_w(x)]- \mathbb{E}_{z \sim p(z)} [f_w(g_\theta(z))]
$$

Now to converge $g_\theta$ with $\mathbb{P}_{r}$ we can gradient descent on $\theta$
to minimize above formula.

Similarly we can find $\max_{w \in \mathcal{W}}$ by ascending on $w$,
while keeping $K$ bounded. *One way to keep $K$ bounded is to clip all weights in the neural
network that defines $f$ clipped within a range.*

Here is the code to try this on a [simple MNIST generation experiment](experiment.html).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/gan/wasserstein/experiment.ipynb)
"""

import torch.utils.data
from torch.nn import functional as F

from labml_helpers.module import Module


class DiscriminatorLoss(Module):
    """
    ## Discriminator Loss

    We want to find $w$ to maximize
    $$\mathbb{E}_{x \sim \mathbb{P}_r} [f_w(x)]- \mathbb{E}_{z \sim p(z)} [f_w(g_\theta(z))]$$,
    so we minimize,
    $$-\frac{1}{m} \sum_{i=1}^m f_w \big(x^{(i)} \big) +
     \frac{1}{m} \sum_{i=1}^m f_w \big( g_\theta(z^{(i)}) \big)$$
    """

    def __call__(self, f_real: torch.Tensor, f_fake: torch.Tensor):
        """
        * `f_real` is $f_w(x)$
        * `f_fake` is $f_w(g_\theta(z))$

        This returns the a tuple with losses for $f_w(x)$ and $f_w(g_\theta(z))$,
        which are later added.
        They are kept separate for logging.
        """

        # We use ReLUs to clip the loss to keep $f \in [-1, +1]$ range.
        return F.relu(1 - f_real).mean(), F.relu(1 + f_fake).mean()


class GeneratorLoss(Module):
    """
    ## Generator Loss

    We want to find $\theta$ to minimize
    $$\mathbb{E}_{x \sim \mathbb{P}_r} [f_w(x)]- \mathbb{E}_{z \sim p(z)} [f_w(g_\theta(z))]$$
    The first component is independent of $\theta$,
    so we minimize,
    $$-\frac{1}{m} \sum_{i=1}^m f_w \big( g_\theta(z^{(i)}) \big)$$

    """

    def __call__(self, f_fake: torch.Tensor):
        """
        * `f_fake` is $f_w(g_\theta(z))$
        """
        return -f_fake.mean()
