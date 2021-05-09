r"""
---
title: Gradient Penalty for Wasserstein GAN (WGAN-GP)
summary: >
 An annotated PyTorch implementation/tutorial of
  Improved Training of Wasserstein GANs.
---

# Gradient Penalty for Wasserstein GAN (WGAN-GP)

This is an implementation of
[Improved Training of Wasserstein GANs](https://arxiv.org/abs/1704.00028).

[WGAN](../index.html) suggests clipping weights to enforce Lipschitz constraint
on the discriminator network (critic).
This and other weight constraints like L2 norm clipping, weight normalization,
L1, L2 weight decay have problems:

1. Limiting the capacity of the discriminator
2. Exploding and vanishing gradients (without [Batch Normalization](../../../normalization/batch_norm/index.html)).

The paper [Improved Training of Wasserstein GANs](https://arxiv.org/abs/1704.00028)
proposal a better way to improve Lipschitz constraint, a gradient penalty.

$$\mathcal{L}_{GP} = \lambda \underset{\hat{x} \sim \mathbb{P}_{\hat{x}}}{\mathbb{E}}
\Big[ \big(\Vert \nabla_{\hat{x}} D(\hat{x}) \Vert_2 - 1\big)^2 \Big]
$$

where $\lambda$ is the penalty weight and

\begin{align}
x &\sim \mathbb{P}_r \\
z &\sim p(z) \\
\epsilon &\sim U[0,1] \\
\tilde{x} &\leftarrow G_\theta (z) \\
\hat{x} &\leftarrow \epsilon x + (1 - \epsilon) \tilde{x}
\end{align}

That is we try to keep the gradient norm $\Vert \nabla_{\hat{x}} D(\hat{x}) \Vert_2$ close to $1$.

In this implementation we set $\epsilon = 1$.

Here is the [code for an experiment](experiment.html) that uses gradient penalty.
"""

import torch
import torch.autograd

from labml_helpers.module import Module


class GradientPenalty(Module):
    """
    ## Gradient Penalty
    """

    def __call__(self, x: torch.Tensor, f: torch.Tensor):
        """
        * `x` is $x \sim \mathbb{P}_r$
        * `f` is $D(x)$

        $\hat{x} \leftarrow x$
        since we set $\epsilon = 1$ for this implementation.
        """

        # Get batch size
        batch_size = x.shape[0]

        # Calculate gradients of $D(x)$ with respect to $x$.
        # `grad_outputs` is set to ones since we want the gradients of $D(x)$,
        # and we need to create and retain graph since we have to compute gradients
        # with respect to weight on this loss.
        gradients, *_ = torch.autograd.grad(outputs=f,
                                            inputs=x,
                                            grad_outputs=f.new_ones(f.shape),
                                            create_graph=True)

        # Reshape gradients to calculate the norm
        gradients = gradients.reshape(batch_size, -1)
        # Calculate the norm $\Vert \nabla_{\hat{x}} D(\hat{x}) \Vert_2$
        norm = gradients.norm(2, dim=-1)
        # Return the loss $\big(\Vert \nabla_{\hat{x}} D(\hat{x}) \Vert_2 - 1\big)^2$
        return torch.mean((norm - 1) ** 2)
