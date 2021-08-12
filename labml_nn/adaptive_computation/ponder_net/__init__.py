"""
---
title: "PonderNet: Learning to Ponder"
summary: >
 A PyTorch implementation/tutorial of PonderNet: Learning to Ponder.
---

# PonderNet: Learning to Ponder

This is a [PyTorch](https://pytorch.org) implementation of the paper
[PonderNet: Learning to Ponder](https://papers.labml.ai/paper/2107.05407).

PonderNet adapts the computation based on the input.
It changes the number of steps to take on a recurrent network based on the input.
PonderNet learns this with end-to-end gradient descent.

PonderNet has a step function of the form

$$\hat{y}_n, h_{n+1}, \lambda_n = s(x, h_n)$$

where $x$ is the input, $h_n$ is the state, $\hat{y}_n$ is the prediction at step $n$,
and $\lambda_n$ is the probability of halting (stopping) at current step.

$s$ can be any neural network (e.g. LSTM, MLP, GRU, Attention layer).

The unconditioned probability of halting at step $n$ is then,

$$p_n = \lambda_n \prod_{j=1}^{n-1} (1 - \lambda_j)$$

That is the probability of not being halted at any of the previous steps and halting at step $n$.

During inference, we halt by sampling based on the halting probability $\lambda_n$
 and get the prediction at the halting layer $\hat{y}_n$ as the final output.

During training, we get the predictions from all the layers and calculate the losses for each of them.
And then take the weighted average of the losses based on the probabilities of getting halted at each layer
$p_n$.

The step function is applied to a maximum number of steps donated by $N$.

The overall loss of PonderNet is
\begin{align}
L &= L_{Rec} + \beta L_{Reg} \\
L_{Rec} &= \sum_{n=1}^N p_n \mathcal{L}(y, \hat{y}_n) \\
L_{Reg} &= \mathop{KL} \Big(p_n \Vert p_G(\lambda_p) \Big)
\end{align}

$\mathcal{L}$ is the normal loss function between target $y$ and prediction $\hat{y}_n$.

$\mathop{KL}$ is the [Kullback–Leibler divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence).

$p_G$ is the [Geometric distribution](https://en.wikipedia.org/wiki/Geometric_distribution) parameterized by
$\lambda_p$. *$\lambda_p$ has nothing to do with $\lambda_n$; we are just sticking to same notation as the paper*.
$$Pr_{p_G(\lambda_p)}(X = k) = (1 - \lambda_p)^k \lambda_p$$.

The regularization loss biases the network towards taking $\frac{1}{\lambda_p}$ steps and incentivizes
 non-zero probabilities for all steps; i.e. promotes exploration.

Here is the [training code `experiment.py`](experiment.html) to train a PonderNet on [Parity Task](../parity.html).

[![View Run](https://img.shields.io/badge/labml-experiment-brightgreen)](https://app.labml.ai/run/bfdcea24fa8f11eb89a54df6f6e862b9)
"""

from typing import Tuple

import torch
from torch import nn

from labml_helpers.module import Module


class ParityPonderGRU(Module):
    """
    ## PonderNet with GRU for Parity Task

    This is a simple model that uses a [GRU Cell](https://pytorch.org/docs/stable/generated/torch.nn.GRUCell.html)
    as the step function.

    This model is for the [Parity Task](../parity.html) where the input is a vector of `n_elems`.
    Each element of the vector is either `0`, `1` or `-1` and the output is the parity
    - a binary value that is true if the number of `1`s is odd and false otherwise.

    The prediction of the model is the log probability of the parity being $1$.
    """

    def __init__(self, n_elems: int, n_hidden: int, max_steps: int):
        """
        * `n_elems` is the number of elements in the input vector
        * `n_hidden` is the state vector size of the GRU
        * `max_steps` is the maximum number of steps $N$
        """
        super().__init__()

        self.max_steps = max_steps
        self.n_hidden = n_hidden

        # GRU
        # $$h_{n+1} = s_h(x, h_n)$$
        self.gru = nn.GRUCell(n_elems, n_hidden)
        # $$\hat{y}_n = s_y(h_n)$$
        # We could use a layer that takes the concatenation of $h$ and $x$ as input
        # but we went with this for simplicity.
        self.output_layer = nn.Linear(n_hidden, 1)
        # $$\lambda_n = s_\lambda(h_n)$$
        self.lambda_layer = nn.Linear(n_hidden, 1)
        self.lambda_prob = nn.Sigmoid()
        # An option to set during inference so that computation is actually halted at inference time
        self.is_halt = False

    def __call__(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        * `x` is the input of shape `[batch_size, n_elems]`

        This outputs a tuple of four tensors:

        1. $p_1 \dots p_N$ in a tensor of shape `[N, batch_size]`
        2. $\hat{y}_1 \dots \hat{y}_N$ in a tensor of shape `[N, batch_size]` - the log probabilities of the parity being $1$
        3. $p_m$ of shape `[batch_size]`
        4. $\hat{y}_m$ of shape `[batch_size]` where the computation was halted at step $m$
        """

        #
        batch_size = x.shape[0]

        # We get initial state $h_1 = s_h(x)$
        h = x.new_zeros((x.shape[0], self.n_hidden))
        h = self.gru(x, h)

        # Lists to store $p_1 \dots p_N$ and $\hat{y}_1 \dots \hat{y}_N$
        p = []
        y = []
        # $\prod_{j=1}^{n-1} (1 - \lambda_j)$
        un_halted_prob = h.new_ones((batch_size,))

        # A vector to maintain which samples has halted computation
        halted = h.new_zeros((batch_size,))
        # $p_m$ and $\hat{y}_m$ where the computation was halted at step $m$
        p_m = h.new_zeros((batch_size,))
        y_m = h.new_zeros((batch_size,))

        # Iterate for $N$ steps
        for n in range(1, self.max_steps + 1):
            # The halting probability $\lambda_N = 1$ for the last step
            if n == self.max_steps:
                lambda_n = h.new_ones(h.shape[0])
            # $\lambda_n = s_\lambda(h_n)$
            else:
                lambda_n = self.lambda_prob(self.lambda_layer(h))[:, 0]
            # $\hat{y}_n = s_y(h_n)$
            y_n = self.output_layer(h)[:, 0]

            # $$p_n = \lambda_n \prod_{j=1}^{n-1} (1 - \lambda_j)$$
            p_n = un_halted_prob * lambda_n
            # Update $\prod_{j=1}^{n-1} (1 - \lambda_j)$
            un_halted_prob = un_halted_prob * (1 - lambda_n)

            # Halt based on halting probability $\lambda_n$
            halt = torch.bernoulli(lambda_n) * (1 - halted)

            # Collect $p_n$ and $\hat{y}_n$
            p.append(p_n)
            y.append(y_n)

            # Update $p_m$ and $\hat{y}_m$ based on what was halted at current step $n$
            p_m = p_m * (1 - halt) + p_n * halt
            y_m = y_m * (1 - halt) + y_n * halt

            # Update halted samples
            halted = halted + halt
            # Get next state $h_{n+1} = s_h(x, h_n)$
            h = self.gru(x, h)

            # Stop the computation if all samples have halted
            if self.is_halt and halted.sum() == batch_size:
                break

        #
        return torch.stack(p), torch.stack(y), p_m, y_m


class ReconstructionLoss(Module):
    """
    ## Reconstruction loss

    $$L_{Rec} = \sum_{n=1}^N p_n \mathcal{L}(y, \hat{y}_n)$$

    $\mathcal{L}$ is the normal loss function between target $y$ and prediction $\hat{y}_n$.
    """

    def __init__(self, loss_func: nn.Module):
        """
        * `loss_func` is the loss function $\mathcal{L}$
        """
        super().__init__()
        self.loss_func = loss_func

    def __call__(self, p: torch.Tensor, y_hat: torch.Tensor, y: torch.Tensor):
        """
        * `p` is $p_1 \dots p_N$ in a tensor of shape `[N, batch_size]`
        * `y_hat` is $\hat{y}_1 \dots \hat{y}_N$ in a tensor of shape `[N, batch_size, ...]`
        * `y` is the target of shape `[batch_size, ...]`
        """

        # The total $\sum_{n=1}^N p_n \mathcal{L}(y, \hat{y}_n)$
        total_loss = p.new_tensor(0.)
        # Iterate upto $N$
        for n in range(p.shape[0]):
            # $p_n \mathcal{L}(y, \hat{y}_n)$ for each sample and the mean of them
            loss = (p[n] * self.loss_func(y_hat[n], y)).mean()
            # Add to total loss
            total_loss = total_loss + loss

        #
        return total_loss


class RegularizationLoss(Module):
    """
    ## Regularization loss

    $$L_{Reg} = \mathop{KL} \Big(p_n \Vert p_G(\lambda_p) \Big)$$

    $\mathop{KL}$ is the [Kullback–Leibler divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence).

    $p_G$ is the [Geometric distribution](https://en.wikipedia.org/wiki/Geometric_distribution) parameterized by
    $\lambda_p$. *$\lambda_p$ has nothing to do with $\lambda_n$; we are just sticking to same notation as the paper*.
    $$Pr_{p_G(\lambda_p)}(X = k) = (1 - \lambda_p)^k \lambda_p$$.

    The regularization loss biases the network towards taking $\frac{1}{\lambda_p}$ steps and incentivies non-zero probabilities
    for all steps; i.e. promotes exploration.
    """

    def __init__(self, lambda_p: float, max_steps: int = 1_000):
        """
        * `lambda_p` is $\lambda_p$ - the success probability of geometric distribution
        * `max_steps` is the highest $N$; we use this to pre-compute $p_G(\lambda_p)$
        """
        super().__init__()

        # Empty vector to calculate $p_G(\lambda_p)$
        p_g = torch.zeros((max_steps,))
        # $(1 - \lambda_p)^k$
        not_halted = 1.
        # Iterate upto `max_steps`
        for k in range(max_steps):
            # $$Pr_{p_G(\lambda_p)}(X = k) = (1 - \lambda_p)^k \lambda_p$$
            p_g[k] = not_halted * lambda_p
            # Update $(1 - \lambda_p)^k$
            not_halted = not_halted * (1 - lambda_p)

        # Save $Pr_{p_G(\lambda_p)}$
        self.p_g = nn.Parameter(p_g, requires_grad=False)

        # KL-divergence loss
        self.kl_div = nn.KLDivLoss(reduction='batchmean')

    def __call__(self, p: torch.Tensor):
        """
        * `p` is $p_1 \dots p_N$ in a tensor of shape `[N, batch_size]`
        """
        # Transpose `p` to `[batch_size, N]`
        p = p.transpose(0, 1)
        # Get $Pr_{p_G(\lambda_p)}$ upto $N$ and expand it across the batch dimension
        p_g = self.p_g[None, :p.shape[1]].expand_as(p)

        # Calculate the KL-divergence.
        # *The [PyTorch KL-divergence](https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html)
        # implementation accepts log probabilities.*
        return self.kl_div(p.log(), p_g)
