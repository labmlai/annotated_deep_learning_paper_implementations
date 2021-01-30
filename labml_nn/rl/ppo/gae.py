"""
---
title: Generalized Advantage Estimation (GAE)
summary: A PyTorch implementation/tutorial of Generalized Advantage Estimation (GAE).
---

# Generalized Advantage Estimation (GAE)

This is a [PyTorch](https://pytorch.org) implementation of paper
[Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438).
"""

import numpy as np


class GAE:
    def __init__(self, n_workers: int, worker_steps: int, gamma: float, lambda_: float):
        self.lambda_ = lambda_
        self.gamma = gamma
        self.worker_steps = worker_steps
        self.n_workers = n_workers

    def __call__(self, done: np.ndarray, rewards: np.ndarray, values: np.ndarray) -> np.ndarray:
        """
        ### Calculate advantages
        \begin{align}
        \hat{A_t^{(1)}} &= r_t + \gamma V(s_{t+1}) - V(s)
        \\
        \hat{A_t^{(2)}} &= r_t + \gamma r_{t+1} +\gamma^2 V(s_{t+2}) - V(s)
        \\
        ...
        \\
        \hat{A_t^{(\infty)}} &= r_t + \gamma r_{t+1} +\gamma^2 r_{t+1} + ... - V(s)
        \end{align}

        $\hat{A_t^{(1)}}$ is high bias, low variance whilst
        $\hat{A_t^{(\infty)}}$ is unbiased, high variance.

        We take a weighted average of $\hat{A_t^{(k)}}$ to balance bias and variance.
        This is called Generalized Advantage Estimation.
        $$\hat{A_t} = \hat{A_t^{GAE}} = \sum_k w_k \hat{A_t^{(k)}}$$
        We set $w_k = \lambda^{k-1}$, this gives clean calculation for
        $\hat{A_t}$

        \begin{align}
        \delta_t &= r_t + \gamma V(s_{t+1}) - V(s_t)$
        \\
        \hat{A_t} &= \delta_t + \gamma \lambda \delta_{t+1} + ... +
                             (\gamma \lambda)^{T - t + 1} \delta_{T - 1}$
        \\
        &= \delta_t + \gamma \lambda \hat{A_{t+1}}
        \end{align}
        """

        # advantages table
        advantages = np.zeros((self.n_workers, self.worker_steps), dtype=np.float32)
        last_advantage = 0

        # $V(s_{t+1})$
        last_value = values[:, -1]

        for t in reversed(range(self.worker_steps)):
            # mask if episode completed after step $t$
            mask = 1.0 - done[:, t]
            last_value = last_value * mask
            last_advantage = last_advantage * mask
            # $\delta_t$
            delta = rewards[:, t] + self.gamma * last_value - values[:, t]

            # $\hat{A_t} = \delta_t + \gamma \lambda \hat{A_{t+1}}$
            last_advantage = delta + self.gamma * self.lambda_ * last_advantage

            # note that we are collecting in reverse order.
            # *My initial code was appending to a list and
            #   I forgot to reverse it later.
            # It took me around 4 to 5 hours to find the bug.
            # The performance of the model was improving
            #  slightly during initial runs,
            #  probably because the samples are similar.*
            advantages[:, t] = last_advantage

            last_value = values[:, t]

        return advantages
