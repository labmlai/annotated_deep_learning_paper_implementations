"""
---
title: Deep Q Networks (DQN)
summary: >
  This is a PyTorch implementation/tutorial of Deep Q Networks (DQN) from paper
  Playing Atari with Deep Reinforcement Learning.
  This includes dueling network architecture, a prioritized replay buffer and
  double-Q-network training.
---


# Deep Q Networks (DQN)

This is a [PyTorch](https://pytorch.org) implementation of paper
 [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)
 along with [Dueling Network](model.html), [Prioritized Replay](replay_buffer.html)
 and Double Q Network.

Here are the [experiment](experiment.html) and [model](model.html) implementation.

\(
   \def\green#1{{\color{yellowgreen}{#1}}}
\)

"""

from typing import Tuple

import torch
from torch import nn

from labml import tracker
from labml_helpers.module import Module
from labml_nn.rl.dqn.replay_buffer import ReplayBuffer


class QFuncLoss(Module):
    """
    ## Train the model

    We want to find optimal action-value function.

    \begin{align}
        Q^*(s,a) &= \max_\pi \mathbb{E} \Big[
            r_t + \gamma r_{t + 1} + \gamma^2 r_{t + 2} + ... | s_t = s, a_t = a, \pi
        \Big]
    \\
        Q^*(s,a) &= \mathop{\mathbb{E}}_{s' \sim \large{\varepsilon}} \Big[
            r + \gamma \max_{a'} Q^* (s', a') | s, a
        \Big]
    \end{align}

    ### Target network 🎯
    In order to improve stability we use experience replay that randomly sample
    from previous experience $U(D)$. We also use a Q network
    with a separate set of paramters $\color{orangle}{\theta_i^{-}}$ to calculate the target.
    $\color{orangle}{\theta_i^{-}}$ is updated periodically.
    This is according to paper
    [Human Level Control Through Deep Reinforcement Learning](https://deepmind.com/research/dqn/).

    So the loss function is,
    $$
    \mathcal{L}_i(\theta_i) = \mathop{\mathbb{E}}_{(s,a,r,s') \sim U(D)}
    \bigg[
        \Big(
            r + \gamma \max_{a'} Q(s', a'; \color{orange}{\theta_i^{-}}) - Q(s,a;\theta_i)
        \Big) ^ 2
    \bigg]
    $$

    ### Double $Q$-Learning
    The max operator in the above calculation uses same network for both
    selecting the best action and for evaluating the value.
    That is,
    $$
    \max_{a'} Q(s', a'; \theta) = \color{cyan}{Q}
    \Big(
        s', \mathop{\operatorname{argmax}}_{a'}
        \color{cyan}{Q}(s', a'; \color{cyan}{\theta}); \color{cyan}{\theta}
    \Big)
    $$
    We use [double Q-learning](https://arxiv.org/abs/1509.06461), where
    the $\operatorname{argmax}$ is taken from $\color{cyan}{\theta_i}$ and
    the value is taken from $\color{orange}{\theta_i^{-}}$.

    And the loss function becomes,
    \begin{align}
        \mathcal{L}_i(\theta_i) = \mathop{\mathbb{E}}_{(s,a,r,s') \sim U(D)}
        \Bigg[
            \bigg(
                &r + \gamma \color{orange}{Q}
                \Big(
                    s',
                    \mathop{\operatorname{argmax}}_{a'}
                        \color{cyan}{Q}(s', a'; \color{cyan}{\theta_i}); \color{orange}{\theta_i^{-}}
                \Big)
                \\
                - &Q(s,a;\theta_i)
            \bigg) ^ 2
        \Bigg]
    \end{align}
    """

    def __init__(self, gamma: float):
        super().__init__()
        self.gamma = gamma
        self.huber_loss = nn.SmoothL1Loss(reduction='none')

    def __call__(self, q: torch.Tensor, action: torch.Tensor, double_q: torch.Tensor,
                 target_q: torch.Tensor, done: torch.Tensor, reward: torch.Tensor,
                 weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        * `q` - $Q(s;\theta_i)$
        * `action` - $a$
        * `double_q` - $\color{cyan}Q(s';\color{cyan}{\theta_i})$
        * `target_q` - $\color{orange}Q(s';\color{orange}{\theta_i^{-}})$
        * `done` - whether the game ended after taking the action
        * `reward` - $r$
        * `weights` - weights of the samples from prioritized experienced replay
        """

        # $Q(s,a;\theta_i)$
        q_sampled_action = q.gather(-1, action.to(torch.long).unsqueeze(-1)).squeeze(-1)
        tracker.add('q_sampled_action', q_sampled_action)

        # Gradients shouldn't propagate gradients
        # $$r + \gamma \color{orange}{Q}
        #                 \Big(s',
        #                     \mathop{\operatorname{argmax}}_{a'}
        #                         \color{cyan}{Q}(s', a'; \color{cyan}{\theta_i}); \color{orange}{\theta_i^{-}}
        #                 \Big)$$
        with torch.no_grad():
            # Get the best action at state $s'$
            # $$\mathop{\operatorname{argmax}}_{a'}
            #  \color{cyan}{Q}(s', a'; \color{cyan}{\theta_i})$$
            best_next_action = torch.argmax(double_q, -1)
            # Get the q value from the target network for the best action at state $s'$
            # $$\color{orange}{Q}
            # \Big(s',\mathop{\operatorname{argmax}}_{a'}
            # \color{cyan}{Q}(s', a'; \color{cyan}{\theta_i}); \color{orange}{\theta_i^{-}}
            # \Big)$$
            best_next_q_value = target_q.gather(-1, best_next_action.unsqueeze(-1)).squeeze(-1)

            # Calculate the desired Q value.
            # We multiply by `(1 - done)` to zero out
            # the next state Q values if the game ended.
            #
            # $$r + \gamma \color{orange}{Q}
            #                 \Big(s',
            #                     \mathop{\operatorname{argmax}}_{a'}
            #                         \color{cyan}{Q}(s', a'; \color{cyan}{\theta_i}); \color{orange}{\theta_i^{-}}
            #                 \Big)$$
            q_update = reward + self.gamma * best_next_q_value * (1 - done)
            tracker.add('q_update', q_update)

            # Temporal difference error $\delta$ is used to weigh samples in replay buffer
            td_error = q_sampled_action - q_update
            tracker.add('td_error', td_error)

        # We take [Huber loss](https://en.wikipedia.org/wiki/Huber_loss) instead of
        # mean squared error loss because it is less sensitive to outliers
        losses = self.huber_loss(q_sampled_action, q_update)
        # Get weighted means
        loss = torch.mean(weights * losses)
        tracker.add('loss', loss)

        return td_error, loss
