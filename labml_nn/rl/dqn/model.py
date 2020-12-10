"""
---
title: Deep Q Network (DQN) Model
summary: Implementation of neural network model for Deep Q Network (DQN).
---

# Deep Q Network (DQN) Model
"""

import torch
from torch import nn

from labml_helpers.module import Module


class Model(Module):
    """
    ## Dueling Network ⚔️ Model for $Q$ Values

    We are using a [dueling network](https://arxiv.org/abs/1511.06581)
     to calculate Q-values.
    Intuition behind dueling network architecture is that in most states
     the action doesn't matter,
    and in some states the action is significant. Dueling network allows
     this to be represented very well.

    \begin{align}
        Q^\pi(s,a) &= V^\pi(s) + A^\pi(s, a)
        \\
        \mathop{\mathbb{E}}_{a \sim \pi(s)}
         \Big[
          A^\pi(s, a)
         \Big]
        &= 0
    \end{align}

    So we create two networks for $V$ and $A$ and get $Q$ from them.
    $$
        Q(s, a) = V(s) +
        \Big(
            A(s, a) - \frac{1}{|\mathcal{A}|} \sum_{a' \in \mathcal{A}} A(s, a')
        \Big)
    $$
    We share the initial layers of the $V$ and $A$ networks.
    """

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            # The first convolution layer takes a
            # $84\times84$ frame and produces a $20\times20$ frame
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),

            # The second convolution layer takes a
            # $20\times20$ frame and produces a $9\times9$ frame
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),

            # The third convolution layer takes a
            # $9\times9$ frame and produces a $7\times7$ frame
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        # A fully connected layer takes the flattened
        # frame from third convolution layer, and outputs
        # $512$ features
        self.lin = nn.Linear(in_features=7 * 7 * 64, out_features=512)
        self.activation = nn.ReLU()

        # This head gives the state value $V$
        self.state_value = nn.Sequential(
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=1),
        )
        # This head gives the action value $A$
        self.action_value = nn.Sequential(
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=4),
        )

    def __call__(self, obs: torch.Tensor):
        # Convolution
        h = self.conv(obs)
        # Reshape for linear layers
        h = h.reshape((-1, 7 * 7 * 64))

        # Linear layer
        h = self.activation(self.lin(h))

        # $A$
        action_value = self.action_value(h)
        # $V$
        state_value = self.state_value(h)

        # $A(s, a) - \frac{1}{|\mathcal{A}|} \sum_{a' \in \mathcal{A}} A(s, a')$
        action_score_centered = action_value - action_value.mean(dim=-1, keepdim=True)
        # $Q(s, a) =V(s) + \Big(A(s, a) - \frac{1}{|\mathcal{A}|} \sum_{a' \in \mathcal{A}} A(s, a')\Big)$
        q = state_value + action_score_centered

        return q
