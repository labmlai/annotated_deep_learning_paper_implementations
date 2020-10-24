"""
# Neural Network Model
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
        """
        ### Initialize

        We need `scope` because we need multiple copies of variables
         for target network and training network.
        """

        super().__init__()
        self.conv = nn.Sequential(
            # The first convolution layer takes a
            # 84x84 frame and produces a 20x20 frame
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),

            # The second convolution layer takes a
            # 20x20 frame and produces a 9x9 frame
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),

            # The third convolution layer takes a
            # 9x9 frame and produces a 7x7 frame
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        # A fully connected layer takes the flattened
        # frame from third convolution layer, and outputs
        # 512 features
        self.lin = nn.Linear(in_features=7 * 7 * 64, out_features=512)

        self.state_score = nn.Sequential(
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=1),
        )
        self.action_score = nn.Sequential(
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=4),
        )

        #
        self.activation = nn.ReLU()

    def __call__(self, obs: torch.Tensor):
        h = self.conv(obs)
        h = h.reshape((-1, 7 * 7 * 64))

        h = self.activation(self.lin(h))

        action_score = self.action_score(h)
        state_score = self.state_score(h)

        # $Q(s, a) =V(s) + \Big(A(s, a) - \frac{1}{|\mathcal{A}|} \sum_{a' \in \mathcal{A}} A(s, a')\Big)$
        action_score_centered = action_score - action_score.mean(dim=-1, keepdim=True)
        q = state_score + action_score_centered

        return q
