"""
This is a Deep Q Learning implementation with:
* Double Q Network
* Dueling Network
* Prioritized Replay
"""

from typing import Tuple

import torch

from labml import tracker
from labml_helpers.module import Module
from labml_nn.rl.dqn.replay_buffer import ReplayBuffer


class QFuncLoss(Module):
    def __init__(self, gamma: float):
        super().__init__()
        self.gamma = gamma

    def __call__(self, q: torch.Tensor,
                 action: torch.Tensor,
                 double_q: torch.Tensor,
                 target_q: torch.Tensor,
                 done: torch.Tensor,
                 reward: torch.Tensor,
                 weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        q_sampled_action = q.gather(-1, action.to(torch.long).unsqueeze(-1)).squeeze(-1)
        tracker.add('q_sampled_action', q_sampled_action)

        with torch.no_grad():
            best_next_action = torch.argmax(double_q, -1)
            best_next_q_value = target_q.gather(-1, best_next_action.unsqueeze(-1)).squeeze(-1)

            best_next_q_value *= (1 - done)

            q_update = reward + self.gamma * best_next_q_value
            tracker.add('q_update', q_update)

            td_error = q_sampled_action - q_update
            tracker.add('td_error', td_error)

        # Huber loss
        losses = torch.nn.functional.smooth_l1_loss(q_sampled_action, q_update, reduction='none')
        loss = torch.mean(weights * losses)
        tracker.add('loss', loss)

        return td_error, loss

