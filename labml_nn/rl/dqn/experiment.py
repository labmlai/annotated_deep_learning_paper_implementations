"""
---
title: DQN Experiment with Atari Breakout
summary: Implementation of DQN experiment with Atari Breakout
---

# DQN Experiment with Atari Breakout

This experiment trains a Deep Q Network (DQN) to play Atari Breakout game on OpenAI Gym.
It runs the [game environments on multiple processes](../game.html) to sample efficiently.
"""

import numpy as np
import torch

from labml import tracker, experiment, logger, monit
from labml_helpers.schedule import Piecewise
from labml_nn.rl.dqn import QFuncLoss
from labml_nn.rl.dqn.model import Model
from labml_nn.rl.dqn.replay_buffer import ReplayBuffer
from labml_nn.rl.game import Worker

# Select device
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")


def obs_to_torch(obs: np.ndarray) -> torch.Tensor:
    """Scale observations from `[0, 255]` to `[0, 1]`"""
    return torch.tensor(obs, dtype=torch.float32, device=device) / 255.


class Trainer:
    """
    ## Trainer
    """

    def __init__(self):
        # #### Configurations

        # number of workers
        self.n_workers = 8
        # steps sampled on each update
        self.worker_steps = 4
        # number of training iterations
        self.train_epochs = 8

        # number of updates
        self.updates = 1_000_000
        # size of mini batch for training
        self.mini_batch_size = 32

        # exploration as a function of updates
        self.exploration_coefficient = Piecewise(
            [
                (0, 1.0),
                (25_000, 0.1),
                (self.updates / 2, 0.01)
            ], outside_value=0.01)

        # update target network every 250 update
        self.update_target_model = 250

        # $\beta$ for replay buffer as a function of updates
        self.prioritized_replay_beta = Piecewise(
            [
                (0, 0.4),
                (self.updates, 1)
            ], outside_value=1)

        # Replay buffer with $\alpha = 0.6$. Capacity of the replay buffer must be a power of 2.
        self.replay_buffer = ReplayBuffer(2 ** 14, 0.6)

        # Model for sampling and training
        self.model = Model().to(device)
        # target model to get $\color{orange}Q(s';\color{orange}{\theta_i^{-}})$
        self.target_model = Model().to(device)

        # create workers
        self.workers = [Worker(47 + i) for i in range(self.n_workers)]

        # initialize tensors for observations
        self.obs = np.zeros((self.n_workers, 4, 84, 84), dtype=np.uint8)
        for worker in self.workers:
            worker.child.send(("reset", None))
        for i, worker in enumerate(self.workers):
            self.obs[i] = worker.child.recv()

        # loss function
        self.loss_func = QFuncLoss(0.99)
        # optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=2.5e-4)

    def _sample_action(self, q_value: torch.Tensor, exploration_coefficient: float):
        """
        #### $\epsilon$-greedy Sampling
        When sampling actions we use a $\epsilon$-greedy strategy, where we
        take a greedy action with probabiliy $1 - \epsilon$ and
        take a random action with probability $\epsilon$.
        We refer to $\epsilon$ as `exploration_coefficient`.
        """

        # Sampling doesn't need gradients
        with torch.no_grad():
            # Sample the action with highest Q-value. This is the greedy action.
            greedy_action = torch.argmax(q_value, dim=-1)
            # Uniformly sample and action
            random_action = torch.randint(q_value.shape[-1], greedy_action.shape, device=q_value.device)
            # Whether to chose greedy action or the random action
            is_choose_rand = torch.rand(greedy_action.shape, device=q_value.device) < exploration_coefficient
            # Pick the action based on `is_choose_rand`
            return torch.where(is_choose_rand, random_action, greedy_action).cpu().numpy()

    def sample(self, exploration_coefficient: float):
        """### Sample data"""

        # This doesn't need gradients
        with torch.no_grad():
            # Sample `worker_steps`
            for t in range(self.worker_steps):
                # Get Q_values for the current observation
                q_value = self.model(obs_to_torch(self.obs))
                # Sample actions
                actions = self._sample_action(q_value, exploration_coefficient)

                # Run sampled actions on each worker
                for w, worker in enumerate(self.workers):
                    worker.child.send(("step", actions[w]))

                # Collect information from each worker
                for w, worker in enumerate(self.workers):
                    # Get results after executing the actions
                    next_obs, reward, done, info = worker.child.recv()

                    # Add transition to replay buffer
                    self.replay_buffer.add(self.obs[w], actions[w], reward, next_obs, done)

                    # update episode information
                    # collect episode info, which is available if an episode finished;
                    #  this includes total reward and length of the episode -
                    #  look at `Game` to see how it works.
                    if info:
                        tracker.add('reward', info['reward'])
                        tracker.add('length', info['length'])

                    # update current observation
                    self.obs[w] = next_obs

    def train(self, beta: float):
        """
        ### Train the model
        """
        for _ in range(self.train_epochs):
            # Sample from priority replay buffer
            samples = self.replay_buffer.sample(self.mini_batch_size, beta)
            # Get the predicted Q-value
            q_value = self.model(obs_to_torch(samples['obs']))

            # Get the Q-values of the next state for [Double Q-learning](index.html).
            # Gradients shouldn't propagate for these
            with torch.no_grad():
                # Get $\color{cyan}Q(s';\color{cyan}{\theta_i})$
                double_q_value = self.model(obs_to_torch(samples['next_obs']))
                # Get $\color{orange}Q(s';\color{orange}{\theta_i^{-}})$
                target_q_value = self.target_model(obs_to_torch(samples['next_obs']))

            # Compute Temporal Difference (TD) errors, $\delta$, and the loss, $\mathcal{L}(\theta)$.
            td_errors, loss = self.loss_func(q_value,
                                             q_value.new_tensor(samples['action']),
                                             double_q_value, target_q_value,
                                             q_value.new_tensor(samples['done']),
                                             q_value.new_tensor(samples['reward']),
                                             q_value.new_tensor(samples['weights']))

            # Calculate priorities for replay buffer $p_i = |\delta_i| + \epsilon$
            new_priorities = np.abs(td_errors.cpu().numpy()) + 1e-6
            # Update replay buffer priorities
            self.replay_buffer.update_priorities(samples['indexes'], new_priorities)

            # Zero out the previously calculated gradients
            self.optimizer.zero_grad()
            # Calculate gradients
            loss.backward()
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            # Update parameters based on gradients
            self.optimizer.step()

    def run_training_loop(self):
        """
        ### Run training loop
        """

        # Last 100 episode information
        tracker.set_queue('reward', 100, True)
        tracker.set_queue('length', 100, True)

        # Copy to target network initially
        self.target_model.load_state_dict(self.model.state_dict())

        for update in monit.loop(self.updates):
            # $\epsilon$, exploration fraction
            exploration = self.exploration_coefficient(update)
            tracker.add('exploration', exploration)
            # $\beta$ for prioritized replay
            beta = self.prioritized_replay_beta(update)
            tracker.add('beta', beta)

            # Sample with current policy
            self.sample(exploration)

            # Start training after the buffer is full
            if self.replay_buffer.is_full():
                # Train the model
                self.train(beta)

                # Periodically update target network
                if update % self.update_target_model == 0:
                    self.target_model.load_state_dict(self.model.state_dict())

            # Save tracked indicators.
            tracker.save()
            # Add a new line to the screen periodically
            if (update + 1) % 1_000 == 0:
                logger.log()

    def destroy(self):
        """
        ### Destroy
        Stop the workers
        """
        for worker in self.workers:
            worker.child.send(("close", None))


def main():
    # Create the experiment
    experiment.create(name='dqn')
    # Initialize the trainer
    m = Trainer()
    # Run and monitor the experiment
    with experiment.start():
        m.run_training_loop()
    # Stop the workers
    m.destroy()


# ## Run it
if __name__ == "__main__":
    main()
