"""
---
title: PPO Experiment with Atari Breakout
summary: Annotated implementation to train a PPO agent on Atari Breakout game.
---

# PPO Experiment with Atari Breakout

This experiment trains Proximal Policy Optimization (PPO) agent  Atari Breakout game on OpenAI Gym.
It runs the [game environments on multiple processes](../game.html) to sample efficiently.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lab-ml/nn/blob/master/labml_nn/rl/ppo/experiment.ipynb)
[![View Run](https://img.shields.io/badge/labml-experiment-brightgreen)](https://app.labml.ai/run/6eff28a0910e11eb9b008db315936e2f)
"""

from typing import Dict

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.distributions import Categorical

from labml import monit, tracker, logger, experiment
from labml.configs import FloatDynamicHyperParam, IntDynamicHyperParam
from labml_helpers.module import Module
from labml_nn.rl.game import Worker
from labml_nn.rl.ppo import ClippedPPOLoss, ClippedValueFunctionLoss
from labml_nn.rl.ppo.gae import GAE

# Select device
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")


class Model(Module):
    """
    ## Model
    """

    def __init__(self):
        super().__init__()

        # The first convolution layer takes a
        # 84x84 frame and produces a 20x20 frame
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4)

        # The second convolution layer takes a
        # 20x20 frame and produces a 9x9 frame
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)

        # The third convolution layer takes a
        # 9x9 frame and produces a 7x7 frame
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        # A fully connected layer takes the flattened
        # frame from third convolution layer, and outputs
        # 512 features
        self.lin = nn.Linear(in_features=7 * 7 * 64, out_features=512)

        # A fully connected layer to get logits for $\pi$
        self.pi_logits = nn.Linear(in_features=512, out_features=4)

        # A fully connected layer to get value function
        self.value = nn.Linear(in_features=512, out_features=1)

        #
        self.activation = nn.ReLU()

    def __call__(self, obs: torch.Tensor):
        h = self.activation(self.conv1(obs))
        h = self.activation(self.conv2(h))
        h = self.activation(self.conv3(h))
        h = h.reshape((-1, 7 * 7 * 64))

        h = self.activation(self.lin(h))

        pi = Categorical(logits=self.pi_logits(h))
        value = self.value(h).reshape(-1)

        return pi, value


def obs_to_torch(obs: np.ndarray) -> torch.Tensor:
    """Scale observations from `[0, 255]` to `[0, 1]`"""
    return torch.tensor(obs, dtype=torch.float32, device=device) / 255.


class Trainer:
    """
    ## Trainer
    """

    def __init__(self, *,
                 updates: int, epochs: IntDynamicHyperParam,
                 n_workers: int, worker_steps: int, batches: int,
                 value_loss_coef: FloatDynamicHyperParam,
                 entropy_bonus_coef: FloatDynamicHyperParam,
                 clip_range: FloatDynamicHyperParam,
                 learning_rate: FloatDynamicHyperParam,
                 ):
        # #### Configurations

        # number of updates
        self.updates = updates
        # number of epochs to train the model with sampled data
        self.epochs = epochs
        # number of worker processes
        self.n_workers = n_workers
        # number of steps to run on each process for a single update
        self.worker_steps = worker_steps
        # number of mini batches
        self.batches = batches
        # total number of samples for a single update
        self.batch_size = self.n_workers * self.worker_steps
        # size of a mini batch
        self.mini_batch_size = self.batch_size // self.batches
        assert (self.batch_size % self.batches == 0)

        # Value loss coefficient
        self.value_loss_coef = value_loss_coef
        # Entropy bonus coefficient
        self.entropy_bonus_coef = entropy_bonus_coef

        # Clipping range
        self.clip_range = clip_range
        # Learning rate
        self.learning_rate = learning_rate

        # #### Initialize

        # create workers
        self.workers = [Worker(47 + i) for i in range(self.n_workers)]

        # initialize tensors for observations
        self.obs = np.zeros((self.n_workers, 4, 84, 84), dtype=np.uint8)
        for worker in self.workers:
            worker.child.send(("reset", None))
        for i, worker in enumerate(self.workers):
            self.obs[i] = worker.child.recv()

        # model
        self.model = Model().to(device)

        # optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=2.5e-4)

        # GAE with $\gamma = 0.99$ and $\lambda = 0.95$
        self.gae = GAE(self.n_workers, self.worker_steps, 0.99, 0.95)

        # PPO Loss
        self.ppo_loss = ClippedPPOLoss()

        # Value Loss
        self.value_loss = ClippedValueFunctionLoss()

    def sample(self) -> Dict[str, torch.Tensor]:
        """
        ### Sample data with current policy
        """

        rewards = np.zeros((self.n_workers, self.worker_steps), dtype=np.float32)
        actions = np.zeros((self.n_workers, self.worker_steps), dtype=np.int32)
        done = np.zeros((self.n_workers, self.worker_steps), dtype=np.bool)
        obs = np.zeros((self.n_workers, self.worker_steps, 4, 84, 84), dtype=np.uint8)
        log_pis = np.zeros((self.n_workers, self.worker_steps), dtype=np.float32)
        values = np.zeros((self.n_workers, self.worker_steps + 1), dtype=np.float32)

        with torch.no_grad():
            # sample `worker_steps` from each worker
            for t in range(self.worker_steps):
                # `self.obs` keeps track of the last observation from each worker,
                #  which is the input for the model to sample the next action
                obs[:, t] = self.obs
                # sample actions from $\pi_{\theta_{OLD}}$ for each worker;
                #  this returns arrays of size `n_workers`
                pi, v = self.model(obs_to_torch(self.obs))
                values[:, t] = v.cpu().numpy()
                a = pi.sample()
                actions[:, t] = a.cpu().numpy()
                log_pis[:, t] = pi.log_prob(a).cpu().numpy()

                # run sampled actions on each worker
                for w, worker in enumerate(self.workers):
                    worker.child.send(("step", actions[w, t]))

                for w, worker in enumerate(self.workers):
                    # get results after executing the actions
                    self.obs[w], rewards[w, t], done[w, t], info = worker.child.recv()

                    # collect episode info, which is available if an episode finished;
                    #  this includes total reward and length of the episode -
                    #  look at `Game` to see how it works.
                    if info:
                        tracker.add('reward', info['reward'])
                        tracker.add('length', info['length'])

            # Get value of after the final step
            _, v = self.model(obs_to_torch(self.obs))
            values[:, self.worker_steps] = v.cpu().numpy()

        # calculate advantages
        advantages = self.gae(done, rewards, values)

        #
        samples = {
            'obs': obs,
            'actions': actions,
            'values': values[:, :-1],
            'log_pis': log_pis,
            'advantages': advantages
        }

        # samples are currently in `[workers, time_step]` table,
        # we should flatten it for training
        samples_flat = {}
        for k, v in samples.items():
            v = v.reshape(v.shape[0] * v.shape[1], *v.shape[2:])
            if k == 'obs':
                samples_flat[k] = obs_to_torch(v)
            else:
                samples_flat[k] = torch.tensor(v, device=device)

        return samples_flat

    def train(self, samples: Dict[str, torch.Tensor]):
        """
        ### Train the model based on samples
        """

        # It learns faster with a higher number of epochs,
        #  but becomes a little unstable; that is,
        #  the average episode reward does not monotonically increase
        #  over time.
        # May be reducing the clipping range might solve it.
        for _ in range(self.epochs()):
            # shuffle for each epoch
            indexes = torch.randperm(self.batch_size)

            # for each mini batch
            for start in range(0, self.batch_size, self.mini_batch_size):
                # get mini batch
                end = start + self.mini_batch_size
                mini_batch_indexes = indexes[start: end]
                mini_batch = {}
                for k, v in samples.items():
                    mini_batch[k] = v[mini_batch_indexes]

                # train
                loss = self._calc_loss(mini_batch)

                # Set learning rate
                for pg in self.optimizer.param_groups:
                    pg['lr'] = self.learning_rate()
                # Zero out the previously calculated gradients
                self.optimizer.zero_grad()
                # Calculate gradients
                loss.backward()
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                # Update parameters based on gradients
                self.optimizer.step()

    @staticmethod
    def _normalize(adv: torch.Tensor):
        """#### Normalize advantage function"""
        return (adv - adv.mean()) / (adv.std() + 1e-8)

    def _calc_loss(self, samples: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        ### Calculate total loss
        """

        # $R_t$ returns sampled from $\pi_{\theta_{OLD}}$
        sampled_return = samples['values'] + samples['advantages']

        # $\bar{A_t} = \frac{\hat{A_t} - \mu(\hat{A_t})}{\sigma(\hat{A_t})}$,
        # where $\hat{A_t}$ is advantages sampled from $\pi_{\theta_{OLD}}$.
        # Refer to sampling function in [Main class](#main) below
        #  for the calculation of $\hat{A}_t$.
        sampled_normalized_advantage = self._normalize(samples['advantages'])

        # Sampled observations are fed into the model to get $\pi_\theta(a_t|s_t)$ and $V^{\pi_\theta}(s_t)$;
        #  we are treating observations as state
        pi, value = self.model(samples['obs'])

        # $-\log \pi_\theta (a_t|s_t)$, $a_t$ are actions sampled from $\pi_{\theta_{OLD}}$
        log_pi = pi.log_prob(samples['actions'])

        # Calculate policy loss
        policy_loss = self.ppo_loss(log_pi, samples['log_pis'], sampled_normalized_advantage, self.clip_range())

        # Calculate Entropy Bonus
        #
        # $\mathcal{L}^{EB}(\theta) =
        #  \mathbb{E}\Bigl[ S\bigl[\pi_\theta\bigr] (s_t) \Bigr]$
        entropy_bonus = pi.entropy()
        entropy_bonus = entropy_bonus.mean()

        # Calculate value function loss
        value_loss = self.value_loss(value, samples['values'], sampled_return, self.clip_range())

        # $\mathcal{L}^{CLIP+VF+EB} (\theta) =
        #  \mathcal{L}^{CLIP} (\theta) +
        #  c_1 \mathcal{L}^{VF} (\theta) - c_2 \mathcal{L}^{EB}(\theta)$
        loss = (policy_loss
                + self.value_loss_coef() * value_loss
                - self.entropy_bonus_coef() * entropy_bonus)

        # for monitoring
        approx_kl_divergence = .5 * ((samples['log_pis'] - log_pi) ** 2).mean()

        # Add to tracker
        tracker.add({'policy_reward': -policy_loss,
                     'value_loss': value_loss,
                     'entropy_bonus': entropy_bonus,
                     'kl_div': approx_kl_divergence,
                     'clip_fraction': self.ppo_loss.clip_fraction})

        return loss

    def run_training_loop(self):
        """
        ### Run training loop
        """

        # last 100 episode information
        tracker.set_queue('reward', 100, True)
        tracker.set_queue('length', 100, True)

        for update in monit.loop(self.updates):
            # sample with current policy
            samples = self.sample()

            # train the model
            self.train(samples)

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
    experiment.create(name='ppo')
    # Configurations
    configs = {
        # Number of updates
        'updates': 10000,
        # ⚙️ Number of epochs to train the model with sampled data.
        # You can change this while the experiment is running.
        # [![Example](https://img.shields.io/badge/example-hyperparams-brightgreen)](https://app.labml.ai/run/6eff28a0910e11eb9b008db315936e2f/hyper_params)
        'epochs': IntDynamicHyperParam(8),
        # Number of worker processes
        'n_workers': 8,
        # Number of steps to run on each process for a single update
        'worker_steps': 128,
        # Number of mini batches
        'batches': 4,
        # ⚙️ Value loss coefficient.
        # You can change this while the experiment is running.
        # [![Example](https://img.shields.io/badge/example-hyperparams-brightgreen)](https://app.labml.ai/run/6eff28a0910e11eb9b008db315936e2f/hyper_params)
        'value_loss_coef': FloatDynamicHyperParam(0.5),
        # ⚙️ Entropy bonus coefficient.
        # You can change this while the experiment is running.
        # [![Example](https://img.shields.io/badge/example-hyperparams-brightgreen)](https://app.labml.ai/run/6eff28a0910e11eb9b008db315936e2f/hyper_params)
        'entropy_bonus_coef': FloatDynamicHyperParam(0.01),
        # ⚙️ Clip range.
        'clip_range': FloatDynamicHyperParam(0.1),
        # You can change this while the experiment is running.
        # [![Example](https://img.shields.io/badge/example-hyperparams-brightgreen)](https://app.labml.ai/run/6eff28a0910e11eb9b008db315936e2f/hyper_params)
        # ⚙️ Learning rate.
        'learning_rate': FloatDynamicHyperParam(1e-3, (0, 1e-3)),
    }

    experiment.configs(configs)

    # Initialize the trainer
    m = Trainer(
        updates=configs['updates'],
        epochs=configs['epochs'],
        n_workers=configs['n_workers'],
        worker_steps=configs['worker_steps'],
        batches=configs['batches'],
        value_loss_coef=configs['value_loss_coef'],
        entropy_bonus_coef=configs['entropy_bonus_coef'],
        clip_range=configs['clip_range'],
        learning_rate=configs['learning_rate'],
    )

    # Run and monitor the experiment
    with experiment.start():
        m.run_training_loop()
    # Stop the workers
    m.destroy()


# ## Run it
if __name__ == "__main__":
    main()
