"""
\(
   \def\hl1#1{{\color{orange}{#1}}}
   \def\blue#1{{\color{cyan}{#1}}}
   \def\green#1{{\color{yellowgreen}{#1}}}
\)
"""

import numpy as np
import torch

from labml import tracker, experiment, logger, monit
from labml_helpers.schedule import Piecewise
from labml_nn.rl.dqn import QFuncLoss
from labml_nn.rl.dqn.model import Model
from labml_nn.rl.dqn.replay_buffer import ReplayBuffer
from labml_nn.rl.game import Worker

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")


def obs_to_torch(obs: np.ndarray) -> torch.Tensor:
    """Scale observations from `[0, 255]` to `[0, 1]`"""
    return torch.tensor(obs, dtype=torch.float32, device=device) / 255.


class Trainer:
    """
    ## <a name="main"></a>Main class
    This class runs the training loop.
    It initializes TensorFlow, handles logging and monitoring,
     and runs workers as multiple processes.
    """

    def __init__(self):
        """
        ### Initialize
        """

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

        # exploration as a function of time step
        self.exploration_coefficient = Piecewise(
            [
                (0, 1.0),
                (25_000, 0.1),
                (self.updates / 2, 0.01)
            ], outside_value=0.01)

        # update target network every 250 update
        self.update_target_model = 250

        # $\beta$ for replay buffer as a function of time steps
        self.prioritized_replay_beta = Piecewise(
            [
                (0, 0.4),
                (self.updates, 1)
            ], outside_value=1)

        # replay buffer
        self.replay_buffer = ReplayBuffer(2 ** 14, 0.6)

        self.model = Model().to(device)
        self.target_model = Model().to(device)

        # last observation for each worker
        # create workers
        self.workers = [Worker(47 + i) for i in range(self.n_workers)]

        # initialize tensors for observations
        self.obs = np.zeros((self.n_workers, 4, 84, 84), dtype=np.uint8)
        for worker in self.workers:
            worker.child.send(("reset", None))
        for i, worker in enumerate(self.workers):
            self.obs[i] = worker.child.recv()

        self.loss_func = QFuncLoss(0.99)
        # optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=2.5e-4)

    def _sample_action(self, q_value: torch.Tensor, exploration_coefficient: float):
        """
        #### $\epsilon$-greedy Sampling
        When sampling actions we use a $\epsilon$-greedy strategy, where we
        take a greedy action with probabiliy $1 - \epsilon$ and
        take a random action with probability $\epsilon$.
        We refer to $\epsilon$ as *exploration*.
        """

        with torch.no_grad():
            greedy_action = torch.argmax(q_value, dim=-1)
            random_action = torch.randint(q_value.shape[-1], greedy_action.shape, device=q_value.device)

            is_choose_rand = torch.rand(greedy_action.shape, device=q_value.device) < exploration_coefficient

            return torch.where(is_choose_rand, random_action, greedy_action).cpu().numpy()

    def sample(self, exploration_coefficient: float):
        """### Sample data"""

        with torch.no_grad():
            # sample `SAMPLE_STEPS`
            for t in range(self.worker_steps):
                # sample actions
                q_value = self.model(obs_to_torch(self.obs))
                actions = self._sample_action(q_value, exploration_coefficient)

                # run sampled actions on each worker
                for w, worker in enumerate(self.workers):
                    worker.child.send(("step", actions[w]))

                # collect information from each worker
                for w, worker in enumerate(self.workers):
                    # get results after executing the actions
                    next_obs, reward, done, info = worker.child.recv()

                    # add transition to replay buffer
                    self.replay_buffer.add(self.obs[w], actions[w], reward, next_obs, done)

                    # update episode information
                    # collect episode info, which is available if an episode finished;
                    #  this includes total reward and length of the episode -
                    #  look at `Game` to see how it works.
                    # We also add a game frame to it for monitoring.
                    if info:
                        tracker.add('reward', info['reward'])
                        tracker.add('length', info['length'])

                    # update current observation
                    self.obs[w] = next_obs

    def train(self, beta: float):
        for _ in range(self.train_epochs):
            # sample from priority replay buffer
            samples = self.replay_buffer.sample(self.mini_batch_size, beta)
            # train network
            q_value = self.model(obs_to_torch(samples['obs']))

            with torch.no_grad():
                double_q_value = self.model(obs_to_torch(samples['next_obs']))
                target_q_value = self.target_model(obs_to_torch(samples['next_obs']))

            td_errors, loss = self.loss_func(q_value,
                                             q_value.new_tensor(samples['action']),
                                             double_q_value, target_q_value,
                                             q_value.new_tensor(samples['done']),
                                             q_value.new_tensor(samples['reward']),
                                             q_value.new_tensor(samples['weights']))

            # $p_i = |\delta_i| + \epsilon$
            new_priorities = np.abs(td_errors.cpu().numpy()) + 1e-6
            # update replay buffer
            self.replay_buffer.update_priorities(samples['indexes'], new_priorities)

            # compute gradients
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            self.optimizer.step()

    def run_training_loop(self):
        """
        ### Run training loop
        """

        # copy to target network initially
        self.target_model.load_state_dict(self.model.state_dict())

        # last 100 episode information
        tracker.set_queue('reward', 100, True)
        tracker.set_queue('length', 100, True)

        for update in monit.loop(self.updates):
            # $\epsilon$, exploration fraction
            exploration = self.exploration_coefficient(update)
            tracker.add('exploration', exploration)
            # $\beta$ for priority replay
            beta = self.prioritized_replay_beta(update)
            tracker.add('beta', beta)

            # sample with current policy
            self.sample(exploration)

            if self.replay_buffer.is_full():
                # train the model
                self.train(beta)

                # periodically update target network
                if update % self.update_target_model == 0:
                    self.target_model.load_state_dict(self.model.state_dict())

            tracker.save()
            if (update + 1) % 1_000 == 0:
                logger.log()

    def destroy(self):
        """
        ### Destroy
        Stop the workers
        """
        for worker in self.workers:
            worker.child.send(("close", None))


# ## Run it
if __name__ == "__main__":
    experiment.create(name='dqn')
    m = Trainer()
    with experiment.start():
        m.run_training_loop()
    m.destroy()
