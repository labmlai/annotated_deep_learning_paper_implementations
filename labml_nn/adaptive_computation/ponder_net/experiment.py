from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from labml import tracker, experiment
from labml_helpers.metrics.accuracy import AccuracyDirect
from labml_helpers.train_valid import SimpleTrainValidConfigs, BatchIndex
from labml_nn.adaptive_computation.parity import ParityDataset
from labml_nn.adaptive_computation.ponder_net import SimplePonderGRU, ReconstructionLoss, RegularizationLoss


class Configs(SimpleTrainValidConfigs):
    """
    Configurations with MNIST data and Train & Validation setup
    """
    epochs: int = 100
    model: SimplePonderGRU
    accuracy = AccuracyDirect()

    loss_rec: ReconstructionLoss
    loss_reg: RegularizationLoss

    n_elems: int = 8
    n_hidden: int = 64
    max_steps: int = 20

    lambda_p: float = 0.2
    beta: float = 0.01

    batch_size: int = 128
    n_batches: int = 500

    grad_norm_clip: float = 1.0

    train_loader: DataLoader
    valid_loader: DataLoader

    def init(self):
        # Print losses and accuracy to screen
        tracker.set_scalar('loss.*', True)
        tracker.set_scalar('loss_reg.*', True)
        tracker.set_scalar('accuracy.*', True)
        tracker.set_scalar('steps.*', True)

        # We need to set the metrics to calculate them for the epoch for training and validation
        self.state_modules = [self.accuracy]

        self.model = SimplePonderGRU(self.n_elems, self.n_hidden, self.max_steps).to(self.device)
        self.loss_rec = ReconstructionLoss(nn.BCEWithLogitsLoss(reduction='none')).to(self.device)
        self.loss_reg = RegularizationLoss(self.lambda_p, self.max_steps).to(self.device)
        self.train_loader = DataLoader(ParityDataset(self.batch_size * self.n_batches, self.n_elems),
                                       batch_size=self.batch_size)
        self.valid_loader = DataLoader(ParityDataset(self.batch_size * 32, self.n_elems),
                                       batch_size=self.batch_size)

    def step(self, batch: Any, batch_idx: BatchIndex):
        """
        This method gets called by the trainer
        """
        # Set the model mode
        self.model.train(self.mode.is_train)

        # Get the images and labels and move them to the model's device
        data, target = batch[0].to(self.device), batch[1].to(self.device)

        # Increment step in training mode
        if self.mode.is_train:
            tracker.add_global_step(len(data))

        # Run the model
        p, y_hat, p_sampled, y_hat_sampled = self.model(data)

        # Calculate the total loss
        loss = self.loss_rec(p, y_hat, target.to(torch.float))
        tracker.add("loss.", loss)

        loss_reg = self.loss_reg(p)
        loss = loss + self.beta * loss_reg
        tracker.add("loss_reg.", loss_reg)

        steps = torch.arange(1, p.shape[0] + 1, device=p.device)
        mean_steps = (p * steps[:, None]).sum(dim=0)

        tracker.add("steps.", mean_steps)

        # Call accuracy metric
        self.accuracy(y_hat_sampled[-1] > 0, target)

        if self.mode.is_train:
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_norm_clip)

            self.optimizer.step()
            # Log parameters and gradients
            self.optimizer.zero_grad()

            tracker.save()


def main():
    """
    Run the experiment
    """
    experiment.create(name='ponder_net')

    conf = Configs()
    experiment.configs(conf, {
        'optimizer.optimizer': 'Adam',
        'optimizer.learning_rate': 0.0003,
    })

    with experiment.start():
        conf.run()


if __name__ == '__main__':
    main()
