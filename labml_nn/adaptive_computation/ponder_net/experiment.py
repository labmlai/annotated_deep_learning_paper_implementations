"""
---
title: "PonderNet Parity Task Experiment"
summary: >
  This trains is a PonderNet on Parity Task
---

# [PonderNet](index.html) [Parity Task](../parity.html) Experiment

This trains a [PonderNet](index.html) on [Parity Task](../parity.html).
"""

from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from labml import tracker, experiment
from labml_helpers.metrics.accuracy import AccuracyDirect
from labml_helpers.train_valid import SimpleTrainValidConfigs, BatchIndex
from labml_nn.adaptive_computation.parity import ParityDataset
from labml_nn.adaptive_computation.ponder_net import ParityPonderGRU, ReconstructionLoss, RegularizationLoss


class Configs(SimpleTrainValidConfigs):
    """
    Configurations with a
     [simple training loop](https://docs.labml.ai/api/helpers.html#labml_helpers.train_valid.SimpleTrainValidConfigs)
    """

    # Number of epochs
    epochs: int = 100
    # Number of batches per epoch
    n_batches: int = 500
    # Batch size
    batch_size: int = 128

    # Model
    model: ParityPonderGRU

    # $L_{Rec}$
    loss_rec: ReconstructionLoss
    # $L_{Reg}$
    loss_reg: RegularizationLoss

    # The number of elements in the input vector.
    # *We keep it low for demonstration; otherwise, training takes a lot of time.
    # Although the parity task seems simple, figuring out the pattern by looking at samples
    # is quite hard.*
    n_elems: int = 8
    # Number of units in the hidden layer (state)
    n_hidden: int = 64
    # Maximum number of steps $N$
    max_steps: int = 20

    # $\lambda_p$ for the geometric distribution $p_G(\lambda_p)$
    lambda_p: float = 0.2
    # Regularization loss $L_{Reg}$ coefficient $\beta$
    beta: float = 0.01

    # Gradient clipping by norm
    grad_norm_clip: float = 1.0

    # Training and validation loaders
    train_loader: DataLoader
    valid_loader: DataLoader

    # Accuracy calculator
    accuracy = AccuracyDirect()

    def init(self):
        # Print indicators to screen
        tracker.set_scalar('loss.*', True)
        tracker.set_scalar('loss_reg.*', True)
        tracker.set_scalar('accuracy.*', True)
        tracker.set_scalar('steps.*', True)

        # We need to set the metrics to calculate them for the epoch for training and validation
        self.state_modules = [self.accuracy]

        # Initialize the model
        self.model = ParityPonderGRU(self.n_elems, self.n_hidden, self.max_steps).to(self.device)
        # $L_{Rec}$
        self.loss_rec = ReconstructionLoss(nn.BCEWithLogitsLoss(reduction='none')).to(self.device)
        # $L_{Reg}$
        self.loss_reg = RegularizationLoss(self.lambda_p, self.max_steps).to(self.device)

        # Training and validation loaders
        self.train_loader = DataLoader(ParityDataset(self.batch_size * self.n_batches, self.n_elems),
                                       batch_size=self.batch_size)
        self.valid_loader = DataLoader(ParityDataset(self.batch_size * 32, self.n_elems),
                                       batch_size=self.batch_size)

    def step(self, batch: Any, batch_idx: BatchIndex):
        """
        This method gets called by the trainer for each batch
        """
        # Set the model mode
        self.model.train(self.mode.is_train)

        # Get the input and labels and move them to the model's device
        data, target = batch[0].to(self.device), batch[1].to(self.device)

        # Increment step in training mode
        if self.mode.is_train:
            tracker.add_global_step(len(data))

        # Run the model
        p, y_hat, p_sampled, y_hat_sampled = self.model(data)

        # Calculate the reconstruction loss
        loss_rec = self.loss_rec(p, y_hat, target.to(torch.float))
        tracker.add("loss.", loss_rec)

        # Calculate the regularization loss
        loss_reg = self.loss_reg(p)
        tracker.add("loss_reg.", loss_reg)

        # $L = L_{Rec} + \beta L_{Reg}$
        loss = loss_rec + self.beta * loss_reg

        # Calculate the expected number of steps taken
        steps = torch.arange(1, p.shape[0] + 1, device=p.device)
        expected_steps = (p * steps[:, None]).sum(dim=0)
        tracker.add("steps.", expected_steps)

        # Call accuracy metric
        self.accuracy(y_hat_sampled > 0, target)

        if self.mode.is_train:
            # Compute gradients
            loss.backward()
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_norm_clip)
            # Optimizer
            self.optimizer.step()
            # Clear gradients
            self.optimizer.zero_grad()
            #
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

#
if __name__ == '__main__':
    main()
