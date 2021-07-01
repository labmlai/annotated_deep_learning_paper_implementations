"""
---
title: MNIST Experiment
summary: >
  This is a reusable trainer for MNIST dataset
---

# MNIST Experiment
"""

import torch.nn as nn
import torch.utils.data
from labml_helpers.module import Module

from labml import tracker
from labml.configs import option
from labml_helpers.datasets.mnist import MNISTConfigs as MNISTDatasetConfigs
from labml_helpers.device import DeviceConfigs
from labml_helpers.metrics.accuracy import Accuracy
from labml_helpers.train_valid import TrainValidConfigs, BatchIndex, hook_model_outputs
from labml_nn.optimizers.configs import OptimizerConfigs


class MNISTConfigs(MNISTDatasetConfigs, TrainValidConfigs):
    """
    <a id="MNISTConfigs">
    ## Trainer configurations
    </a>
    """

    # Optimizer
    optimizer: torch.optim.Adam
    # Training device
    device: torch.device = DeviceConfigs()

    # Classification model
    model: Module
    # Number of epochs to train for
    epochs: int = 10

    # Number of times to switch between training and validation within an epoch
    inner_iterations = 10

    # Accuracy function
    accuracy = Accuracy()
    # Loss function
    loss_func = nn.CrossEntropyLoss()

    def init(self):
        """
        ### Initialization
        """
        # Set tracker configurations
        tracker.set_scalar("loss.*", True)
        tracker.set_scalar("accuracy.*", True)
        # Add a hook to log module outputs
        hook_model_outputs(self.mode, self.model, 'model')
        # Add accuracy as a state module.
        # The name is probably confusing, since it's meant to store
        # states between training and validation for RNNs.
        # This will keep the accuracy metric stats separate for training and validation.
        self.state_modules = [self.accuracy]

    def step(self, batch: any, batch_idx: BatchIndex):
        """
        ### Training or validation step
        """

        # Training/Evaluation mode
        self.model.train(self.mode.is_train)

        # Move data to the device
        data, target = batch[0].to(self.device), batch[1].to(self.device)

        # Update global step (number of samples processed) when in training mode
        if self.mode.is_train:
            tracker.add_global_step(len(data))

        # Whether to capture model outputs
        with self.mode.update(is_log_activations=batch_idx.is_last):
            # Get model outputs.
            output = self.model(data)

        # Calculate and log loss
        loss = self.loss_func(output, target)
        tracker.add("loss.", loss)

        # Calculate and log accuracy
        self.accuracy(output, target)
        self.accuracy.track()

        # Train the model
        if self.mode.is_train:
            # Calculate gradients
            loss.backward()
            # Take optimizer step
            self.optimizer.step()
            # Log the model parameters and gradients on last batch of every epoch
            if batch_idx.is_last:
                tracker.add('model', self.model)
            # Clear the gradients
            self.optimizer.zero_grad()

        # Save the tracked metrics
        tracker.save()


@option(MNISTConfigs.optimizer)
def _optimizer(c: MNISTConfigs):
    """
    ### Default optimizer configurations
    """
    opt_conf = OptimizerConfigs()
    opt_conf.parameters = c.model.parameters()
    opt_conf.optimizer = 'Adam'
    return opt_conf
