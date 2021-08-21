"""
---
title: "Evidential Deep Learning to Quantify Classification Uncertainty Experiment"
summary: >
  This trains is EDL model on MNIST
---

# [Evidential Deep Learning to Quantify Classification Uncertainty](index.html) Experiment

This trains a model based on [Evidential Deep Learning to Quantify Classification Uncertainty](index.html)
 on MNIST dataset.
"""

from typing import Any

import torch.nn as nn
import torch.utils.data

from labml import tracker, experiment
from labml.configs import option, calculate
from labml_helpers.module import Module
from labml_helpers.schedule import Schedule, RelativePiecewise
from labml_helpers.train_valid import BatchIndex
from labml_nn.experiments.mnist import MNISTConfigs
from labml_nn.uncertainty.evidence import KLDivergenceLoss, TrackStatistics, MaximumLikelihoodLoss, \
    CrossEntropyBayesRisk, SquaredErrorBayesRisk


class Model(Module):
    """
    ## LeNet based model fro MNIST classification
    """

    def __init__(self, dropout: float):
        super().__init__()
        # First $5x5$ convolution layer
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5)
        # ReLU activation
        self.act1 = nn.ReLU()
        # $2x2$ max-pooling
        self.max_pool1 = nn.MaxPool2d(2, 2)
        # Second $5x5$ convolution layer
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5)
        # ReLU activation
        self.act2 = nn.ReLU()
        # $2x2$ max-pooling
        self.max_pool2 = nn.MaxPool2d(2, 2)
        # First fully-connected layer that maps to $500$ features
        self.fc1 = nn.Linear(50 * 4 * 4, 500)
        # ReLU activation
        self.act3 = nn.ReLU()
        # Final fully connected layer to output evidence for $10$ classes.
        # The ReLU or Softplus activation is applied to this outside the model to get the
        # non-negative evidence
        self.fc2 = nn.Linear(500, 10)
        # Dropout for the hidden layer
        self.dropout = nn.Dropout(p=dropout)

    def __call__(self, x: torch.Tensor):
        """
        * `x` is the batch of MNIST images of shape `[batch_size, 1, 28, 28]`
        """
        # Apply first convolution and max pooling.
        # The result has shape `[batch_size, 20, 12, 12]`
        x = self.max_pool1(self.act1(self.conv1(x)))
        # Apply second convolution and max pooling.
        # The result has shape `[batch_size, 50, 4, 4]`
        x = self.max_pool2(self.act2(self.conv2(x)))
        # Flatten the tensor to shape `[batch_size, 50 * 4 * 4]`
        x = x.view(x.shape[0], -1)
        # Apply hidden layer
        x = self.act3(self.fc1(x))
        # Apply dropout
        x = self.dropout(x)
        # Apply final layer and return
        return self.fc2(x)


class Configs(MNISTConfigs):
    """
    ## Configurations

    We use [`MNISTConfigs`](../../experiments/mnist.html#MNISTConfigs) configurations.
    """

    # [KL Divergence regularization](index.html#KLDivergenceLoss)
    kl_div_loss = KLDivergenceLoss()
    # KL Divergence regularization coefficient schedule
    kl_div_coef: Schedule
    # KL Divergence regularization coefficient schedule
    kl_div_coef_schedule = [(0, 0.), (0.2, 0.01), (1, 1.)]
    # [Stats module](index.html#TrackStatistics) for tracking
    stats = TrackStatistics()
    # Dropout
    dropout: float = 0.5
    # Module to convert the model output to non-zero evidences
    outputs_to_evidence: Module

    def init(self):
        """
        ### Initialization
        """
        # Set tracker configurations
        tracker.set_scalar("loss.*", True)
        tracker.set_scalar("accuracy.*", True)
        tracker.set_histogram('u.*', True)
        tracker.set_histogram('prob.*', False)
        tracker.set_scalar('annealing_coef.*', False)
        tracker.set_scalar('kl_div_loss.*', False)

        #
        self.state_modules = []

    def step(self, batch: Any, batch_idx: BatchIndex):
        """
        ### Training or validation step
        """

        # Training/Evaluation mode
        self.model.train(self.mode.is_train)

        # Move data to the device
        data, target = batch[0].to(self.device), batch[1].to(self.device)

        # One-hot coded targets
        eye = torch.eye(10).to(torch.float).to(self.device)
        target = eye[target]

        # Update global step (number of samples processed) when in training mode
        if self.mode.is_train:
            tracker.add_global_step(len(data))

        # Get model outputs
        outputs = self.model(data)
        # Get evidences $e_k \ge 0$
        evidence = self.outputs_to_evidence(outputs)

        # Calculate loss
        loss = self.loss_func(evidence, target)
        # Calculate KL Divergence regularization loss
        kl_div_loss = self.kl_div_loss(evidence, target)
        tracker.add("loss.", loss)
        tracker.add("kl_div_loss.", kl_div_loss)

        # KL Divergence loss coefficient $\lambda_t$
        annealing_coef = min(1., self.kl_div_coef(tracker.get_global_step()))
        tracker.add("annealing_coef.", annealing_coef)

        # Total loss
        loss = loss + annealing_coef * kl_div_loss

        # Track statistics
        self.stats(evidence, target)

        # Train the model
        if self.mode.is_train:
            # Calculate gradients
            loss.backward()
            # Take optimizer step
            self.optimizer.step()
            # Clear the gradients
            self.optimizer.zero_grad()

        # Save the tracked metrics
        tracker.save()


@option(Configs.model)
def mnist_model(c: Configs):
    """
    ### Create model
    """
    return Model(c.dropout).to(c.device)


@option(Configs.kl_div_coef)
def kl_div_coef(c: Configs):
    """
    ### KL Divergence Loss Coefficient Schedule
    """

    # Create a [relative piecewise schedule](https://docs.labml.ai/api/helpers.html#labml_helpers.schedule.Piecewise)
    return RelativePiecewise(c.kl_div_coef_schedule, c.epochs * len(c.train_dataset))


# [Maximum Likelihood Loss](index.html#MaximumLikelihoodLoss)
calculate(Configs.loss_func, 'max_likelihood_loss', lambda: MaximumLikelihoodLoss())
# [Cross Entropy Bayes Risk](index.html#CrossEntropyBayesRisk)
calculate(Configs.loss_func, 'cross_entropy_bayes_risk', lambda: CrossEntropyBayesRisk())
# [Squared Error Bayes Risk](index.html#SquaredErrorBayesRisk)
calculate(Configs.loss_func, 'squared_error_bayes_risk', lambda: SquaredErrorBayesRisk())

# ReLU to calculate evidence
calculate(Configs.outputs_to_evidence, 'relu', lambda: nn.ReLU())
# Softplus to calculate evidence
calculate(Configs.outputs_to_evidence, 'softplus', lambda: nn.Softplus())


def main():
    # Create experiment
    experiment.create(name='evidence_mnist')
    # Create configurations
    conf = Configs()
    # Load configurations
    experiment.configs(conf, {
        'optimizer.optimizer': 'Adam',
        'optimizer.learning_rate': 0.001,
        'optimizer.weight_decay': 0.005,

        # 'loss_func': 'max_likelihood_loss',
        # 'loss_func': 'cross_entropy_bayes_risk',
        'loss_func': 'squared_error_bayes_risk',

        'outputs_to_evidence': 'softplus',

        'dropout': 0.5,
    })
    # Start the experiment and run the training loop
    with experiment.start():
        conf.run()


#
if __name__ == '__main__':
    main()
