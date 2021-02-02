"""
---
title: MNIST example to test the optimizers
summary: This is a simple MNIST example with a CNN model to test the optimizers.
---

# MNIST example to test the optimizers
"""
import torch.nn as nn
import torch.utils.data
from labml_helpers.module import Module

from labml import experiment, tracker
from labml.configs import option
from labml_helpers.datasets.mnist import MNISTConfigs
from labml_helpers.device import DeviceConfigs
from labml_helpers.metrics.accuracy import Accuracy
from labml_helpers.seed import SeedConfigs
from labml_helpers.train_valid import TrainValidConfigs, BatchIndex, hook_model_outputs
from labml_nn.optimizers.configs import OptimizerConfigs


class Model(Module):
    """
    ## The model
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(16 * 50, 500)
        self.fc2 = nn.Linear(500, 10)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.pool1(x)
        x = self.activation(self.conv2(x))
        x = self.pool2(x)
        x = self.activation(self.fc1(x.view(-1, 16 * 50)))
        return self.fc2(x)


class Configs(MNISTConfigs, TrainValidConfigs):
    """
    ## Configurable Experiment Definition
    """
    optimizer: torch.optim.Adam
    model: nn.Module
    set_seed = SeedConfigs()
    device: torch.device = DeviceConfigs()
    epochs: int = 10

    is_save_models = True
    model: nn.Module
    inner_iterations = 10

    accuracy_func = Accuracy()
    loss_func = nn.CrossEntropyLoss()

    def init(self):
        tracker.set_queue("loss.*", 20, True)
        tracker.set_scalar("accuracy.*", True)
        hook_model_outputs(self.mode, self.model, 'model')
        self.state_modules = [self.accuracy_func]

    def step(self, batch: any, batch_idx: BatchIndex):
        # Get the batch
        data, target = batch[0].to(self.device), batch[1].to(self.device)

        # Add global step if we are in training mode
        if self.mode.is_train:
            tracker.add_global_step(len(data))

        # Run the model and specify whether to log the activations
        with self.mode.update(is_log_activations=batch_idx.is_last):
            output = self.model(data)

        # Calculate the loss
        loss = self.loss_func(output, target)
        # Calculate the accuracy
        self.accuracy_func(output, target)
        # Log the loss
        tracker.add("loss.", loss)

        # Optimize if we are in training mode
        if self.mode.is_train:
            # Calculate the gradients
            loss.backward()

            # Take optimizer step
            self.optimizer.step()
            # Log the parameter and gradient L2 norms once per epoch
            if batch_idx.is_last:
                tracker.add('model', self.model)
                tracker.add('optimizer', (self.optimizer, {'model': self.model}))
            # Clear the gradients
            self.optimizer.zero_grad()

        # Save logs
        tracker.save()


@option(Configs.model)
def model(c: Configs):
    return Model().to(c.device)


@option(Configs.optimizer)
def _optimizer(c: Configs):
    """
    Create a configurable optimizer.
    We can change the optimizer type and hyper-parameters using configurations.
    """
    opt_conf = OptimizerConfigs()
    opt_conf.parameters = c.model.parameters()
    return opt_conf


def main():
    conf = Configs()
    conf.inner_iterations = 10
    experiment.create(name='mnist_ada_belief')
    experiment.configs(conf, {'inner_iterations': 10,
                              # Specify the optimizer
                              'optimizer.optimizer': 'Adam',
                              'optimizer.learning_rate': 1.5e-4})
    conf.set_seed.set()
    experiment.add_pytorch_models(dict(model=conf.model))
    with experiment.start():
        conf.run()


if __name__ == '__main__':
    main()
