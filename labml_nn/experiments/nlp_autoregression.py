"""
---
title: NLP auto-regression trainer
summary: >
  This is a reusable trainer for auto-regressive tasks
---

# Auto-regressive NLP model trainer
"""

from typing import Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from labml import lab, monit, logger, tracker
from labml.configs import option
from labml.logger import Text
from labml_helpers.datasets.text import TextDataset, SequentialDataLoader, SequentialUnBatchedDataset, TextFileDataset
from labml_helpers.device import DeviceConfigs
from labml_helpers.metrics.accuracy import Accuracy
from labml_helpers.module import Module
from labml_helpers.train_valid import TrainValidConfigs, hook_model_outputs, BatchIndex
from labml_nn.optimizers.configs import OptimizerConfigs


class CrossEntropyLoss(Module):
    """
    ### Cross entropy loss
    """

    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

    def __call__(self, outputs, targets):
        return self.loss(outputs.view(-1, outputs.shape[-1]), targets.view(-1))


class NLPAutoRegressionConfigs(TrainValidConfigs):
    """
    <a id="NLPAutoRegressionConfigs">
    ## Trainer configurations
    </a>

    This has the basic configurations for NLP auto-regressive task training.
    All the properties are configurable.
    """

    # Optimizer
    optimizer: torch.optim.Adam
    # Training device
    device: torch.device = DeviceConfigs()

    # Autoregressive model
    model: Module
    # Text dataset
    text: TextDataset
    # Batch size
    batch_size: int = 16
    # Length of the sequence, or context size
    seq_len: int = 512
    # Number of token in vocabulary
    n_tokens: int
    # Tokenizer
    tokenizer: Callable = 'character'

    # Text prompt to start sampling (for illustration)
    prompt: str
    # The token separator when sampling (blank for character level tokenization)
    prompt_separator: str

    # Whether to periodically save models
    is_save_models = True

    # Loss function
    loss_func = CrossEntropyLoss()
    # Accuracy function
    accuracy = Accuracy()
    # Model embedding size
    d_model: int = 512
    # Gradient clipping
    grad_norm_clip: float = 1.0

    # Training data loader
    train_loader: DataLoader = 'shuffled_train_loader'
    # Validation data loader
    valid_loader: DataLoader = 'shuffled_valid_loader'

    def init(self):
        """
        ### Initialization
        """
        # Set tracker configurations
        tracker.set_scalar("accuracy.*", True)
        tracker.set_scalar("loss.*", True)
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

        # Move data to the device
        data, target = batch[0].to(self.device), batch[1].to(self.device)

        # Update global step (number of tokens processed) when in training mode
        if self.mode.is_train:
            tracker.add_global_step(data.shape[0] * data.shape[1])

        # Whether to capture model outputs
        with self.mode.update(is_log_activations=batch_idx.is_last):
            # Get model outputs.
            # It's returning a tuple for states when using RNNs.
            # This is not implemented yet. 😜
            output, *_ = self.model(data)

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
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_norm_clip)
            # Take optimizer step
            self.optimizer.step()
            # Log the model parameters and gradients on last batch of every epoch
            if batch_idx.is_last:
                tracker.add('model', self.model)
            # Clear the gradients
            self.optimizer.zero_grad()

        # Save the tracked metrics
        tracker.save()

    def sample(self):
        """
        ### Sampling function to generate samples periodically while training
        """

        # Starting prompt
        prompt = self.prompt
        # Collect output for printing
        log = [(prompt, Text.subtle)]
        # Sample 25 tokens
        for i in monit.iterate('Sample', 25):
            # Tokenize the prompt
            data = self.text.text_to_i(prompt).unsqueeze(-1)
            data = data.to(self.device)
            # Get the model output
            output, *_ = self.model(data)
            # Get the model prediction (greedy)
            output = output.argmax(dim=-1).squeeze()
            # Add the prediction to prompt
            prompt += self.prompt_separator + self.text.itos[output[-1]]
            # Add the prediction for logging
            log += [(self.prompt_separator + self.text.itos[output[-1]], Text.value)]

        # Print the sampled output
        logger.log(log)


@option(NLPAutoRegressionConfigs.optimizer)
def _optimizer(c: NLPAutoRegressionConfigs):
    """
    ### Default [optimizer configurations](../optimizers/configs.html)
    """

    optimizer = OptimizerConfigs()
    optimizer.parameters = c.model.parameters()
    optimizer.optimizer = 'Adam'
    optimizer.d_model = c.d_model

    return optimizer


@option(NLPAutoRegressionConfigs.n_tokens)
def _n_tokens(c: NLPAutoRegressionConfigs):
    """
    Get number of tokens
    """
    return c.text.n_tokens


@option(NLPAutoRegressionConfigs.tokenizer)
def basic_english():
    """
    ### Basic  english tokenizer

    We use character level tokenizer in this experiment.
    You can switch by setting,

    ```
        'tokenizer': 'basic_english',
    ```

    as the configurations dictionary when starting the experiment.

    """
    from torchtext.data import get_tokenizer
    return get_tokenizer('basic_english')


def character_tokenizer(x: str):
    """
    ### Character level tokenizer
    """
    return list(x)


@option(NLPAutoRegressionConfigs.tokenizer)
def character():
    """
    ### Character level tokenizer configuration
    """
    return character_tokenizer


@option(NLPAutoRegressionConfigs.text)
def tiny_shakespeare(c: NLPAutoRegressionConfigs):
    """
    ### Tiny Shakespeare dataset

    It will download from the url if not present
    """
    return TextFileDataset(
        lab.get_data_path() / 'tiny_shakespeare.txt',
        c.tokenizer,
        url='https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt')


@option(NLPAutoRegressionConfigs.train_loader)
def sequential_train_loader(c: NLPAutoRegressionConfigs):
    """
    ### Sequential training data loader
    """
    return SequentialDataLoader(text=c.text.train,
                                dataset=c.text,
                                batch_size=c.batch_size,
                                seq_len=c.seq_len)


@option(NLPAutoRegressionConfigs.valid_loader)
def sequential_valid_loader(c: NLPAutoRegressionConfigs):
    """
    ### Sequential validation data loader
    """
    return SequentialDataLoader(text=c.text.valid,
                                dataset=c.text,
                                batch_size=c.batch_size,
                                seq_len=c.seq_len)


def transpose_batch(batch):
    """
    ### Transpose batch

    `DataLoader` collects the batches on the first dimension.
    We need to transpose it to be sequence first.
    """

    transposed_data = list(zip(*batch))
    # Stack the batch along the second dimension `dim=1`
    src = torch.stack(transposed_data[0], dim=1)
    tgt = torch.stack(transposed_data[1], dim=1)

    return src, tgt


@option(NLPAutoRegressionConfigs.train_loader)
def shuffled_train_loader(c: NLPAutoRegressionConfigs):
    """
    ### Shuffled training data loader
    """
    return DataLoader(SequentialUnBatchedDataset(text=c.text.train,
                                                 dataset=c.text,
                                                 seq_len=c.seq_len),
                      batch_size=c.batch_size,
                      collate_fn=transpose_batch,
                      shuffle=True)


@option(NLPAutoRegressionConfigs.valid_loader)
def shuffled_valid_loader(c: NLPAutoRegressionConfigs):
    """
    ### Shuffled validation data loader
    """
    return DataLoader(SequentialUnBatchedDataset(text=c.text.valid,
                                                 dataset=c.text,
                                                 seq_len=c.seq_len),
                      batch_size=c.batch_size,
                      collate_fn=transpose_batch,
                      shuffle=True)
