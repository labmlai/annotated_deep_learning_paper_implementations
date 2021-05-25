from collections import Counter
from typing import Callable

import torch
import torchtext
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchtext.vocab import Vocab

from labml import lab, tracker, monit
from labml.configs import option
from labml_helpers.device import DeviceConfigs
from labml_helpers.metrics.accuracy import Accuracy
from labml_helpers.module import Module
from labml_helpers.train_valid import TrainValidConfigs, hook_model_outputs, BatchIndex
from labml_nn.optimizers.configs import OptimizerConfigs


class NLPClassificationConfigs(TrainValidConfigs):
    # Optimizer
    optimizer: torch.optim.Adam
    # Training device
    device: torch.device = DeviceConfigs()

    # Autoregressive model
    model: Module
    # Batch size
    batch_size: int = 16
    # Length of the sequence, or context size
    seq_len: int = 512
    # Vocabulary
    vocab: Vocab = 'ag_news'
    # Number of token in vocabulary
    n_tokens: int
    # Number of classes
    n_classes: int = 'ag_news'
    # Tokenizer
    tokenizer: Callable = 'character'

    # Whether to periodically save models
    is_save_models = True

    # Loss function
    loss_func = nn.CrossEntropyLoss()
    # Accuracy function
    accuracy = Accuracy()
    # Model embedding size
    d_model: int = 512
    # Gradient clipping
    grad_norm_clip: float = 1.0

    # Training data loader
    train_loader: DataLoader = 'ag_news'
    # Validation data loader
    valid_loader: DataLoader = 'ag_news'

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
            tracker.add_global_step(data.shape[1])

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


@option(NLPClassificationConfigs.optimizer)
def _optimizer(c: NLPClassificationConfigs):
    """
    ### Default optimizer configurations
    """

    optimizer = OptimizerConfigs()
    optimizer.parameters = c.model.parameters()
    optimizer.optimizer = 'Adam'
    optimizer.d_model = c.d_model

    return optimizer


@option(NLPClassificationConfigs.tokenizer)
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


@option(NLPClassificationConfigs.tokenizer)
def character():
    """
    ### Character level tokenizer configuration
    """
    return character_tokenizer


@option(NLPClassificationConfigs.n_tokens)
def _n_tokens(c: NLPClassificationConfigs):
    """
    Get number of tokens
    """
    return len(c.vocab) + 2


class CollateFunc:
    def __init__(self, tokenizer, vocab: Vocab, seq_len: int, padding: int, classifier_token: int):
        self.classifier_token = classifier_token
        self.padding = padding
        self.seq_len = seq_len
        self.vocab = vocab
        self.tokenizer = tokenizer

    def __call__(self, batch):
        label_list = []
        padded = torch.full((self.seq_len, len(batch)), self.padding, dtype=torch.long)

        for (i, (_label, _text)) in enumerate(batch):
            label_list.append(int(_label) - 1)
            _text = [self.vocab[token] for token in self.tokenizer(_text)]
            _text = _text[:self.seq_len]

            padded[:len(_text), i] = padded.new_tensor(_text)

        padded[-1, :] = self.classifier_token
        label_list = torch.tensor(label_list, dtype=torch.long)
        return padded, label_list


class AGNewsDataset(Dataset):
    def __init__(self, dataset):
        self.data = [d for d in dataset]

    def __getitem__(self, item):
        return self.data[item]

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)


@option([NLPClassificationConfigs.n_classes,
         NLPClassificationConfigs.vocab,
         NLPClassificationConfigs.train_loader,
         NLPClassificationConfigs.valid_loader])
def ag_news(c: NLPClassificationConfigs):
    train, valid = torchtext.datasets.AG_NEWS(root=str(lab.get_data_path() / 'ag_news'), split=('train', 'test'))
    with monit.section('Load data'):
        train, valid = AGNewsDataset(train), AGNewsDataset(valid)

    tokenizer = c.tokenizer
    counter = Counter()
    for (label, line) in train:
        counter.update(tokenizer(line))
    for (label, line) in valid:
        counter.update(tokenizer(line))
    vocab = Vocab(counter, min_freq=1)

    train_loader = DataLoader(train, batch_size=c.batch_size, shuffle=True,
                              collate_fn=CollateFunc(tokenizer, vocab, c.seq_len, len(vocab), len(vocab) + 1))
    valid_loader = DataLoader(valid, batch_size=c.batch_size, shuffle=True,
                              collate_fn=CollateFunc(tokenizer, vocab, c.seq_len, len(vocab), len(vocab) + 1))

    return 4, vocab, train_loader, valid_loader
