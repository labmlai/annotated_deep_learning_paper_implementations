from typing import Callable

import torch
import torch.nn as nn
from labml import lab, monit, logger, tracker
from labml.configs import option
from labml.logger import Text
from labml_helpers.datasets.text import TextDataset, SequentialDataLoader, SequentialUnBatchedDataset, TextFileDataset
from labml_helpers.device import DeviceConfigs
from labml_helpers.metrics.accuracy import Accuracy
from labml_helpers.module import Module
from labml_helpers.train_valid import TrainValidConfigs, hook_model_outputs, BatchIndex
from torch.utils.data import DataLoader

from labml_nn.optimizers.configs import OptimizerConfigs


class CrossEntropyLoss(Module):
    """
    Cross entropy loss
    """

    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

    def __call__(self, outputs, targets):
        return self.loss(outputs.view(-1, outputs.shape[-1]), targets.view(-1))


class NLPAutoRegressionConfigs(TrainValidConfigs):
    optimizer: torch.optim.Adam
    device: torch.device = DeviceConfigs()

    model: Module
    text: TextDataset
    batch_size: int = 16
    seq_len: int = 512
    n_tokens: int
    tokenizer: Callable = 'character'

    prompt: str
    prompt_separator: str

    is_save_models = True

    loss_func = CrossEntropyLoss()
    accuracy = Accuracy()
    d_model: int = 512

    def init(self):
        tracker.set_scalar("accuracy.*", True)
        tracker.set_scalar("loss.*", True)
        hook_model_outputs(self.mode, self.model, 'model')
        self.state_modules = [self.accuracy]

    def step(self, batch: any, batch_idx: BatchIndex):
        data, target = batch[0].to(self.device), batch[1].to(self.device)

        if self.mode.is_train:
            tracker.add_global_step(data.shape[0] * data.shape[1])

        with self.mode.update(is_log_activations=batch_idx.is_last):
            output, *_ = self.model(data)

        loss = self.loss_func(output, target)
        self.accuracy(output, target)
        self.accuracy.track()
        tracker.add("loss.", loss)

        if self.mode.is_train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.)
            self.optimizer.step()
            if batch_idx.is_last:
                tracker.add('model', self.model)
            self.optimizer.zero_grad()

        tracker.save()

    def sample(self):
        """
        Sampling function to generate samples periodically while training
        """
        prompt = self.prompt
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

        logger.log(log)


@option(NLPAutoRegressionConfigs.optimizer)
def _optimizer(c: NLPAutoRegressionConfigs):
    optimizer = OptimizerConfigs()
    optimizer.parameters = c.model.parameters()
    optimizer.optimizer = 'Adam'
    optimizer.d_model = c.d_model

    return optimizer


@option(NLPAutoRegressionConfigs.n_tokens)
def _n_tokens(c: NLPAutoRegressionConfigs):
    return c.text.n_tokens


@option(NLPAutoRegressionConfigs.tokenizer)
def basic_english():
    """
    Basic  english tokenizer

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
    return list(x)


@option(NLPAutoRegressionConfigs.tokenizer)
def character():
    return character_tokenizer


@option(NLPAutoRegressionConfigs.text)
def tiny_shakespeare(c: NLPAutoRegressionConfigs):
    return TextFileDataset(
        lab.get_data_path() / 'tiny_shakespeare.txt', c.tokenizer,
        url='https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt')


@option(NLPAutoRegressionConfigs.train_loader)
def sequential_train_loader(c: NLPAutoRegressionConfigs):
    return SequentialDataLoader(text=c.text.train,
                                dataset=c.text,
                                batch_size=c.batch_size,
                                seq_len=c.seq_len)


@option(NLPAutoRegressionConfigs.valid_loader)
def sequential_valid_loader(c: NLPAutoRegressionConfigs):
    return SequentialDataLoader(text=c.text.valid,
                                dataset=c.text,
                                batch_size=c.batch_size,
                                seq_len=c.seq_len)


def transpose_batch(batch):
    transposed_data = list(zip(*batch))
    src = torch.stack(transposed_data[0], 1)
    tgt = torch.stack(transposed_data[1], 1)

    return src, tgt


@option(NLPAutoRegressionConfigs.train_loader)
def shuffled_train_loader(c: NLPAutoRegressionConfigs):
    return DataLoader(SequentialUnBatchedDataset(text=c.text.train,
                                                 dataset=c.text,
                                                 seq_len=c.seq_len),
                      batch_size=c.batch_size,
                      collate_fn=transpose_batch,
                      shuffle=True)


@option(NLPAutoRegressionConfigs.valid_loader)
def shuffled_valid_loader(c: NLPAutoRegressionConfigs):
    return DataLoader(SequentialUnBatchedDataset(text=c.text.valid,
                                                 dataset=c.text,
                                                 seq_len=c.seq_len),
                      batch_size=c.batch_size,
                      collate_fn=transpose_batch,
                      shuffle=True)
