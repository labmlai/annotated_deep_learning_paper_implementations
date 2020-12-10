"""
---
title: Train Autoregressive Transformer
summary: This is training code with notes for a basic auto-regressive transformer.
---

# Train Autoregressive Transformer

This trains a simple [transformer](../../) model for auto-regression.
"""

from typing import Callable, Any

import torch
import torch.nn as nn
from torchtext.data.utils import get_tokenizer

from labml import lab, experiment, monit, tracker, logger
from labml.configs import option
from labml.logger import Text
from labml.utils.pytorch import get_modules
from labml_helpers.datasets.text import TextDataset, SequentialDataLoader, TextFileDataset
from labml_helpers.metrics.accuracy import Accuracy
from labml_helpers.module import Module
from labml_helpers.optimizer import OptimizerConfigs
from labml_helpers.train_valid import SimpleTrainValidConfigs, BatchIndex
from labml_nn.transformers import Encoder, Generator, TransformerConfigs
from labml_nn.transformers.utils import subsequent_mask


class AutoregressiveModel(Module):
    """
    ## Auto regressive model
    """

    def __init__(self, src_embed: Module, encoder: Encoder, generator: Generator, *,
                 is_save_ff_input: bool = False):
        super().__init__()
        # Token embedding module
        self.src_embed = src_embed
        # Transformer based encoder
        self.encoder = encoder
        # Whether the last layer of the encoder should
        # save the input to the feed-forward layer.
        # This is out $f(c_t)$, the embedding of the context.
        self.encoder.layers[-1].is_save_ff_input = is_save_ff_input
        # Next token generation layer;
        # this give logits  of the the next token
        self.generator = generator
        # This will be initialized on the first call
        self.src_mask = None

    @property
    def ff_input(self) -> torch.Tensor:
        """
        Retrieve saved $f(c_t)$
        """
        return self.encoder.layers[-1].ff_input

    def __call__(self, src: torch.Tensor):
        # Create subsequent mask, so that the transformer can only pay attention to past tokens.
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            self.src_mask = subsequent_mask(len(src)).to(src.device)
        # Embed the tokens (`src`) and run it through the the transformer
        res = self.encoder(self.src_embed(src), self.src_mask)
        # Generate logits of the next token
        return self.generator(res)


class CrossEntropyLoss(Module):
    """
    Cross entropy loss
    """

    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

    def __call__(self, outputs, targets):
        return self.loss(outputs.view(-1, outputs.shape[-1]), targets.view(-1))


class Configs(SimpleTrainValidConfigs):
    """
    ## Configurations

    The default configs can and will be over-ridden when we start the experiment
    """

    transformer: TransformerConfigs
    model: AutoregressiveModel
    text: TextDataset
    batch_size: int = 20
    seq_len: int = 32
    n_tokens: int
    tokenizer: Callable = 'character'

    is_save_models = True
    prompt: str
    prompt_separator: str

    is_save_ff_input = False
    optimizer: torch.optim.Adam = 'transformer_optimizer'

    accuracy = Accuracy()
    loss_func = CrossEntropyLoss()

    def init(self):
        # Create a configurable optimizer.
        # Parameters like learning rate can be changed by passing a dictionary when starting the experiment.
        optimizer = OptimizerConfigs()
        optimizer.parameters = self.model.parameters()
        optimizer.d_model = self.transformer.d_model
        optimizer.optimizer = 'Noam'
        self.optimizer = optimizer

        # Create a sequential data loader for training
        self.train_loader = SequentialDataLoader(text=self.text.train,
                                                 dataset=self.text,
                                                 batch_size=self.batch_size,
                                                 seq_len=self.seq_len)

        # Create a sequential data loader for validation
        self.valid_loader = SequentialDataLoader(text=self.text.valid,
                                                 dataset=self.text,
                                                 batch_size=self.batch_size,
                                                 seq_len=self.seq_len)

        self.state_modules = [self.accuracy]

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
            output = self.model(data)
            # Get the model prediction (greedy)
            output = output.argmax(dim=-1).squeeze()
            # Add the prediction to prompt
            prompt += self.prompt_separator + self.text.itos[output[-1]]
            # Add the prediction for logging
            log += [(self.prompt_separator + self.text.itos[output[-1]], Text.value)]

        logger.log(log)

    def step(self, batch: Any, batch_idx: BatchIndex):
        """
        This method is called for each batch
        """
        self.model.train(self.mode.is_train)

        # Get data and target labels
        data, target = batch[0].to(self.model.device), batch[1].to(self.model.device)

        if self.mode.is_train:
            tracker.add_global_step(data.shape[0] * data.shape[1])

        # Run the model
        output = self.model(data)

        # Calculate loss
        loss = self.loss_func(output, target)
        # Calculate accuracy
        self.accuracy(output, target)

        # Log the loss
        tracker.add("loss.", loss)

        #  If we are in training mode, calculate the gradients
        if self.mode.is_train:
            loss.backward()
            self.optimizer.step()
            if batch_idx.is_last:
                tracker.add('model', self.model)
            self.optimizer.zero_grad()

        tracker.save()


@option(Configs.tokenizer)
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
    return get_tokenizer('basic_english')


def character_tokenizer(x: str):
    return list(x)


@option(Configs.tokenizer)
def character():
    """
    Character level tokenizer
    """
    return character_tokenizer


@option(Configs.text)
def tiny_shakespeare(c: Configs):
    """
    Initialize/load tiny shakespeare dataset

    This dataset is from Andrej Karpathy's [char-rnn](https://github.com/karpathy/char-rnn) project.
    """
    return TextFileDataset(
        lab.get_data_path() / 'tiny_shakespeare.txt', c.tokenizer,
        url='https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt')


@option(Configs.model)
def autoregressive_model(c: Configs):
    """
    Initialize the auto-regressive model
    """
    m = AutoregressiveModel(
        # Get the source token embedding layer, encoder and
        # final token generator from configurable transformer
        src_embed=c.transformer.src_embed,
        encoder=c.transformer.encoder,
        generator=c.transformer.generator,
        # Whether to save $f(c_t)$
        is_save_ff_input=c.is_save_ff_input)
    return m.to(c.device)


@option(Configs.transformer)
def transformer_c(c: Configs):
    """
    Initialize the configurable transformer encoder for our autoregressive model
    """
    tc = TransformerConfigs()
    tc.n_src_vocab = c.n_tokens
    tc.n_tgt_vocab = c.n_tokens

    return tc


def main():
    # Create experiment
    experiment.create(name="knn_lm", comment='', writers={'tensorboard', 'sqlite', 'screen'})
    # Create configs
    conf = Configs()
    # Load configurations
    experiment.configs(conf,
                       # A dictionary of configurations to override
                       {'tokenizer': 'character',
                        'prompt_separator': '',
                        'prompt': 'It is ',
                        'text': 'tiny_shakespeare',

                        'seq_len': 1024,
                        'epochs': 128,
                        'batch_size': 6,
                        'inner_iterations': 10,

                        # Transformer configurations
                        'transformer.d_model': 256,
                        'transformer.d_ff': 1024,
                        'transformer.n_heads': 8,
                        'transformer.n_layers': 6})

    # This is needed to initialize models
    conf.n_tokens = conf.text.n_tokens

    # Set models for saving and loading
    experiment.add_pytorch_models(get_modules(conf))

    # Start the experiment
    with experiment.start():
        # `TrainValidConfigs.run`
        conf.run()


if __name__ == '__main__':
    main()
