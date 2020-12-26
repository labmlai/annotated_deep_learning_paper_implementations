from typing import Callable, Any

import torch
import torch.nn as nn
from labml import lab, experiment, monit, tracker, logger
from labml.configs import option
from labml.logger import Text
from labml.utils.pytorch import get_modules
from labml_helpers.datasets.text import TextDataset, SequentialDataLoader, TextFileDataset
from labml_helpers.metrics.accuracy import Accuracy
from labml_helpers.module import Module
from labml_helpers.optimizer import OptimizerConfigs
from labml_helpers.train_valid import SimpleTrainValidConfigs, BatchIndex

from labml_nn.hypernetworks.hyper_lstm import HyperLSTM


class AutoregressiveModel(Module):
    """
    ## Auto regressive model
    """

    def __init__(self, n_vocab: int, d_model: int, n_rhn, n_z):
        super().__init__()
        # Token embedding module
        self.src_embed = nn.Embedding(n_vocab, d_model, n_rhn, n_z)
        self.lstm = HyperLSTM(d_model, d_model, n_rhn, n_z, 1)
        self.generator = nn.Linear(d_model, n_vocab)

    def __call__(self, x: torch.Tensor):
        x = self.src_embed(x)
        # Embed the tokens (`src`) and run it through the the transformer
        res, state = self.lstm(x)
        # Generate logits of the next token
        return self.generator(res), state


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

    model: AutoregressiveModel
    text: TextDataset
    batch_size: int = 20
    seq_len: int = 512
    n_tokens: int
    tokenizer: Callable = 'character'

    is_save_models = True

    optimizer: torch.optim.Adam = 'transformer_optimizer'

    accuracy = Accuracy()
    loss_func = CrossEntropyLoss()

    def init(self):
        # Create a configurable optimizer.
        # Parameters like learning rate can be changed by passing a dictionary when starting the experiment.
        optimizer = OptimizerConfigs()
        optimizer.parameters = self.model.parameters()
        optimizer.optimizer = 'Adam'
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
        prompt = 'It is'
        log = [(prompt, Text.subtle)]
        # Sample 25 tokens
        for i in monit.iterate('Sample', 25):
            # Tokenize the prompt
            data = self.text.text_to_i(prompt).unsqueeze(-1)
            data = data.to(self.device)
            # Get the model output
            output, state = self.model(data)
            output = output.cpu()
            # Get the model prediction (greedy)
            output = output.argmax(dim=-1).squeeze()
            # Add the prediction to prompt
            prompt += self.text.itos[output[-1]]
            # Add the prediction for logging
            log += [(self.text.itos[output[-1]], Text.value)]

        logger.log(log)

    def step(self, batch: Any, batch_idx: BatchIndex):
        """
        This method is called for each batch
        """
        self.model.train(self.mode.is_train)

        # Get data and target labels
        data, target = batch[0].to(self.device), batch[1].to(self.device)

        if self.mode.is_train:
            tracker.add_global_step(data.shape[0] * data.shape[1])

        # Run the model
        output, state = self.model(data)

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
    return TextFileDataset(
        lab.get_data_path() / 'tiny_shakespeare.txt', c.tokenizer,
        url='https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt')


@option(Configs.model)
def autoregressive_model(c: Configs):
    """
    Initialize the auto-regressive model
    """
    m = AutoregressiveModel(c.n_tokens, 512, 16, 16)
    return m.to(c.device)


def main():
    # Create experiment
    experiment.create(name="knn_lm", comment='')
    # Create configs
    conf = Configs()
    # Load configurations
    experiment.configs(conf,
                       # A dictionary of configurations to override
                       {'tokenizer': 'character',
                        'text': 'tiny_shakespeare',
                        'optimizer.learning_rate': 1e-4,

                        'seq_len': 512,
                        'epochs': 128,
                        'batch_size': 2,
                        'inner_iterations': 10})

    # This is needed to initialize models
    conf.n_tokens = conf.text.n_tokens

    # Set models for saving and loading
    experiment.add_pytorch_models(get_modules(conf))

    conf.init()
    # Start the experiment
    with experiment.start():
        # `TrainValidConfigs.run`
        conf.run()


if __name__ == '__main__':
    main()
