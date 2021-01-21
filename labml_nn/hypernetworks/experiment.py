import torch
import torch.nn as nn
from labml import experiment
from labml.configs import option
from labml.utils.pytorch import get_modules
from labml_helpers.module import Module

from labml_nn.experiments.nlp_autoregression import NLPAutoRegressionConfigs
from labml_nn.hypernetworks.hyper_lstm import HyperLSTM
from labml_nn.lstm import LSTM


class AutoregressiveModel(Module):
    """
    ## Auto regressive model
    """

    def __init__(self, n_vocab: int, d_model: int, rnn_model: Module):
        super().__init__()
        # Token embedding module
        self.src_embed = nn.Embedding(n_vocab, d_model)
        self.lstm = rnn_model
        self.generator = nn.Linear(d_model, n_vocab)

    def __call__(self, x: torch.Tensor):
        x = self.src_embed(x)
        # Embed the tokens (`src`) and run it through the the transformer
        res, state = self.lstm(x)
        # Generate logits of the next token
        return self.generator(res), state


class Configs(NLPAutoRegressionConfigs):
    """
    ## Configurations

    The default configs can and will be over-ridden when we start the experiment
    """

    model: AutoregressiveModel
    rnn_model: Module

    d_model: int = 512
    n_rhn: int = 16
    n_z: int = 16


@option(Configs.model)
def autoregressive_model(c: Configs):
    """
    Initialize the auto-regressive model
    """
    m = AutoregressiveModel(c.n_tokens, c.d_model, c.rnn_model)
    return m.to(c.device)


@option(Configs.rnn_model)
def hyper_lstm(c: Configs):
    return HyperLSTM(c.d_model, c.d_model, c.n_rhn, c.n_z, 1)


@option(Configs.rnn_model)
def lstm(c: Configs):
    return LSTM(c.d_model, c.d_model, 1)


def main():
    # Create experiment
    experiment.create(name="hyper_lstm", comment='')
    # Create configs
    conf = Configs()
    # Load configurations
    experiment.configs(conf,
                       # A dictionary of configurations to override
                       {'tokenizer': 'character',
                        'text': 'tiny_shakespeare',
                        'optimizer.learning_rate': 2.5e-4,
                        'optimizer.optimizer': 'Adam',
                        'prompt': 'It is',
                        'prompt_separator': '',

                        'rnn_model': 'hyper_lstm',

                        'train_loader': 'shuffled_train_loader',
                        'valid_loader': 'shuffled_valid_loader',

                        'seq_len': 512,
                        'epochs': 128,
                        'batch_size': 2,
                        'inner_iterations': 25})

    # Set models for saving and loading
    experiment.add_pytorch_models(get_modules(conf))

    # Start the experiment
    with experiment.start():
        # `TrainValidConfigs.run`
        conf.run()


if __name__ == '__main__':
    main()
