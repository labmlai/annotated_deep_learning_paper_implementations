import torch
import torch.nn as nn
from labml import experiment
from labml.configs import option
from labml.utils.pytorch import get_modules
from labml_helpers.module import Module

from labml_nn.experiments.nlp_autoregression import NLPAutoRegressionConfigs
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


class Configs(NLPAutoRegressionConfigs):
    """
    ## Configurations

    The default configs can and will be over-ridden when we start the experiment
    """

    model: AutoregressiveModel


@option(Configs.model)
def autoregressive_model(c: Configs):
    """
    Initialize the auto-regressive model
    """
    m = AutoregressiveModel(c.n_tokens, 512, 16, 16)
    return m.to(c.device)


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

                        'train_loader': 'shuffled_train_loader',
                        'valid_loader': 'shuffled_valid_loader',

                        'seq_len': 512,
                        'epochs': 128,
                        'batch_size': 2,
                        'inner_iterations': 25})

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
