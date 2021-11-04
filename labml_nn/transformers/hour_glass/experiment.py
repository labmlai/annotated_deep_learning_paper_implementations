import math
from typing import List

from torch import nn

from labml import experiment
from labml.configs import option
from labml_helpers.module import Module
from labml_nn.experiments.nlp_autoregression import NLPAutoRegressionConfigs
from labml_nn.transformers.positional_encoding import PositionalEncoding
from transformer.hierarchy.hourglass import HourGlass


class AutoregressiveTransformer(Module):
    def __init__(self, n_tokens, d_model, dropout: float, hour_glass: HourGlass):
        super().__init__()
        self.pos_embedding = PositionalEncoding(d_model, dropout)
        self.hour_glass = hour_glass
        self.norm = nn.LayerNorm([d_model])
        self.embedding = nn.Embedding(n_tokens, d_model)
        self.d_model = d_model
        self.output = nn.Linear(d_model, n_tokens)

    def __call__(self, x):
        x = self.embedding(x)

        if self.pos_embedding is not None:
            x = self.pos_embedding(x * math.sqrt(self.d_model))

        x = self.hour_glass(x)
        x = self.norm(x)

        output = self.output(x)

        return output, None


class Configs(NLPAutoRegressionConfigs):
    model: AutoregressiveTransformer
    n_heads: int = 4
    dropout: float = 0.1
    d_ff: int = 512
    d_model: int = 128
    shortening_factors: List[int] = [8]


@option(Configs.model)
def _model(c: Configs):
    hour_glass = HourGlass(c.n_heads, c.d_model, c.dropout, c.d_ff, c.shortening_factors)
    m = AutoregressiveTransformer(c.n_tokens, c.d_model, c.dropout, hour_glass).to(c.device)

    return m


def main():
    # Create experiment
    experiment.create(name="hour_glass")
    # Create configs
    conf = Configs()
    # Override configurations
    experiment.configs(conf, {
        # Use character level tokenizer
        'tokenizer': 'character',
        # Prompt separator is blank
        'prompt_separator': '',
        # Starting prompt for sampling
        'prompt': 'It is ',
        # Use Tiny Shakespeare dataset
        'text': 'tiny_shakespeare',

        # Use a context size of $256$
        'seq_len': 256,
        # Train for $128$ epochs
        'epochs': 128,
        # Batch size $32$
        'batch_size': 32,
        # Switch between training and validation for $10$ times
        # per epoch
        'inner_iterations': 10,

        # Use [Noam optimizer](../../optimizers/noam.html)
        'optimizer.optimizer': 'Noam',
        'optimizer.learning_rate': 1.,
    })

    # Set models for saving and loading
    experiment.add_pytorch_models({'model': conf.model})

    # Start the experiment
    with experiment.start():
        # Run training
        conf.run()


#
if __name__ == '__main__':
    main()
