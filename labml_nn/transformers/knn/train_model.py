"""
---
title: Train Autoregressive Transformer
summary: This is training code with notes for a basic auto-regressive transformer.
---

# Train Autoregressive Transformer

This trains a simple [transformer](../../) model for auto-regression.
"""

import torch
from labml import experiment
from labml.configs import option
from labml.utils.pytorch import get_modules
from labml_helpers.module import Module

from labml_nn.experiments.nlp_autoregression import NLPAutoRegressionConfigs
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

    def forward(self, src: torch.Tensor):
        # Create subsequent mask, so that the transformer can only pay attention to past tokens.
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            self.src_mask = subsequent_mask(len(src)).to(src.device)
        # Embed the tokens (`src`) and run it through the the transformer
        res = self.encoder(self.src_embed(src), self.src_mask)
        # Generate logits of the next token
        return self.generator(res), None


class Configs(NLPAutoRegressionConfigs):
    """
    ## Configurations

    The default configs can and will be over-ridden when we start the experiment
    """

    transformer: TransformerConfigs
    model: AutoregressiveModel

    is_save_ff_input = False


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
    experiment.create(name="knn_lm")
    # Create configs
    conf = Configs()
    # Load configurations
    experiment.configs(conf,
                       # A dictionary of configurations to override
                       {'tokenizer': 'character',
                        'prompt_separator': '',
                        'prompt': 'It is ',
                        'text': 'tiny_shakespeare',

                        'optimizer.optimizer': 'Noam',
                        'optimizer.learning_rate': 1.,
                        'optimizer.d_model': 256,

                        'seq_len': 1024,
                        'epochs': 128,
                        'batch_size': 6,
                        'inner_iterations': 10,

                        # Transformer configurations
                        'transformer.d_model': 256,
                        'transformer.ffn.d_ff': 1024,
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
