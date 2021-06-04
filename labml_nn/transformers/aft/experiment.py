"""
---
title: Attention Free Transformer (AFT) Experiment
summary: This experiment trains an Attention Free Transformer (AFT) based model on Tiny Shakespeare dataset.
---

# [Attention Free Transformer (AFT)](index.html) Experiment

This is an annotated PyTorch experiment to train a [AFT model](index.html).

This is based on
[general training loop and configurations for auto-regressive NLP task](../../experiments/nlp_autoregression.html).

[![View Run](https://img.shields.io/badge/labml-experiment-brightgreen)](https://app.labml.ai/run/6348e504c3a511eba9529daa283fb495)
"""
import torch

from labml import experiment
from labml.configs import option
from labml_helpers.module import Module
from labml_nn.experiments.nlp_autoregression import NLPAutoRegressionConfigs
from labml_nn.transformers import TransformerConfigs, Encoder
from labml_nn.transformers.utils import subsequent_mask


class AutoregressiveTransformer(Module):
    """
    ## Simple autoregressive model

    This consists of a token embedding layer, transformer encoder, and
    a final linear layer that gives token logits.
    """

    def __init__(self, encoder: Encoder, src_embed: Module, generator: Module):
        """
        * `encoder` is the transformer [Encoder](../models.html#Encoder)
        * `src_embed` is the token
        [embedding module (with positional encodings)](../models.html#EmbeddingsWithLearnedPositionalEncoding)
        * `generator` is the [final fully connected layer](../models.html#Generator) that gives the logits.
        """
        super().__init__()
        self.src_embed = src_embed
        self.encoder = encoder
        self.generator = generator

        # The mask will be initialized on the first call
        self.mask = None

    def forward(self, x: torch.Tensor):
        # Create subsequent mask if mask is not initialized
        # or if the size of the mask is different
        if self.mask is None or self.mask.size(0) != len(x):
            # Subsequent mask, will mask out tokens from seeing future tokens
            self.mask = subsequent_mask(len(x)).to(x.device)

        # Get the token embeddings with positional encodings
        x = self.src_embed(x)
        # Transformer encoder
        x = self.encoder(x, self.mask)
        # Get logits
        x = self.generator(x)

        # Return results
        # (second value is for state, since our trainer is used with RNNs also)
        return x, None


class Configs(NLPAutoRegressionConfigs):
    """
    ## Configurations

    This inherits from
    [`NLPAutoRegressionConfigs`](../../experiments/nlp_autoregression.html#NLPAutoRegressionConfigs)
    """

    # GPT model
    model: AutoregressiveTransformer
    # Transformer
    transformer: TransformerConfigs

    local_window_size: int = 32


@option(Configs.transformer, 'Transformer')
def _transformer_configs(c: Configs):
    """
    ### Transformer configurations
    """

    # We use our
    # [configurable transformer implementation](../configs.html#TransformerConfigs)
    conf = TransformerConfigs()
    # Set the vocabulary sizes for embeddings and generating logits
    conf.n_src_vocab = c.n_tokens
    conf.n_tgt_vocab = c.n_tokens
    # Set the embedding size
    conf.d_model = c.d_model
    # Replace self-attention with an [AFT Local Module](index.html)
    from labml_nn.transformers.aft import AFTLocal
    conf.encoder_attn = AFTLocal(c.d_model, c.seq_len, c.local_window_size)

    #
    return conf


@option(Configs.model)
def _model(c: Configs):
    """
    Create an auto-regressive model
    """
    m = AutoregressiveTransformer(c.transformer.encoder,
                                  c.transformer.src_embed,
                                  c.transformer.generator).to(c.device)

    return m


def main():
    # Create experiment
    experiment.create(name="aft")
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

        # Use a context size of $128$
        'seq_len': 256,
        # Train for $32$ epochs
        'epochs': 128,
        # Batch size $128$
        'batch_size': 32,
        # Switch between training and validation for $10$ times
        # per epoch
        'inner_iterations': 10,

        # Embedding size
        'd_model': 128,
        # FFN hidden dimension size
        'transformer.ffn.d_ff': 256,

        # Optimizer
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
