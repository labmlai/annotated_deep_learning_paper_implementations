"""
---
title: Transformer Auto-Regression Experiment
summary: >
  This trains a simple transformer model on NLP auto-regression.
---

# Transformer Auto-Regression Experiment

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/transformers/basic/autoregressive_experiment.ipynb)

This trains a simple transformer introduced in [Attention Is All You Need](https://papers.labml.ai/paper/1706.03762)
on an NLP auto-regression task (with Tiny Shakespeare dataset).
"""

import torch
from torch import nn

from labml import experiment
from labml.configs import option
from labml_nn.experiments.nlp_autoregression import NLPAutoRegressionConfigs
from labml_nn.transformers import TransformerConfigs, Encoder
from labml_nn.transformers.utils import subsequent_mask


class AutoregressiveTransformer(nn.Module):
    """
    ## Auto-Regressive model
    """
    def __init__(self, encoder: Encoder, src_embed: nn.Module, generator: nn.Module):
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
    #
    conf.d_model = c.d_model

    #
    return conf


@option(Configs.model)
def _model(c: Configs):
    """
    Create GPT model and initialize weights
    """
    m = AutoregressiveTransformer(c.transformer.encoder,
                                  c.transformer.src_embed,
                                  c.transformer.generator).to(c.device)

    return m


def main():
    # Create experiment
    experiment.create(name="transformer")
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
        'seq_len': 512,
        # Train for 32 epochs
        'epochs': 32,
        # Batch size $32$
        'batch_size': 16,
        # Switch between training and validation for $10$ times
        # per epoch
        'inner_iterations': 10,

        # Model size
        'd_model': 256,
        'transformer.n_heads': 16,
        'transformer.ffn.d_ff': 1024,

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
