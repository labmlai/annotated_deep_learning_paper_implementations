"""
---
title:  Pay Attention to MLPs (gMLP) Experiment
summary: This experiment trains a gMLP based model on Tiny Shakespeare dataset.
---

# [Pay Attention to MLPs (gMLP)](index.html) Experiment

This is an annotated PyTorch experiment to train a [gMLP model](index.html).
The paper also applies a Stochastic Depth regularization where some layers are removed randomly during training.
We have not implemented that here.

This is based on
[training loop and configurations for a simple transformer auto-regressive NLP task](../basic/autoregressive_experiment.html).

[![View Run](https://img.shields.io/badge/labml-experiment-brightgreen)](https://app.labml.ai/run/01bd941ac74c11eb890c1d9196651a4a)
"""
from labml import experiment
from labml.configs import option
from labml_nn.transformers import TransformerConfigs
from labml_nn.transformers.basic.autoregressive_experiment import Configs as BasicAutoRegressionConfigs
from labml_nn.transformers.gmlp import GMLPBlock


class Configs(BasicAutoRegressionConfigs):
    """
    ## Configurations

    This inherits from
    [training loop and configurations for a simple transformer auto-regressive NLP task](../basic/autoregressive_transformer.html).
    """

    # Transformer
    transformer: TransformerConfigs = 'gMLP'
    # gMLP Block
    gmlp: GMLPBlock
    # `d_ffn` for gMLP projection layer
    d_ffn: int = 2048


@option(Configs.gmlp, 'gMLP')
def _gmlp_configs(c: Configs):
    """
    ### Create a gMLP block
    """
    return GMLPBlock(c.d_model, c.d_ffn, c.seq_len)


@option(Configs.transformer, 'gMLP')
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
    # Set model size
    conf.d_model = c.d_model
    # Replace the encoder layer with a gMLP layer
    conf.encoder_layer = c.gmlp

    return conf


def main():
    # Create experiment
    experiment.create(name="gMLP")
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

        # Model size
        'd_model': 512,
        'd_ffn': 2048,

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
