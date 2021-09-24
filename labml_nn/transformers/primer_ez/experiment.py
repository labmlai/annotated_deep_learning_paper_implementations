"""
---
title: Primer EZ experiment
summary: This experiment trains Primer EZ on Tiny Shakespeare dataset.
---

# [Primer EZ](index.html) Experiment

This is an annotated PyTorch experiment to train a [Primer EZ transformer](index.html).

This is based on our [vanilla transformer experiment](../basic/experiment.html).
We use the same experiment and add the Primer EZ modifications.
"""

from labml import experiment
from labml.configs import option
from labml_nn.transformers import TransformerConfigs
from labml_nn.transformers.basic.autoregressive_experiment import Configs
from labml_nn.transformers.configs import FeedForwardConfigs
from labml_nn.transformers.primer_ez import SquaredReLU


@option(FeedForwardConfigs.activation, 'SquaredReLU')
def _squared_relu():
    """
    Add the [option](https://docs.labml.ai/api/configs.html#labml.configs.option)
     of [**squared ReLU**](index.html) to [configurable](../configs.html#FFN)
     [feed forward module](../feed_forward.html).
    """
    return SquaredReLU()


@option(TransformerConfigs.encoder_attn, 'MultiDConvHeadAttention')
def _d_conv_mha(c: TransformerConfigs):
    """
    Add the [option](https://docs.labml.ai/api/configs.html#labml.configs.option)
     of [**Multi-DConv-Head Attention**](index.html) to
     [configurable transformer](../configs.html#TransformerConfigs)
    """
    from labml_nn.transformers.primer_ez import MultiDConvHeadAttention
    return MultiDConvHeadAttention(c.n_heads, c.d_model, dropout_prob=c.dropout)


@option(TransformerConfigs.encoder_attn, 'MultiDSharedConvHeadAttention')
def _d_shared_conv_mha(c: TransformerConfigs):
    """
    Add the [option](https://docs.labml.ai/api/configs.html#labml.configs.option)
     of [**Multi Depth-wise Shared Conv Head Attention**](variations.html) to
     [configurable transformer](../configs.html#TransformerConfigs)

    📝 *This is a variation we tried*
    """
    from labml_nn.transformers.primer_ez.variations import MultiDSharedConvHeadAttention
    return MultiDSharedConvHeadAttention(c.n_heads, c.d_model, dropout_prob=c.dropout)


@option(TransformerConfigs.encoder_attn, 'MultiDPHConvHeadAttention')
def _d_per_head_conv_mha(c: TransformerConfigs):
    """
    Add the [option](https://docs.labml.ai/api/configs.html#labml.configs.option)
     of [**Multi Depth-wise Per Head Conv Head Attention**](variation.html) to
     [configurable transformer](../configs.html#TransformerConfigs)

    📝 *This is a variation we tried*
    """
    from labml_nn.transformers.primer_ez.variations import MultiDPHConvHeadAttention
    return MultiDPHConvHeadAttention(c.n_heads, c.d_model, dropout_prob=c.dropout)


def main():
    # Create experiment
    experiment.create(name="primer_ez")
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
        'transformer.ffn.d_ff': 2048,

        # Use Adam optimizer
        'optimizer.optimizer': 'Adam',
        'optimizer.learning_rate': 2.5e-4,

        # ⭐️ Use [**squared ReLU**](index.html) activation in the feed forward network.
        #
        # *Replace this with `ReLU` for $ReLU$.*
        'transformer.ffn.activation': 'SquaredReLU',

        # ⭐️ Use [**Multi-DConv-Head Attention**](index.html) for encoder attention.
        #
        # *Replace this with `mha` for original multi-head attention.*
        'transformer.encoder_attn': 'MultiDConvHeadAttention',
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
