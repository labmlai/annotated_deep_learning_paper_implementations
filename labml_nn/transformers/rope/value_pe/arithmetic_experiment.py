"""
---
title: Rotary Positional Embeddings with Relative distance (RoPER) Experiment
summary: This experiment trains a transformer model with Rotary Positional Embeddings with
 Relative Distance (RoPER) on the arithmetic addition task.
---

# Rotary Positional Embeddings with Relative distance ([RoPER](index.html)) Experiment
"""

from labml import experiment
from labml.configs import calculate
from labml_nn.experiments.arithmetic_dataset import ArithmeticAutoregression
from labml_nn.transformers import TransformerConfigs
from labml_nn.transformers.rope.experiment import Configs as RoPEConfigs


class Configs(RoPEConfigs, ArithmeticAutoregression):
    """
    We inherit [RoPE experiment](../experiment.html) and use it for
    [arithmetic addition task](../../experiments/arithmetic_dataset.html).

    We add the option to change attention to use Rotary Positional Embeddings with Relative distance (RoPER)
    below.
    """
    pass


def _rotary_value_pe_mha(c: TransformerConfigs):
    """
    Use Rotary Positional Embeddings with Relative distance ([RoPER](index.html)) in attention.
    """
    from labml_nn.transformers.rope.value_pe import RotaryValuePEMultiHeadAttention
    return RotaryValuePEMultiHeadAttention(c.n_heads, c.d_model, 1., 1.)


# Configuration options
calculate(TransformerConfigs.encoder_attn, 'rotary_value', _rotary_value_pe_mha)
calculate(TransformerConfigs.decoder_attn, 'rotary_value', _rotary_value_pe_mha)
calculate(TransformerConfigs.decoder_mem_attn, 'rotary_value', _rotary_value_pe_mha)


def main():
    # Create experiment
    experiment.create(name="roper_addition", comment="rotary value 7", writers={'screen', 'labml'})
    # Create configs
    conf = Configs()
    # Override configurations
    experiment.configs(conf, {
        'max_digits': 7,

        # No fixed positional embeddings
        'transformer.src_embed': 'no_pos',
        'transformer.tgt_embed': 'no_pos',

        # Encoder with RoPER attention
        'transformer.encoder_attn': 'rotary_value',
        # Encoder with RoPE attention
        # 'transformer.encoder_attn': 'rotary',

        #
        'model': 'rotary_pe_transformer',

        # Use a context size of $256$
        'seq_len': 512,
        # Train for 32 epochs
        'epochs': 20,
        # Batch size $4$
        'batch_size': 16,

        # Model size
        'd_model': 128,
        'transformer.ffn.d_ff': 512,
        'transformer.n_heads': 4,
        'transformer.dropout': 0.0,

        # Use [Adam optimizer](../../optimizers/noam.html)
        'optimizer.optimizer': 'Adam',
        'optimizer.learning_rate': 2.5e-4,
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
