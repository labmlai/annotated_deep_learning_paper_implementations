from labml import experiment
from labml.configs import option
from labml_nn.transformers import TransformerConfigs
from labml_nn.transformers.basic.autoregressive_experiment import Configs
from labml_nn.transformers.configs import FeedForwardConfigs
from labml_nn.transformers.primer_ez import SquaredReLU


@option(FeedForwardConfigs.activation, 'SquaredReLU')
def _squared_relu():
    return SquaredReLU()


@option(TransformerConfigs.encoder_attn, 'MultiDConvHeadAttention')
def _primer_ez_mha(c: TransformerConfigs):
    from labml_nn.transformers.primer_ez import MultiDConvHeadAttention
    return MultiDConvHeadAttention(c.n_heads, c.d_model, dropout_prob=c.dropout)


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

        'transformer.ffn.activation': 'SquaredReLU',
        'transformer.encoder_attn': 'MultiDConvHeadAttention',

        # 'transformer.ffn.activation': 'ReLU',
        # 'transformer.encoder_attn': 'mha',

        # Use Adam optimizer
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
