from labml import experiment
from labml.configs import option
from labml_nn.transformers import TransformerConfigs
from labml_nn.transformers.basic.autoregressive_experiment import Configs as BasicAutoRegressionConfigs
from labml_nn.transformers.gmlp import GMLPBlock


class Configs(BasicAutoRegressionConfigs):
    """
    ## Configurations

    This inherits from
    [`NLPAutoRegressionConfigs`](../../experiments/nlp_autoregression.html#NLPAutoRegressionConfigs)
    """

    # Transformer
    transformer: TransformerConfigs = 'gMLP'
    # GMLP Block
    gmlp: GMLPBlock
    d_ffn: int = 2048


@option(Configs.gmlp, 'gMLP')
def _gmlp_configs(c: Configs):
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
    conf.d_model = c.d_model
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

        # Use a context size of $128$
        'seq_len': 256,
        # Train for $32$ epochs
        'epochs': 128,
        # Batch size $128$
        'batch_size': 32,
        # Switch between training and validation for $10$ times
        # per epoch
        'inner_iterations': 10,

        'd_model': 512,
        'd_ffn': 2048,

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
