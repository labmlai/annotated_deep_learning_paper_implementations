import torch
import torch.nn as nn

from labml import experiment
from labml.configs import option
from labml_helpers.module import Module
from labml_nn.experiments.nlp_autoregression import NLPAutoRegressionConfigs


class AutoregressiveModel(Module):
    """
    ## Auto regressive model
    """

    def __init__(self, n_vocab: int, d_model: int, transformer: Module):
        super().__init__()
        # Token embedding module
        self.src_embed = nn.Embedding(n_vocab, d_model)
        self.transformer = transformer
        self.generator = nn.Linear(d_model, n_vocab)

    def __call__(self, x: torch.Tensor):
        x = self.src_embed(x)
        # Embed the tokens (`src`) and run it through the the transformer
        res = self.transformer(x)
        # Generate logits of the next token
        return self.generator(res), None


class Configs(NLPAutoRegressionConfigs):
    """
    ## Configurations

    The default configs can and will be over-ridden when we start the experiment
    """

    model: AutoregressiveModel

    d_model: int = 512
    heads: int = 8
    dropout: float = 0.0
    d_ff: int = 2048
    n_layers: int = 6


@option(Configs.model)
def autoregressive_model(c: Configs):
    from labml_nn.transformers.feedback import FeedbackTransformer, FeedbackTransformerLayer, \
        FeedbackAttention, FeedForward

    return AutoregressiveModel(
        c.n_tokens, c.d_model,
        FeedbackTransformer(
            FeedbackTransformerLayer(d_model=c.d_model,
                                     attn=FeedbackAttention(c.heads, c.d_model, c.dropout),
                                     feed_forward=FeedForward(c.d_model, c.d_ff, c.dropout),
                                     dropout_prob=c.dropout),
            c.n_layers)).to(c.device)


def main():
    # Create experiment
    experiment.create(name="feedback_transformer")
    # Create configs
    conf = Configs()
    # Load configurations
    experiment.configs(conf,
                       # A dictionary of configurations to override
                       {'tokenizer': 'character',
                        'text': 'tiny_shakespeare',
                        'optimizer.learning_rate': 1.0,
                        'optimizer.optimizer': 'Noam',
                        'prompt': 'It is',
                        'prompt_separator': '',

                        'train_loader': 'shuffled_train_loader',
                        'valid_loader': 'shuffled_valid_loader',

                        'seq_len': 64,
                        'epochs': 128,
                        'batch_size': 80,
                        'inner_iterations': 25})

    # Set models for saving and loading
    experiment.add_pytorch_models({'model': conf.model})

    conf.init()
    # Start the experiment
    with experiment.start():
        conf.run()


if __name__ == '__main__':
    main()
