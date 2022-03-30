import copy

import torch
import torch.nn as nn

from labml import experiment
from labml.configs import option
from labml_helpers.module import Module
from labml_nn.experiments.nlp_autoregression import NLPAutoRegressionConfigs
from labml_nn.normalization.deep_norm import DeepNorm
from labml_nn.transformers import MultiHeadAttention
from labml_nn.transformers.feed_forward import FeedForward
from labml_nn.transformers.utils import subsequent_mask


class TransformerLayer(nn.Module):
    def __init__(self, *,
                 d_model: int,
                 self_attn: MultiHeadAttention,
                 feed_forward: FeedForward,
                 deep_norm_alpha: float,
                 deep_norm_beta: float,
                 ):
        """
        * `d_model` is the token embedding size
        * `self_attn` is the self attention module
        * `feed_forward` is the feed forward module
        * `dropout_prob` is the probability of dropping out after self attention and FFN
        """
        super().__init__()
        self.size = d_model
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.self_attn_norm = DeepNorm(deep_norm_alpha, [d_model])
        self.feed_forward_norm = DeepNorm(deep_norm_alpha, [d_model])

        with torch.no_grad():
            feed_forward.layer1.weight *= deep_norm_beta
            feed_forward.layer2.weight *= deep_norm_beta

            self_attn.value.linear.weight *= deep_norm_beta
            self_attn.output.weight *= deep_norm_beta

        # The mask will be initialized on the first call
        self.mask = None

    def forward(self, x: torch.Tensor):
        if self.mask is None or self.mask.size(0) != len(x):
            # Subsequent mask, will mask out tokens from seeing future tokens
            self.mask = subsequent_mask(len(x)).to(x.device)

        # Run through self attention, i.e. keys and values are from self
        x = self.self_attn_norm(x, self.self_attn(query=x, key=x, value=x, mask=self.mask))
        # Pass through the feed-forward network
        x = self.feed_forward_norm(x, self.feed_forward(x))

        return x


class AutoregressiveTransformer(Module):
    """
    ## Auto-Regressive model
    """

    def __init__(self, n_tokens: int, d_model: int, n_layers: int, layer: TransformerLayer):
        """
        * `encoder` is the transformer [Encoder](../models.html#Encoder)
        * `src_embed` is the token
        [embedding module (with positional encodings)](../models.html#EmbeddingsWithLearnedPositionalEncoding)
        * `generator` is the [final fully connected layer](../models.html#Generator) that gives the logits.
        """
        super().__init__()
        self.encoder = nn.Sequential(*[copy.deepcopy(layer) for _ in range(n_layers)])

        self.emb = nn.Embedding(n_tokens, d_model)
        self.readout = nn.Linear(d_model, n_tokens)

    def forward(self, x: torch.Tensor):
        # Get the token embeddings with positional encodings
        x = self.emb(x)
        # Transformer encoder
        x = self.encoder(x)
        # Get logits
        x = self.readout(x)

        # Return results
        return x, None


class Configs(NLPAutoRegressionConfigs):
    """
    ## Configurations

    This inherits from
    [`NLPAutoRegressionConfigs`](../../experiments/nlp_autoregression.html#NLPAutoRegressionConfigs)
    """

    # GPT model
    model: AutoregressiveTransformer

    n_layers: int = 64

    deep_norm_alpha: float
    deep_norm_beta: float

    n_heads: int = 4
    d_model: int = 64
    d_k: int = 16


@option(Configs.deep_norm_alpha)
def _deep_norm_alpha(c: Configs):
    return (2. * c.n_layers) ** 0.5


@option(Configs.deep_norm_beta)
def _deep_norm_beta(c: Configs):
    return (8. * c.n_layers) ** -0.5


@option(Configs.model)
def _model(c: Configs):
    m = AutoregressiveTransformer(c.n_tokens, c.d_model, c.n_layers,
                                  TransformerLayer(d_model=c.d_model,
                                                   deep_norm_alpha=c.deep_norm_alpha,
                                                   deep_norm_beta=c.deep_norm_beta,
                                                   feed_forward=FeedForward(d_model=c.d_model, d_ff=c.d_model * 4),
                                                   self_attn=MultiHeadAttention(c.n_heads, c.d_model,
                                                                                dropout_prob=0.0)))

    return m.to(c.device)


def main():
    # Create experiment
    experiment.create(name="deep_norm", writers={'screen', 'web_api', 'comet'})
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
        # Train for 32 epochs
        'epochs': 32,
        # Batch size $32$
        'batch_size': 16,
        # Switch between training and validation for $10$ times
        # per epoch
        'inner_iterations': 10,

        # Use [Noam optimizer](../../optimizers/noam.html)
        'optimizer.optimizer': 'Adam',
        'optimizer.learning_rate': 3e-4,

        # 'optimizer.optimizer': 'Noam',
        # 'optimizer.learning_rate': 1.,
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
