from labml.configs import BaseConfigs


class RWKVConfigs(BaseConfigs):
    """
    ## Transformer Configurations

    This defines configurations for a transformer.
    The configurations are calculate using option functions.
    These are lazy loaded and therefore only the necessary modules
    are calculated.
    """
    # Number of attention heads
    n_heads: int = 8
    # Transformer embedding size
    d_model: int = 512
    # Number of layers
    n_layers: int = 6
    # Dropout probability
    dropout: float = 0.1
    # Number of tokens in the source vocabulary (for token embeddings)
    n_src_vocab: int
    # Number of tokens in the target vocabulary (to generate logits for prediction)
    n_tgt_vocab: int
