from typing import Callable

from labml.configs import BaseConfigs, option


class TokenizerConfigs(BaseConfigs):
    """
    <a id="OptimizerConfigs">
    ## Optimizer Configurations
    </a>
    """

    tokenizer: Callable = 'character'

    def __init__(self):
        super().__init__(_primary='tokenizer')


@option(TokenizerConfigs.tokenizer)
def basic_english():
    """
    ### Basic  english tokenizer

    We use character level tokenizer in this experiment.
    You can switch by setting,

    ```
        'tokenizer': 'basic_english',
    ```

    as the configurations dictionary when starting the experiment.

    """
    from torchtext.data import get_tokenizer
    return get_tokenizer('basic_english')


def character_tokenizer(x: str):
    """
    ### Character level tokenizer
    """
    return list(x)


@option(TokenizerConfigs.tokenizer)
def character():
    """
    Character level tokenizer configuration
    """
    return character_tokenizer
