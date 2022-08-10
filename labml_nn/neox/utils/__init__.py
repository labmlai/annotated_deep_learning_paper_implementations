"""
---
title: Utilities and Helpers
summary: >
    Utilities and helper functions
---

# Utilities and Helpers

* [Cache for intermediate activations (for faster inference)](cache.html)
* [Tools for finetuning](finetune.html)
* [Trainer](trainer.html)
* [Text dataset](text_dataset.html)
"""
import typing
from typing import List, Optional

import torch

from labml import logger
from labml.logger import Text
from labml_nn.neox.tokenizer import get_tokenizer

if typing.TYPE_CHECKING:
    from tokenizers import Tokenizer

# Tokenizer singleton
_TOKENIZER: Optional['Tokenizer'] = None


def get_tokens(text: str) -> List[int]:
    """
    ### Get token ids

    :param text: is the text to tokenize
    :return: the token ids
    """
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = get_tokenizer()
    return _TOKENIZER.encode_batch([text])[0].ids


def print_token_outputs(ids: List[int], *xs: torch.Tensor):
    """
    ### Print tokens from model outputs

    Pretty prints target tokens along side outputs from the model(s).

    :param ids: are the target token ids
    :param xs: are the model(s) outputs
    """
    ids = ids + [-1]
    xs = [[-1] + x[0].max(dim=-1)[1].tolist() for x in xs]

    print_tokens(ids, xs)


def print_tokens(target: List[int], others: List[List[int]]):
    """
    ### Print tokens

    Pretty prints tokens for comparison

    :param target: are the target token ids
    :param others: are the sampled outputs from the model(s)
    """

    # Load tokenizer
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = get_tokenizer()

    # Convert the tokens to list of strings
    text = []
    for i in range(len(target)):
        tokens = [_TOKENIZER.decode([target[i]]) if target[i] != -1 else '---']
        for j in range(len(others)):
            tokens.append(_TOKENIZER.decode([others[j][i]]) if others[j][i] != -1 else '---')

        text.append(tokens)

    # Stats
    correct = [0 for _ in others]
    total = 0

    # Iterate through tokens
    for i in range(len(target)):
        parts = [(f'{i}: ', Text.meta)]
        parts += [('"', Text.subtle), (text[i][0], Text.subtle), ('"', Text.subtle), '\t']

        # Empty target
        if target[i] == -1:
            for j in range(len(others)):
                parts += [('"', Text.subtle), (text[i][j + 1], Text.subtle), ('"', Text.subtle), '\t']

            logger.log(parts)
            continue

        # Number of tokens
        total += 1

        # Other outputs
        for j in range(len(others)):
            correct[j] += 1 if others[j][i] == target[i] else 0

            parts += [('"', Text.subtle),
                      (text[i][j + 1], Text.success if others[j][i] == target[i] else Text.danger),
                      ('"', Text.subtle), '\t']

        logger.log(parts)

    # Stats
    parts = [(f'{total}', Text.highlight), '\t']
    for j in range(len(others)):
        parts += [(f'{correct[j]}', Text.value), '\t']
    logger.log(parts)


def balance_layers_simple(n_layers: int, n_chunks: int):
    """
    ### Balance layers

    Split the `n_layers` into `n_chunks`. This is used for pipeline parallel training.

    :param n_layers: is the number of layers
    :param n_chunks: is the number of chunks
    :return: returns a list with the number of layers for each chunk
    """
    balance = []
    for i in range(n_chunks):
        balance.append((n_layers - sum(balance)) // (n_chunks - i))

    return list(reversed(balance))
