"""
---
title: GPT-NeoX Tokenizer
summary: >
    Loads the GPT-NeoX tokenizer
---

# GPT-NeoX Tokenizer

This initializes a Hugging Face tokenizer from the downloaded vocabulary.
"""

from tokenizers import Tokenizer

from labml import lab, monit


@monit.func('Load NeoX Tokenizer')
def get_tokenizer() -> Tokenizer:
    """
    ### Load NeoX Tokenizer

    :return: the tokenizer
    """
    vocab_file = lab.get_data_path() / 'neox' / 'slim_weights' / '20B_tokenizer.json'
    tokenizer = Tokenizer.from_file(str(vocab_file))

    return tokenizer
