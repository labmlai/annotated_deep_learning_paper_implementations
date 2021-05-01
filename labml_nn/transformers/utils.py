"""
---
title: Utilities for Transformer
summary: A bunch of utility functions and classes for transformers.
---

# Utilities for Transformer
"""

import torch


def subsequent_mask(seq_len):
    """
    ## Subsequent mask to mask out data from future (subsequent) time steps
    """
    mask = torch.tril(torch.ones(seq_len, seq_len)).to(torch.bool).unsqueeze(-1)
    return mask


def _subsequent_mask():
    from labml.logger import inspect
    inspect(subsequent_mask(10)[:, :, 0])


if __name__ == '__main__':
    _subsequent_mask()
