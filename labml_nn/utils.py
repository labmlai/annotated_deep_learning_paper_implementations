"""
---
title: Utilities
summary: A bunch of utility functions and classes
---

# Utilities
"""

import copy

from torch import nn

from labml_helpers.module import Module


def clone_module_list(module: Module, n: int):
    """
    ## Make a `nn.ModuleList` with clones of a given layer
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])
