"""
---
title: Utilities
summary: A bunch of utility functions and classes
---

# Utilities
"""

import copy

from labml_helpers.module import M, TypedModuleList


def clone_module_list(module: M, n: int) -> TypedModuleList[M]:
    """
    ## Make a `nn.ModuleList` with clones of a given layer
    """
    return TypedModuleList([copy.deepcopy(module) for _ in range(n)])


def cycle_dataloader(data_loader):
    while True:
        for batch in data_loader:
            yield batch
