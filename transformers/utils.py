import copy

from torch import nn

from labml.helpers.pytorch.module import Module


def clone_module_list(module: Module, n: int):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])
