"""
---
title: Utilities
summary: A bunch of utility functions and classes
---

# Utilities
"""

import copy

from torch.utils.data import Dataset, IterableDataset

from labml_helpers.module import M, TypedModuleList


def clone_module_list(module: M, n: int) -> TypedModuleList[M]:
    """
    ## Clone Module

    Make a `nn.ModuleList` with clones of a given module
    """
    return TypedModuleList([copy.deepcopy(module) for _ in range(n)])


def cycle_dataloader(data_loader):
    """
    <a id="cycle_dataloader"></a>
    ## Cycle Data Loader

    Infinite loader that recycles the data loader after each epoch
    """
    while True:
        for batch in data_loader:
            yield batch


class MapStyleDataset(Dataset):
    """
    <a id="map_style_dataset"></a>
    ## Map Style Dataset

    This converts an [`IterableDataset`](https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset)
    to a [map-style dataset](https://pytorch.org/docs/stable/data.html#map-style-datasets)
    so that we can shuffle the dataset.

    *This only works when the dataset size is small and can be held in memory.*
    """

    def __init__(self, dataset: IterableDataset):
        # Load the data to memory
        self.data = [d for d in dataset]

    def __getitem__(self, idx: int):
        """Get a sample by index"""
        return self.data[idx]

    def __iter__(self):
        """Create an iterator"""
        return iter(self.data)

    def __len__(self):
        """Size of the dataset"""
        return len(self.data)
