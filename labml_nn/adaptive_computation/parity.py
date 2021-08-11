from typing import Tuple

import torch
from torch.utils.data import Dataset


class ParityDataset(Dataset):
    def __init__(self, size: int, n_elems: int = 64):
        self.size = size
        self.n_elems = n_elems

    def __len__(self):
        return self.size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        n_non_zero = torch.randint(1, self.n_elems + 1, (1,)).item()
        x = torch.zeros((self.n_elems,))
        x[:n_non_zero] = torch.randint(0, 2, (n_non_zero,)) * 2 - 1
        x = x[torch.randperm(self.n_elems)]

        return x, (x == 1.).sum() % 2
