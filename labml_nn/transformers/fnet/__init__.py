from typing import Optional

import torch
from torch import nn


class FNet(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: Optional[torch.Tensor] = None):
        assert query is key and key is value
        assert mask is None

        x = query

        fft_hidden = torch.fft.fft(x, dim=2)
        fft_seq = torch.fft.fft(fft_hidden, dim=0)

        return torch.real(fft_seq)
