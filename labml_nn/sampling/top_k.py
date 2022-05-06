import torch
from torch import nn
from torch.distributions import Categorical

from labml_nn.sampling import Sampler


class TopKSampler(Sampler):
    def __init__(self, k: int, sampler: Sampler):
        self.k = k
        self.sampler = sampler

    def __call__(self, logits: torch.Tensor):
        zeros = logits.new_ones(logits.shape) * float('-inf')
        values, indices = torch.topk(logits, self.k, dim=-1)
        zeros.scatter_(-1, indices, values)

        return self.sampler(zeros)
