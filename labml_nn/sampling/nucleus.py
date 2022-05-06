import torch
from torch import nn
from torch.distributions import Categorical

from labml_nn.sampling import Sampler


class NucleusSampler(Sampler):
    def __init__(self, p: float, sampler: Sampler):
        self.p = p
        self.sampler = sampler
        self.softmax = nn.Softmax(dim=-1)

    def __call__(self, logits: torch.Tensor):
        probs = self.softmax(logits)

        # Sort probabilities
        sorted_probs, indices = torch.sort(probs, dim=-1, descending=True)
        # Find what is outside the nucleus.
        # Get the cumsum
        non_nucleus = (torch.cumsum(sorted_probs, dim=-1) > self.p)[..., 1:]
        # Prepend zeros
        non_nucleus = torch.cat([non_nucleus.new_zeros(non_nucleus.shape[:-1] + (1,)), non_nucleus], dim=-1)

        # Get log probabilities and mask out the non-nucleus
        sorted_log_probs = torch.log(sorted_probs)
        sorted_log_probs[non_nucleus] = float('-inf')

        # Sample
        sampled_sorted_indexes = self.sampler(sorted_log_probs)

        # Get the actual indexes
        res = indices.gather(-1, sampled_sorted_indexes.unsqueeze(-1))

        #
        return res.squeeze(-1)
