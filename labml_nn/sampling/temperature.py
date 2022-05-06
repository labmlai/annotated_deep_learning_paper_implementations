import torch
from torch import nn
from torch.distributions import Categorical

from labml_nn.sampling import Sampler


class TemperatureSampler(Sampler):
    def __init__(self, temperature: float):
        self.temperature = temperature

    def __call__(self, logits: torch.Tensor):
        dist = Categorical(logits=logits / self.temperature)

        return dist.sample()
