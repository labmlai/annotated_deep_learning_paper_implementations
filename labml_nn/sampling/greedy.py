import torch

from labml_nn.sampling import Sampler


class GreedySampler(Sampler):
    def __init__(self):
        pass

    def __call__(self, logits: torch.Tensor):
        return logits.argmax(dim=-1)
