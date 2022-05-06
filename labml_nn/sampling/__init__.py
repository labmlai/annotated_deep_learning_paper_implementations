import torch


class Sampler:
    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()