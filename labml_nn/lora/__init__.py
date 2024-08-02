"""
# LoRA
"""

import torch
import torch.nn as nn


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool,
                 r: int, alpha: int = None):
        super().__init__()
        if alpha is None:
            alpha = r
        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        self.weight.requires_grad = False

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
            self.bias.requires_grad = False
        else:
            self.bias = None

        self.scaling = alpha / r
        self.lora_a = nn.Parameter(torch.empty((in_features, r)))
        self.lora_b = nn.Parameter(torch.empty((r, out_features)))

        with torch.no_grad():
            nn.init.kaiming_uniform_(self.lora_a, a=5 ** 0.5)
            nn.init.zeros_(self.lora_b)

    def forward(self, x: torch.Tensor):
        result = nn.functional.linear(x, self.weight, bias=self.bias)

        result += (x @ self.lora_a @ self.lora_b) * self.scaling

        return result


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int,
                 r: int, alpha: int = None):
        super().__init__()
        if alpha is None:
            alpha = r

        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim)))
        self.weight.requires_grad = False

        self.scaling = alpha / r
        self.lora_a = nn.Parameter(torch.empty((num_embeddings, r)))
        self.lora_b = nn.Parameter(torch.empty((r, embedding_dim)))

        with torch.no_grad():
            nn.init.normal_(self.lora_a)
            nn.init.zeros_(self.lora_b)

    def forward(self, x: torch.Tensor):
        result = nn.functional.embedding(x, self.weight)
        result += (nn.functional.embedding(x, self.lora_a) @ self.lora_b) * self.scaling

        return result
