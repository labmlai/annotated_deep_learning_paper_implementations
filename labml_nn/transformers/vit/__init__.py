import torch
from torch import nn

from labml_helpers.module import Module
from labml_nn.transformers import TransformerLayer
from labml_nn.utils import clone_module_list


class PatchEmbeddings(Module):
    """
    <a id="PatchEmbeddings">
    ## Embed patches
    </a>
    """

    def __init__(self, d_model: int, patch_size: int, in_channels: int):
        super().__init__()
        self.patch_size = patch_size
        self.linear = nn.Linear(patch_size * patch_size * in_channels, d_model)

    def __call__(self, x: torch.Tensor):
        """
        x has shape `[batch_size, channels, height, width]`
        """
        bs, c, h, w = x.shape
        assert h % self.patch_size == 0 and w % self.patch_size == 0
        h_p = h // self.patch_size
        w_p = w // self.patch_size
        x = x.view(bs, c, h_p, self.patch_size, w_p, self.patch_size)
        x = x.permute(2, 4, 0, 1, 3, 5)
        x = x.reshape(h_p * w_p, bs, -1)
        x = self.linear(x)

        return x


class LearnedPositionalEmbeddings(Module):
    """
    <a id="LearnedPositionalEmbeddings">
    ## Add parameterized positional encodings
    </a>
    """

    def __init__(self, d_model: int, max_len: int = 5_000):
        super().__init__()
        self.positional_encodings = nn.Parameter(torch.zeros(max_len, 1, d_model), requires_grad=True)

    def __call__(self, x: torch.Tensor):
        pe = self.positional_encodings[x.shape[0]]
        return x + pe


class ClassificationHead(Module):
    def __init__(self, d_model: int, n_hidden: int, n_classes: int):
        super().__init__()
        self.ln = nn.LayerNorm([d_model])
        self.linear1 = nn.Linear(d_model, n_hidden)
        self.act = nn.ReLU()
        self.linear2 = nn.Linear(n_hidden, n_classes)

    def __call__(self, x: torch.Tensor):
        x = self.ln(x)
        x = self.act(self.linear1(x))
        x = self.linear2(x)

        return x


class VisionTransformer(Module):
    def __init__(self, transformer_layer: TransformerLayer, n_layers: int,
                 patch_emb: PatchEmbeddings, pos_emb: LearnedPositionalEmbeddings,
                 classification: ClassificationHead):
        super().__init__()
        # Make copies of the transformer layer
        self.classification = classification
        self.pos_emb = pos_emb
        self.patch_emb = patch_emb
        self.transformer_layers = clone_module_list(transformer_layer, n_layers)

        self.cls_token_emb = nn.Parameter(torch.randn(1, 1, transformer_layer.size), requires_grad=True)

    def __call__(self, x):
        x = self.patch_emb(x)
        x = self.pos_emb(x)
        cls_token_emb = self.cls_token_emb.expand(-1, x.shape[1], -1)
        x = torch.cat([cls_token_emb, x])
        for layer in self.transformer_layers:
            x = layer(x=x, mask=None)

        x = x[0]

        x = self.classification(x)

        return x
