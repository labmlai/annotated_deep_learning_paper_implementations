from typing import List, Dict

import torch
from torch import nn

from labml_nn.neox.model import TransformerLayer, NeoXModule


class FineTuner:
    def __init__(self, layers: List[NeoXModule]):
        self.layers = layers

    def get_trainable_params(self) -> Dict[str, nn.Parameter]:
        params = {}
        for i, layer in enumerate(self.layers):
            params.update(self.get_layer_trainable_params(layer, prefix=f'layer_{i :02d}'))

        return params

    def get_layer_trainable_params(self, layer: NeoXModule, prefix: str) -> Dict[str, nn.Parameter]:
        raise NotImplementedError

    def set_trainable_params(self):
        for layer in self.layers:
            # Set `requires_grad` to `False` for the entire layer.
            layer.requires_grad_(False)
            #
            for p in self.get_trainable_params().values():
                p.requires_grad_(True)

    def state_dict(self):
        return {n: p.data.cpu() for n, p in self.get_trainable_params().items()}

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        params = self.get_trainable_params()
        for n, p in params.items():
            p.data[:] = state_dict[n].to(p.data.device)

        for n in state_dict.keys():
            assert n in params, n


class FineTuneBiases(FineTuner):
    def get_layer_trainable_params(self, layer: NeoXModule, prefix: str) -> Dict[str, nn.Parameter]:
        params = {}

        if isinstance(layer, TransformerLayer):
            # No need to train the mlp bias because we are adding it with attention output
            params[f'{prefix}.attention.output.bias'] = layer.attention.output.bias
            params[f'{prefix}.attention.qkv_lin.bias'] = layer.attention.qkv_lin.bias
            params[f'{prefix}.ffn.dense_h_h4.bias'] = layer.ffn.dense_h_h4.bias
        else:
            pass

        return params
