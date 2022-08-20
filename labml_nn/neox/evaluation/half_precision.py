"""
---
title: Evaluate GPT-NeoX using LLM.int8() quantization on test suite
summary: >
     Evaluate GPT-NeoX using LLM.int8() quantization on test suite
---

#  Evaluate GPT-NeoX using LLM.int8() quantization on test suite

This code evaluate [GPT-NeoX](../index.html) using, on a suite of tasks.
"""

import torch
from torch import nn

from labml_nn.neox.evaluation import run_eval_harness
from labml_nn.neox.model import LayerGenerator


def main():
    # Device
    device = torch.device('cuda:0')
    # Load layers
    layers = list(LayerGenerator(is_clone_layers=True,
                                 filter_layers=None,
                                 dtype=torch.float16,
                                 device=device
                                 ).load())

    # Create `nn.Sequential` model
    model = nn.Sequential(*layers)

    # Run [evaluation harness](index.html)
    print(run_eval_harness(model, 'half_precision', ['lambada'], device))


#
if __name__ == '__main__':
    main()
