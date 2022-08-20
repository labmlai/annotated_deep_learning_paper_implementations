"""
---
title: Evaluate GPT-NeoX using LLM.int8() quantization on test suite
summary: >
     Evaluate GPT-NeoX using LLM.int8() quantization on test suite
---

#  Evaluate GPT-NeoX using LLM.int8() quantization on test suite

This code evaluate [GPT-NeoX](../index.html) using [LLM.int8() quantization](../utils/llm_int8.html),
on a suite of tasks.
"""

import torch
from torch import nn

from labml import monit
from labml_nn.neox.evaluation import run_eval_harness
from labml_nn.neox.model import LayerGenerator


def main():
    # Device
    device = torch.device('cuda:0')

    # Load layers in float16 into CPU. We convert the layers to int8 later, because doing that
    # on the fly after loading layers to GPU causes CUDA memory fragmentation
    # (about 3GB memory can get lost due to fragmentation).
    layer_generator = LayerGenerator(is_clone_layers=True,
                                     dtype=torch.float16,
                                     device=torch.device('cpu'),
                                     )
    # Load layers
    layers = list(layer_generator.load())

    # This reduces CUDA memory fragmentation
    for layer in monit.iterate('Convert to int8', layers, is_children_silent=True):
        layer_generator.post_load_prepare(layer,
                                          device=device,
                                          is_llm_int8=True,
                                          llm_int8_threshold=6.0,
                                          )
        layer.to(device)

    # Create `nn.Sequential` model
    model = nn.Sequential(*layers)

    # Run [evaluation harness](index.html)
    print(run_eval_harness(model, 'half_precision', [], device))


#
if __name__ == '__main__':
    main()
