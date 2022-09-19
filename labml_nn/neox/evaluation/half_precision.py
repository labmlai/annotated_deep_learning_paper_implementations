"""
---
title: Evaluate GPT-NeoX using LLM.int8() quantization on test suite
summary: >
     Evaluate GPT-NeoX using LLM.int8() quantization on test suite
---

#  Evaluate GPT-NeoX using LLM.int8() quantization on test suite

This code evaluate [GPT-NeoX](../index.html) using, on a suite of tasks.
"""
import argparse

import torch
from torch import nn

from labml_nn.neox.evaluation import run_eval_harness
from labml_nn.neox.model import LayerGenerator


def main():
    # Argument parser
    parser = argparse.ArgumentParser()

    parser.add_argument("--flash", action='store_true', help="whether to use Flash Attention")

    opt = parser.parse_args()

    # Device
    device = torch.device('cuda:0')
    # Load layers
    layers = list(LayerGenerator(is_clone_layers=True,
                                 filter_layers=None,
                                 dtype=torch.float16,
                                 device=device,
                                 is_flash_attention=opt.flash,
                                 ).load())

    # Create `nn.Sequential` model
    model = nn.Sequential(*layers)

    # Run [evaluation harness](index.html)
    print(run_eval_harness(model, 'half_precision', ['lambada'], device))


#
if __name__ == '__main__':
    main()
