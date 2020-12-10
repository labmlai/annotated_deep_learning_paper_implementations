"""
---
title: Test performance of Adam implementations
summary: This experiment compares performance of Adam implementations.
---

# Performance testing Adam

```
TorchAdam warmup...[DONE]	222.59ms
TorchAdam...[DONE]	1,356.01ms
MyAdam warmup...[DONE]	119.15ms
MyAdam...[DONE]	1,192.89ms
```

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ngowaAsADj8VdZfBifu_6L6rtjGoEeoR?usp=sharing)
"""

import torch
import torch.nn as nn
from labml_helpers.device import DeviceInfo
from torch.optim import Adam as TorchAdam

from labml import monit
from labml_nn.optimizers.adam import Adam as MyAdam
from labml_nn.optimizers.mnist_experiment import Model


def test():
    device_info = DeviceInfo(use_cuda=True, cuda_device=0)
    print(device_info)
    inp = torch.randn((64, 1, 28, 28), device=device_info.device)
    target = torch.ones(64, dtype=torch.long, device=device_info.device)
    loss_func = nn.CrossEntropyLoss()
    model = Model().to(device_info.device)
    my_adam = MyAdam(model.parameters())
    torch_adam = TorchAdam(model.parameters())
    loss = loss_func(model(inp), target)
    loss.backward()
    with monit.section('MyAdam warmup'):
        for i in range(100):
            my_adam.step()
    with monit.section('MyAdam'):
        for i in range(1000):
            my_adam.step()
    with monit.section('TorchAdam warmup'):
        for i in range(100):
            torch_adam.step()
    with monit.section('TorchAdam'):
        for i in range(1000):
            torch_adam.step()


if __name__ == '__main__':
    test()
