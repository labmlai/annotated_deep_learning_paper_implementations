"""
---
title: CIFAR10 Experiment
summary: >
  This is a reusable trainer for CIFAR10 dataset
---

# CIFAR10 Experiment
"""

from labml_helpers.datasets.cifar10 import CIFAR10Configs as CIFAR10DatasetConfigs
from labml_nn.experiments.mnist import MNISTConfigs


class CIFAR10Configs(CIFAR10DatasetConfigs, MNISTConfigs):
    dataset_name: str = 'CIFAR10'
