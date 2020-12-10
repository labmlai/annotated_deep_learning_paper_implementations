"""
---
title: Configurable optimizer module
summary: This implements a configurable module for optimizers.
---
"""

from typing import Tuple

import torch

from labml.configs import BaseConfigs, option, meta_config
from labml_nn.optimizers import WeightDecay


class OptimizerConfigs(BaseConfigs):
    optimizer: torch.optim.Adam

    weight_decay_obj: WeightDecay
    weight_decouple: bool = True
    weight_decay: float = 0.0
    weight_decay_absolute: bool = False
    optimized_adam_update: bool = True

    parameters: any

    learning_rate: float = 0.01
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-08

    momentum: float = 0.5
    amsgrad: bool = False
    warmup: int = 0
    degenerated_to_sgd: bool = True
    rectify: bool = True
    d_model: int
    step_factor: int = 1024

    def __init__(self):
        super().__init__(_primary='optimizer')


meta_config(OptimizerConfigs.parameters)


@option(OptimizerConfigs.weight_decay_obj, 'L2')
def _weight_decay(c: OptimizerConfigs):
    return WeightDecay(c.weight_decay, c.weight_decouple, c.weight_decay_absolute)


@option(OptimizerConfigs.optimizer, 'SGD')
def _sgd_optimizer(c: OptimizerConfigs):
    return torch.optim.SGD(c.parameters, c.learning_rate, c.momentum)


@option(OptimizerConfigs.optimizer, 'Adam')
def _adam_optimizer(c: OptimizerConfigs):
    if c.amsgrad:
        from labml_nn.optimizers.amsgrad import AMSGrad
        return AMSGrad(c.parameters,
                       lr=c.learning_rate, betas=c.betas, eps=c.eps,
                       optimized_update=c.optimized_adam_update,
                       weight_decay=c.weight_decay_obj, amsgrad=c.amsgrad)
    else:
        from labml_nn.optimizers.adam import Adam
        return Adam(c.parameters,
                    lr=c.learning_rate, betas=c.betas, eps=c.eps,
                    optimized_update=c.optimized_adam_update,
                    weight_decay=c.weight_decay_obj)


@option(OptimizerConfigs.optimizer, 'AdamW')
def _adam_warmup_optimizer(c: OptimizerConfigs):
    from labml_nn.optimizers.adam_warmup import AdamWarmup
    return AdamWarmup(c.parameters,
                      lr=c.learning_rate, betas=c.betas, eps=c.eps,
                      weight_decay=c.weight_decay_obj, amsgrad=c.amsgrad, warmup=c.warmup)


@option(OptimizerConfigs.optimizer, 'RAdam')
def _radam_optimizer(c: OptimizerConfigs):
    from labml_nn.optimizers.radam import RAdam
    return RAdam(c.parameters,
                 lr=c.learning_rate, betas=c.betas, eps=c.eps,
                 weight_decay=c.weight_decay_obj, amsgrad=c.amsgrad,
                 degenerated_to_sgd=c.degenerated_to_sgd)


@option(OptimizerConfigs.optimizer, 'AdaBelief')
def _ada_belief_optimizer(c: OptimizerConfigs):
    from labml_nn.optimizers.ada_belief import AdaBelief
    return AdaBelief(c.parameters,
                     lr=c.learning_rate, betas=c.betas, eps=c.eps,
                     weight_decay=c.weight_decay_obj, amsgrad=c.amsgrad,
                     degenerated_to_sgd=c.degenerated_to_sgd,
                     rectify=c.rectify)


@option(OptimizerConfigs.optimizer, 'Noam')
def _noam_optimizer(c: OptimizerConfigs):
    from labml_nn.optimizers.noam import Noam
    return Noam(c.parameters,
                lr=c.learning_rate, betas=c.betas, eps=c.eps,
                weight_decay=c.weight_decay_obj, amsgrad=c.amsgrad, warmup=c.warmup,
                d_model=c.d_model)
