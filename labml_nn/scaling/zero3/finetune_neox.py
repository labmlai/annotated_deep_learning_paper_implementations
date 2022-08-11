"""
---
title: Finetune GPT-NeoX with Zero3 memory optimizer
summary: >
    This script trains the bias parameters of the GPT-NeoX on multiple devices with Zero-DP Memory Optimization.
---

#  Finetune [GPT-NeoX](../../neox/index.html) with [Zero3 memory optimizer](index.html)

This script trains the bias parameters of the [GPT-NeoX model](../../neox/model.html)
 on multiple devices with Zero-DP Memory Optimization.
"""

import datetime

import torch
import torch.distributed

from labml import experiment, monit, tracker
from labml.configs import option
from labml.logger import inspect
from labml_nn.neox.samples.finetune import PipelineParallelTrainerConf


# Use the [Pipeline Parallel Trainer configurations](../../neox/samples/finetune.html) and adapt it for
# Zero3 memory optimizer.
class Configs(PipelineParallelTrainerConf):
    rank: int
    world_size: int


@option(Configs.optimizer, 'Zero3Adam')
def _optimizer(c: Configs):
    """
    #### Set the optimizers for the model

    Note that we pass the sharded parameters from `get_trainable_chunk`.
    """
    from labml_nn.optimizers.adam_fp16 import AdamFP16
    return AdamFP16(c.model.get_trainable_chunk(), lr=c.learning_rate)


@option(Configs.model, 'Zero3')
def _model(c: Configs):
    """
    #### Create the model with Zero3 memory optimizer
    """
    from labml_nn.scaling.zero3 import Zero3Layer, Zero3Sequential

    # To make sure the fine tuner sets the trainable parameters
    _ = c.fine_tuner

    # Wrap the layers with `Zero3Layer`
    modules = []
    for m in monit.iterate('Zero3', c.layers):
        modules.append(Zero3Layer(m.to(c.device),
                                  c.rank, c.world_size, c.device, c.dtype))

    # Create a sequential model
    model = Zero3Sequential(modules)

    #
    return model


def main(rank: int, world_size: int, init_method: str = 'tcp://localhost:23456'):
    """
    #### Run the training on the node with rank `rank`.
    """
    # Initialize PyTorch distributed process group
    with monit.section('Distributed'):
        torch.distributed.init_process_group('nccl',
                                             timeout=datetime.timedelta(seconds=30),
                                             init_method=init_method,
                                             rank=rank,
                                             world_size=world_size)

    # Set current device
    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)

    # Create the experiment
    experiment.create(name='zero3_neox', writers={'screen', 'labml'})
    experiment.distributed(rank, world_size)

    # Create configurations
    conf = Configs()

    # Load configurations
    experiment.configs(conf, {
        'model': 'Zero3',
        'optimizer': 'Zero3Adam',

        'device': device,
        'rank': rank,
        'world_size': world_size,

        'learning_rate': 3e-4,
        'max_seq_len': 128,
        'batch_size': 16,
    })

    # Start the experiment
    with experiment.start():
        # Initialize the model. Do this before the loop for cleaner logs.
        _ = conf.model

        # Train the model
        for epoch in monit.loop(conf.epochs):
            conf.train_epoch()
            tracker.new_line()


#
if __name__ == '__main__':
    # Log the machine configurations
    inspect([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
    inspect(
        n_gpus=torch.cuda.device_count(),
        mpi=torch.distributed.is_mpi_available(),
        nccl=torch.distributed.is_nccl_available(),
    )

    n_gpu = torch.cuda.device_count()

    # Start a process for each GPU. You will need a separate launcher if you are using multiple computers.
    torch.multiprocessing.spawn(main, args=(n_gpu,), nprocs=n_gpu, join=True)
