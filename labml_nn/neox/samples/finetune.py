"""
---
title: Fine Tune GPT-NeoX
summary: >
    Fine tune GPT-NeoX biases with Fairscale pipeline parallel module
---

# Fine Tune GPT-NeoX

This shows how to fine tune GPT-NeoX with pipeline parallelism.
"""

import fairscale
import torch
import torch.nn as nn
import torch.utils.data
import torch.utils.data
import typing
from torch.utils.data import DataLoader, RandomSampler

from labml import experiment, monit, tracker, lab
from labml.configs import option
from labml.logger import inspect
from labml_nn.neox.utils.text_dataset import get_training_data
from labml_nn.neox.utils.finetune import FineTuneBiases
from labml_nn.neox.model import LayerGenerator, NeoXModule
from labml_nn.neox.utils import balance_layers_simple
from labml_nn.neox.utils.trainer import PipelineParallelTrainerConf


@option(PipelineParallelTrainerConf.layers, 'PipelineBiases')
def neox_layers(c: PipelineParallelTrainerConf):
    """
    ### Load GPT-NeoX layers
    """
    return list(LayerGenerator(is_clone_layers=c.is_clone_layers,
                               filter_layers=c.filter_layers,
                               dtype=c.dtype,
                               ).load())


@option(PipelineParallelTrainerConf.fine_tuner, 'PipelineBiases')
def fine_tune_biases(c: PipelineParallelTrainerConf):
    """
    ### Create fine tuner for biases
    """

    fine_tuner = FineTuneBiases(typing.cast(typing.List[NeoXModule], c.layers))
    # Mark biases as trainable
    fine_tuner.set_trainable_params()

    #
    return fine_tuner


@option(PipelineParallelTrainerConf.model, 'PipelineBiases')
def pipe_model(c: PipelineParallelTrainerConf):
    """
    ### Create pipeline parallel model
    """

    if c.is_checkpointing:
        raise NotImplementedError()
    else:
        layers = c.layers

    # Make sure the finetuner is initialized
    _ = c.fine_tuner

    # Create the Pipe module
    with monit.section('Pipe'):
        # Get the layer distribution across GPUs
        balance = balance_layers_simple(len(layers), c.n_gpus)
        inspect(balance=balance)
        # Devices for each GPU
        devices = [torch.device(f'cuda:{i}') for i in range(c.n_gpus)]
        # Create Fairscale Pipe module
        pipe_model = fairscale.nn.Pipe(nn.Sequential(*layers),
                                       balance=balance,
                                       devices=devices,
                                       chunks=c.chunks)

    #
    return pipe_model


@option(PipelineParallelTrainerConf.train_loader)
def tiny_shakespeare(c: PipelineParallelTrainerConf):
    """
    #### Tiny Shakespeare dataset
    """
    dataset = get_training_data(c.max_seq_len)

    return DataLoader(dataset,
                      batch_size=c.batch_size,
                      sampler=RandomSampler(dataset, replacement=True))


def main():
    # Create experiment
    experiment.create(name='pipe_neox_biases',
                      writers={'screen', 'web_api'})

    # Initialize configs
    conf = PipelineParallelTrainerConf()
    experiment.configs(conf, {
        'learning_rate': 3e-4,
        'is_checkpointing': False,
        'max_seq_len': 128,
        'batch_size': 64,
        'chunks': 8,
    })

    # Start the experiment
    with experiment.start():
        # Initialize the model. Do this before the loop for cleaner logs.
        _ = conf.model

        # Train
        for epoch in monit.loop(conf.epochs):
            conf.train_epoch()
            tracker.new_line()
            torch.save(conf.fine_tuner.state_dict(), str(lab.get_data_path() / 'fine_tune.pt'))


#
if __name__ == '__main__':
    main()
