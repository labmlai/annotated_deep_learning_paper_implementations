import datetime

import torch
import torch.distributed

from labml import experiment, monit, tracker
from labml.configs import option
from labml.logger import inspect
from labml_nn.neox.samples.finetune import PipelineParallelTrainerConf


class Configs(PipelineParallelTrainerConf):
    rank: int
    world_size: int


@option(Configs.optimizer, 'fsdp')
def _optimizer(c: Configs):
    from labml_nn.optimizers.adam_fp16 import AdamFP16
    return AdamFP16(c.model.get_trainable_chunk(), lr=c.learning_rate)


@option(Configs.model, 'fsdp')
def _model(c: Configs):
    from labml_nn.scaling.zero3 import Zero3Layer, Zero3Sequential

    modules = []
    for m in monit.iterate('Zero3', c.layers):
        modules.append(Zero3Layer(m.to(c.device),
                                  c.rank, c.world_size, c.device, c.dtype))

    model = Zero3Sequential(modules)

    return model


def main(rank, world_size, init_method: str = 'tcp://localhost:23456'):
    # Create experiment
    with monit.section('Distributed'):
        torch.distributed.init_process_group('nccl',
                                             timeout=datetime.timedelta(seconds=30),
                                             init_method=init_method,
                                             rank=rank,
                                             world_size=world_size)

    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)

    experiment.create(name='fsdp_neox', writers={'screen'})
    experiment.distributed(rank, world_size)
    # Create configurations
    conf = Configs()

    # Load configurations
    experiment.configs(conf, {
        'model': 'fsdp',
        'optimizer': 'fsdp',

        'device': device,
        'rank': rank,
        'world_size': world_size,

        'learning_rate': 3e-4,
        'max_seq_len': 128,
        'batch_size': 16,
    })

    with experiment.start():
        _ = conf.model

        for epoch in monit.loop(conf.epochs):
            conf.train_epoch()
            tracker.new_line()


if __name__ == '__main__':
    inspect([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
    inspect(
        n_gpus=torch.cuda.device_count(),
        mpi=torch.distributed.is_mpi_available(),
        nccl=torch.distributed.is_nccl_available(),
    )

    n_gpu = torch.cuda.device_count()

    torch.multiprocessing.spawn(main, args=(n_gpu,), nprocs=n_gpu, join=True)
