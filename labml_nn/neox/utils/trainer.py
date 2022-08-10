from typing import Optional, Set, List

import torch.nn as nn
import torch.optim
import torch.utils.data
from torch.cuda import amp
from torch.cuda.amp import GradScaler

from labml import monit, tracker
from labml.configs import BaseConfigs, option
from labml_nn.neox.utils.finetune import FineTuner


def get_trainable_params(model: nn.Module):
    """
    ### Get trainable parameters

    :param model: is the model to train
    :return: a list of parameters for training
    """

    # Get all parameters
    params = list(model.parameters())
    # Filter parameters that require gradients
    trainable_params = [p for p in params if p.requires_grad]

    #
    return trainable_params


class TrainerConf(BaseConfigs):
    model: nn.Module
    layers: List[nn.Module]
    optimizer: torch.optim.Optimizer = 'Adam'
    train_loader: torch.utils.data.DataLoader
    valid_loader: Optional[torch.utils.data.DataLoader] = None,
    device: torch.device = torch.device('cuda:0')
    scaler: Optional[GradScaler] = 'Default'
    is_amp: bool = True
    dtype: torch.dtype = torch.float16

    is_clone_layers: bool = True

    loss_func: nn.Module = nn.CrossEntropyLoss()
    checkpoints_per_epoch: int = 0
    samples_per_epoch: int = 0

    grad_norm: Optional[float] = 1.0
    learning_rate: float = 3e-4
    max_seq_len: int = 1024
    batch_size: int = 64
    epochs: int = 16

    n_gpus: int = torch.cuda.device_count()

    filter_layers: Optional[Set] = None

    def get_loss(self, sample, dataset_split: str):
        """
        :param dataset_split: train/valid
        :param sample: is the sample
        :return: the loss, output and the target
        """
        data, target = sample

        # Forward pass
        with monit.section('Forward pass'):
            output = self.model(data.to(self.device))
        # Move targets to the same device as output
        target = target.to(output.device)
        # Calculate loss
        loss = self.loss_func(output.view(target.numel(), -1), target.view(-1))

        return loss, output, target

    def train(self):
        for epoch in monit.loop(self.epochs):
            self.train_epoch()
            tracker.new_line()

    def sample(self, idx):
        pass

    def save_checkpoint(self, idx):
        pass

    def get_iterators(self):
        # Iterate through the batches
        iterators = [('train', self.train_loader)]
        if self.valid_loader is not None:
            iterators.append(('valid', self.valid_loader))

        if self.samples_per_epoch > 0:
            iterators.append((self.sample, [i for i in range(self.samples_per_epoch)]))

        if self.checkpoints_per_epoch > 0:
            iterators.append((self.save_checkpoint, [i for i in range(self.checkpoints_per_epoch)]))

        return iterators

    def train_epoch(self):
        # Set model for train
        self.model.train()

        iterators = self.get_iterators()
        for split_name, sample in monit.mix(1024, *iterators):
            if split_name == 'train':
                # Set gradients to zero
                self.optimizer.zero_grad()
                tracker.add_global_step()

            with torch.set_grad_enabled(split_name == 'train'):
                if self.is_amp:
                    # Forward pass
                    with amp.autocast():
                        loss, output, target = self.get_loss(sample, split_name)
                else:
                    loss, output, target = self.get_loss(sample, split_name)

                # Get predictions
                pred = output.argmax(dim=-1)
                # Calculate accuracy
                accuracy = pred.eq(target).sum().item() / (target != -100).sum()

                tracker.add({f'loss.{split_name}': loss, f'acc.{split_name}': accuracy * 100})

            if split_name == 'train':
                if self.scaler is not None:
                    # Backward pass
                    loss = self.scaler.scale(loss)
                    # tracker.add({'loss.scaled': loss})

                with monit.section('Backward pass'):
                    loss.backward()

                # Optimize
                with monit.section('Optimize'):
                    if self.scaler is None:
                        self.optimizer.step()
                    else:
                        self.scaler.unscale_(self.optimizer)
                        if self.grad_norm is not None:
                            torch.nn.utils.clip_grad_norm_(get_trainable_params(self.model), self.grad_norm)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()

            tracker.save()


@option(TrainerConf.optimizer, 'Adam')
def adam_optimizer(c: TrainerConf):
    if c.dtype == torch.float32:
        return torch.optim.Adam(get_trainable_params(c.model), lr=c.learning_rate)
    elif c.dtype == torch.float16:
        from labml_nn.optimizers.adam_fp16 import AdamFP16
        return AdamFP16(get_trainable_params(c.model), lr=c.learning_rate)
    else:
        raise NotImplementedError()


@option(TrainerConf.optimizer, 'SGD')
def sgd_optimizer(c: TrainerConf):
    return torch.optim.SGD(get_trainable_params(c.model), lr=c.learning_rate)


@option(TrainerConf.scaler, 'Default')
def grad_scaler(c: TrainerConf):
    if not c.is_amp:
        return None

    if c.dtype == torch.float16:
        from labml_nn.optimizers.adam_fp16 import GradScalerFP16
        return GradScalerFP16()
    else:
        return GradScaler()


class PipelineParallelTrainerConf(TrainerConf):
    is_checkpointing: bool = False
    chunks: int

    fine_tuner: FineTuner
