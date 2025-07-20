import signal
import typing
from typing import Dict, List, Callable
from typing import Optional, Tuple, Any, Collection

import torch.optim
import torch.optim
import torch.utils.data
import torch.utils.data
from labml import tracker, logger, monit
from labml.configs import BaseConfigs, meta_config, option
from labml.internal.monitor import Loop
from labml.logger import Text
from torch import nn
from .device import DeviceConfigs
from .metrics import StateModule


class TrainingLoopIterator(Collection):
    def __init__(self, start: int, total: int, step: Optional[int]):
        self.step = step
        self.total = total
        self.start = start
        self.i = None

    def __iter__(self):
        self.i = None
        return self

    def __next__(self):
        if self.step is not None:
            if self.i is None:
                self.i = self.start
            else:
                self.i += self.step
        else:
            if self.i is None:
                self.i = 0
            else:
                self.i += 1

        if self.i >= self.total:
            raise StopIteration()

        if self.step is None:
            return tracker.get_global_step()
        else:
            return self.i

    def __len__(self) -> int:
        if self.step is not None:
            return (self.total - self.start) // self.step
        else:
            return self.total

    def __contains__(self, x: object) -> bool:
        return False


class TrainingLoop:
    _iter: Optional[TrainingLoopIterator]
    __loop: Loop
    __signal_received: Optional[Tuple[Any, Any]]

    def __init__(self, *,
                 loop_count: int,
                 loop_step: Optional[int],
                 log_new_line_interval: int,
                 log_write_interval: int,
                 is_loop_on_interrupt: bool):
        self.__loop_count = loop_count
        self.__loop_step = loop_step
        self.__log_new_line_interval = log_new_line_interval
        self.__log_write_interval = log_write_interval
        self.__last_write_step = 0
        self.__last_new_line_step = 0
        self.__last_save_step = 0
        self.__signal_received = None
        self.__is_loop_on_interrupt = is_loop_on_interrupt
        self._iter = None

    def __iter__(self):
        self._iter = TrainingLoopIterator(tracker.get_global_step(),
                                          self.__loop_count,
                                          self.__loop_step)

        self.__loop = monit.loop(typing.cast(Collection, self._iter))

        iter(self.__loop)
        try:
            self.old_handler = signal.signal(signal.SIGINT, self.__handler)
        except ValueError:
            pass
        return self

    @property
    def idx(self):
        if not self._iter:
            return 0
        if not self._iter.i:
            return 0
        if self.__loop_step is None:
            return self._iter.i
        return self._iter.i / self.__loop_step

    def __finish(self):
        try:
            signal.signal(signal.SIGINT, self.old_handler)
        except ValueError:
            pass
        tracker.save()
        tracker.new_line()

    def __next__(self):
        if self.__signal_received is not None:
            logger.log('\nKilling Loop.', Text.danger)
            monit.finish_loop()
            self.__finish()
            raise StopIteration("SIGINT")

        try:
            global_step = next(self.__loop)
        except StopIteration as e:
            self.__finish()
            raise e

        tracker.set_global_step(global_step)

        if global_step - self.__last_write_step >= self.__log_write_interval:
            tracker.save()
            self.__last_write_step = global_step
        if global_step - self.__last_new_line_step >= self.__log_new_line_interval:
            tracker.new_line()
            self.__last_new_line_step = global_step

        return global_step

    def __handler(self, sig, frame):
        # Pass second interrupt without delaying
        if self.__signal_received is not None:
            logger.log('\nSIGINT received twice. Stopping...', Text.danger)
            self.old_handler(*self.__signal_received)
            return

        if self.__is_loop_on_interrupt:
            # Store the interrupt signal for later
            self.__signal_received = (sig, frame)
            logger.log('\nSIGINT received. Delaying KeyboardInterrupt.', Text.danger)
        else:
            self.__finish()
            logger.log('Killing loop...', Text.danger)
            self.old_handler(sig, frame)

    def __str__(self):
        return "LabTrainingLoop"


class TrainingLoopConfigs(BaseConfigs):
    r"""
    This is a configurable training loop. You can extend this class for your configurations
    if it involves a training loop.

    >>> for step in conf.training_loop:
    >>>     ...

    Arguments:
        loop_count (int): Total number of steps. Defaults to ``10``.
        loop_step (int): Number of steps to increment per iteration. Defaults to ``1``.
        log_new_line_interval (int): The interval (in steps) to print a new line to the screen.
         Defaults to ``1``.
        log_write_interval (int): The interval (in steps) to call :func:`labml.tracker.save`.
         Defaults to ``1``.
        is_loop_on_interrupt (bool): Whether to handle keyboard interrupts and wait until a iteration is complete.
         Defaults to ``False``.
    """
    loop_count: int = 10
    loop_step: int = 1
    log_new_line_interval: int = 1
    log_write_interval: int = 1
    is_loop_on_interrupt: bool = False

    training_loop: TrainingLoop


@option(TrainingLoopConfigs.training_loop)
def _loop_configs(c: TrainingLoopConfigs):
    return TrainingLoop(loop_count=c.loop_count,
                        loop_step=c.loop_step,
                        log_new_line_interval=c.log_new_line_interval,
                        log_write_interval=c.log_write_interval,
                        is_loop_on_interrupt=c.is_loop_on_interrupt)


meta_config(TrainingLoopConfigs.loop_step,
            TrainingLoopConfigs.loop_count,
            TrainingLoopConfigs.log_new_line_interval,
            TrainingLoopConfigs.log_write_interval,
            TrainingLoopConfigs.is_loop_on_interrupt)


class ModeState:
    def __init__(self):
        self._rollback_stack = []

        self.is_train = False
        self.is_optimize = False

    def _enter(self, mode: Dict[str, any]):
        rollback = {}
        for k, v in mode.items():
            if v is None:
                continue
            rollback[k] = getattr(self, k)
            setattr(self, k, v)

        self._rollback_stack.append(rollback)

        return len(self._rollback_stack)

    def _exit(self, n: int):
        assert n == len(self._rollback_stack)

        rollback = self._rollback_stack[-1]
        self._rollback_stack.pop(-1)

        for k, v in rollback.items():
            setattr(self, k, v)

    def update(self, *,
               is_train: Optional[bool] = None,
               is_optimize: Optional[bool] = None):
        return Mode(self,
                    is_train=is_train,
                    is_optimize=is_optimize)


class Mode:
    def __init__(self, mode: ModeState, **kwargs: any):
        self.mode = mode
        self.update = {}
        for k, v in kwargs.items():
            if v is not None:
                self.update[k] = v

        self.idx = -1

    def __enter__(self):
        self.idx = self.mode._enter(self.update)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.mode._exit(self.idx)


class Trainer:
    def __init__(self, *,
                 name: str,
                 mode: ModeState,
                 data_loader: torch.utils.data.DataLoader,
                 inner_iterations: int,
                 state_modules: List[StateModule],
                 is_track_time: bool,
                 step: Callable[[any, 'BatchIndex'], None]):
        self.is_track_time = is_track_time
        self.mode = mode
        self.name = name
        self.step = step
        self.state_modules = state_modules
        self.__iterable = None
        self.__states = [sm.create_state() for sm in self.state_modules]
        self.inner_iterations = inner_iterations
        self.data_loader = data_loader
        self._batch_index = BatchIndex(len(self.data_loader), self.inner_iterations)

    def set_data_loader(self, data_loader: torch.utils.data.DataLoader):
        self.data_loader = data_loader
        self._batch_index = BatchIndex(len(data_loader), self.inner_iterations)
        self.__iterable = None

    def __call__(self):
        for sm, s in zip(self.state_modules, self.__states):
            sm.set_state(s)

        if self.__iterable is None or self._batch_index.completed:
            self.__iterable = iter(self.data_loader)
            self._batch_index.reset(len(self.data_loader), self.inner_iterations)
            for sm in self.state_modules:
                sm.on_epoch_start()
        with torch.set_grad_enabled(self.mode.is_train):
            self.__iterate()

        if self._batch_index.completed:
            for sm in self.state_modules:
                sm.on_epoch_end()

    def __iterate(self):
        with monit.section(self.name, is_partial=True, is_track=self.is_track_time):
            if self._batch_index.idx == 0:
                monit.progress(0)
            while not self._batch_index.iteration_completed:
                batch = next(self.__iterable)

                self.step(batch, self._batch_index)

                self._batch_index.step()
                monit.progress(self._batch_index.epoch_progress)

        self._batch_index.step_inner()


class BatchIndex:
    idx: int
    total: int
    iteration: int
    total_iterations: int

    def __init__(self, total: int, total_iterations: int):
        self.total_iterations = total_iterations
        self.total = total

    def is_interval(self, interval: int):
        if interval <= 0:
            return False
        if self.idx + 1 == self.total:
            return True
        else:
            return (self.idx + 1) % interval == 0

    @property
    def is_last(self):
        return self.idx + 1 == self.total

    @property
    def completed(self):
        return self.iteration >= self.total_iterations

    @property
    def iteration_completed(self):
        # // is important so that the last step happens on the last iteration
        return self.idx >= (self.iteration + 1) * self.total // self.total_iterations

    @property
    def epoch_progress(self):
        return self.idx / self.total

    def step(self):
        self.idx += 1

    def step_inner(self):
        self.iteration += 1

    def reset(self, total: int, total_iterations: int):
        self.total = total
        self.total_iterations = total_iterations
        self.idx = 0
        self.iteration = 0


class TrainValidConfigs(TrainingLoopConfigs):
    r"""
    This is a configurable module that you can extend for experiments that involve a
    training and validation datasets (i.e. most DL experiments).

    Arguments:
        epochs (int): Number of epochs to train on. Defaults to ``10``.
        train_loader (torch.utils.data.DataLoader): Training data loader.
        valid_loader (torch.utils.data.DataLoader): Training data loader.
        inner_iterations (int): Number of times to switch between training and validation
         within an epoch. Defaults to ``1``.

    You can override ``init``, ``step`` functions. There is also a ``sample`` function
    that you can override to generate samples ever time it switches between training and validation.
    """
    state_modules: List[StateModule]

    mode: ModeState

    epochs: int = 10

    trainer: Trainer
    validator: Trainer
    train_loader: torch.utils.data.DataLoader
    valid_loader: torch.utils.data.DataLoader

    loop_count = '_data_loop_count'
    loop_step = None

    inner_iterations: int = 1

    is_track_time: bool = False

    def init(self):
        pass

    def step(self, batch: Any, batch_idx: BatchIndex):
        raise NotImplementedError

    def run_step(self):
        for i in range(self.inner_iterations):
            with tracker.namespace('sample'):
                self.sample()
            with self.mode.update(is_train=True):
                with tracker.namespace('train'):
                    self.trainer()
            if self.validator:
                with tracker.namespace('valid'):
                    self.validator()
            tracker.save()

    def run(self):
        with monit.section("Initialize"):
            self.init()
        _ = self.validator
        _ = self.trainer
        for _ in self.training_loop:
            self.run_step()

    def sample(self):
        pass


@option(TrainValidConfigs.trainer)
def _default_trainer(c: TrainValidConfigs):
    return Trainer(name='Train',
                   mode=c.mode,
                   data_loader=c.train_loader,
                   inner_iterations=c.inner_iterations,
                   state_modules=c.state_modules,
                   is_track_time=c.is_track_time,
                   step=c.step)


@option(TrainValidConfigs.validator)
def _default_validator(c: TrainValidConfigs):
    return Trainer(name='Valid',
                   mode=c.mode,
                   data_loader=c.valid_loader,
                   inner_iterations=c.inner_iterations,
                   state_modules=c.state_modules,
                   is_track_time=c.is_track_time,
                   step=c.step)


@option(TrainValidConfigs.loop_count)
def _data_loop_count(c: TrainValidConfigs):
    return c.epochs


class SimpleTrainValidConfigs(TrainValidConfigs):
    r"""
    This is a configurable module that works for many standard DL experiments.

    Arguments:
        model: A PyTorch model.
        optimizer: A PyTorch optimizer to update model.
        device: The device to train the model on. This defaults to a configurable device
        loss_function: A function to calculate the loss. This should accept ``model_output, target`` as
         arguments.
        update_batches (int): Number of batches to accumulate before taking an optimizer step.
         Defaults to ``1``.
        log_save_batches (int): How often to call :func:`labml.tracker.save`.
    """
    optimizer: torch.optim.Adam
    model: nn.Module
    device: torch.device = DeviceConfigs()

    loss_func: nn.Module

    update_batches: int = 1
    log_save_batches: int = 1

    state_modules: List[StateModule] = []

    def init(self):
        pass

    def step(self, batch: Any, batch_idx: BatchIndex):
        self.model.train(self.mode.is_train)
        data, target = batch[0].to(self.device), batch[1].to(self.device)

        if self.mode.is_train:
            tracker.add_global_step(len(data))

        with monit.section("model"):
            output = self.model(data)

        loss = self.loss_func(output, target)
        tracker.add("loss.", loss)

        if self.mode.is_train:
            with monit.section('backward'):
                loss.backward()

            if batch_idx.is_interval(self.update_batches):
                with monit.section('optimize'):
                    self.optimizer.step()
                self.optimizer.zero_grad()

            if batch_idx.is_interval(self.log_save_batches):
                tracker.save()


meta_config(SimpleTrainValidConfigs.update_batches,
            )


@option(SimpleTrainValidConfigs.optimizer)
def _default_optimizer(c: SimpleTrainValidConfigs):
    from .optimizer import OptimizerConfigs
    opt_conf = OptimizerConfigs()
    opt_conf.parameters = c.model.parameters()
    return opt_conf
