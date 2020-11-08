from typing import Callable

import torch
import torch.nn as nn
from torchtext.data.utils import get_tokenizer

from labml import lab, experiment, monit, tracker, logger
from labml.configs import option
from labml.logger import Text
from labml.utils.pytorch import get_modules
from labml_helpers.datasets.text import TextDataset, SequentialDataLoader, TextFileDataset
from labml_helpers.device import DeviceConfigs
from labml_helpers.module import Module
from labml_helpers.optimizer import OptimizerConfigs
from labml_helpers.train_valid import TrainValidConfigs, BatchStep, MODE_STATE
from labml_nn.transformers import Encoder, Generator, TransformerConfigs
from labml_nn.transformers.utils import subsequent_mask


class AutoregressiveModel(Module):
    def __init__(self, src_embed: Module, encoder: Encoder, generator: Generator, *,
                 is_save_ff_input: bool = False):
        super().__init__()
        self.src_mask = None
        self.src_embed = src_embed
        self.encoder = encoder
        self.encoder.layers[-1].is_save_ff_input = is_save_ff_input
        self.generator = generator

    @property
    def ff_input(self) -> torch.Tensor:
        return self.encoder.layers[-1].ff_input

    def __call__(self, src: torch.Tensor):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = subsequent_mask(len(src)).to(device)
            self.src_mask = mask
        res = self.encoder(self.src_embed(src), self.src_mask)
        return self.generator(res)


class Configs(TrainValidConfigs):
    transformer: TransformerConfigs
    model: AutoregressiveModel = 'custom_model'
    device: torch.device = DeviceConfigs()
    text: TextDataset
    batch_size: int = 20
    seq_len: int = 32
    n_tokens: int
    tokenizer: Callable = 'character'

    is_save_models = True
    prompt: str = 'early on'
    prompt_separator: str = ''

    is_save_ff_input = False
    optimizer: torch.optim.Adam = 'transformer_optimizer'

    batch_step = 'auto_regression_batch_step'

    def sample(self):
        prompt = self.prompt
        log = [(prompt, Text.subtle)]
        for i in monit.iterate('Sample', 25):
            data = self.text.text_to_i(prompt).unsqueeze(-1)
            data = data.to(self.device)
            output = self.model(data)
            if isinstance(output, tuple):
                output = output[0]
            output = output.argmax(dim=-1).squeeze()
            prompt += self.prompt_separator + self.text.itos[output[-1]]
            log += [(self.prompt_separator + self.text.itos[output[-1]], Text.value)]

        logger.log(log)


class AutoRegressionBatchStep(BatchStep):
    def process(self, batch: any, state: any):
        device = self.model.device
        data, target = batch
        data, target = data.to(device), target.to(device)
        stats = {
            'samples': data.shape[0] * data.shape[1]
        }

        output = self.model(data)
        if isinstance(output, tuple):
            output = output[0]

        loss = self.loss_func(output, target)
        if self.accuracy_func is not None:
            stats['correct'] = self.accuracy_func(output, target)

        stats['loss'] = loss.detach().item() * stats['samples']
        tracker.add("loss.", loss)

        if MODE_STATE.is_train:
            loss.backward()

        return stats, None


@option(Configs.batch_step)
def auto_regression_batch_step(c: Configs):
    return AutoRegressionBatchStep(model=c.model,
                                   optimizer=c.optimizer,
                                   loss_func=c.loss_func,
                                   accuracy_func=c.accuracy_func)


class SimpleAccuracyFunc(Module):
    def __call__(self, output: torch.Tensor, target: torch.Tensor) -> int:
        pred = output.argmax(dim=-1)
        return pred.eq(target).sum().item()


@option(Configs.accuracy_func)
def simple_accuracy():
    return SimpleAccuracyFunc()


@option(Configs.optimizer)
def transformer_optimizer(c: Configs):
    optimizer = OptimizerConfigs()
    optimizer.parameters = c.model.parameters()
    optimizer.d_model = c.transformer.d_model
    optimizer.optimizer = 'Noam'

    return optimizer


class CrossEntropyLoss(Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

    def __call__(self, outputs, targets):
        return self.loss(outputs.view(-1, outputs.shape[-1]), targets.view(-1))


@option(Configs.loss_func)
def _loss_func():
    return CrossEntropyLoss()


@option(Configs.n_tokens)
def _n_tokens(c: Configs):
    return c.text.n_tokens


@option(Configs.model)
def custom_model(c: Configs):
    m = AutoregressiveModel(src_embed=c.transformer.src_embed,
                            encoder=c.transformer.encoder,
                            generator=c.transformer.generator,
                            is_save_ff_input=c.is_save_ff_input)
    return m.to(c.device)


@option(Configs.tokenizer)
def basic_english():
    return get_tokenizer('basic_english')


def character_tokenizer(x: str):
    return list(x)


@option(Configs.tokenizer)
def character():
    return character_tokenizer


@option(Configs.text)
def tiny_shakespeare(c: Configs):
    dataset = TextFileDataset(
        lab.get_data_path() / 'tiny_shakespeare.txt', c.tokenizer,
        url='https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt')
    return dataset


@option(Configs.train_loader)
def train_loader(c: Configs):
    # May be use a DataLoader but didn't show much of a performance gain
    return SequentialDataLoader(text=c.text.train,
                                dataset=c.text,
                                batch_size=c.batch_size,
                                seq_len=c.seq_len)


@option(Configs.valid_loader)
def train_loader(c: Configs):
    return SequentialDataLoader(text=c.text.valid,
                                dataset=c.text,
                                batch_size=c.batch_size,
                                seq_len=c.seq_len)


@option(Configs.transformer)
def transformer_c(c: Configs):
    tc = TransformerConfigs()
    tc.n_src_vocab = c.n_tokens
    tc.n_tgt_vocab = c.n_tokens

    return tc


def main():
    conf = Configs()
    experiment.create(name="knn_lm", comment='', writers={'tensorboard', 'sqlite'})
    experiment.configs(conf,
                       {'tokenizer': 'character',
                        'prompt_separator': '',
                        'prompt': 'It is ',
                        'seq_len': 1024,
                        'epochs': 128,
                        'batch_size': 6,
                        'inner_iterations': 10,

                        'transformer.d_model': 256,
                        'transformer.d_ff': 1024,
                        'transformer.n_heads': 8,
                        'transformer.n_layers': 6}, 'run')
    experiment.add_pytorch_models(get_modules(conf))

    with experiment.start():
        conf.run()


if __name__ == '__main__':
    main()
