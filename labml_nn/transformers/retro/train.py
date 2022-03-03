import torch
from torch import nn
from torch.utils.data import DataLoader

from labml import monit, lab, tracker
from labml_helpers.datasets.text import TextFileDataset
from labml_nn.optimizers.noam import Noam
from labml_nn.transformers.retro import model as retro
from labml_nn.transformers.retro.dataset import Dataset
from labml_nn.transformers.retro.model import RetroModel, Encoder


class Sampler:
    def __init__(self, tokenizer, model):
        pass

    def sample(self, prompt: str, sample_len):
        neighbors = []
        for i in range(sample_len):
            pass
            # retrieve neighbors if there are new chunks
            # append to neighbors

            # evaluate model
            # get the next token


class Trainer:
    epochs: int = 200
    batch_size: int = 1

    learning_rate = 0.0002
    adam_betas = (0.5, 0.999)
    decay_start = 100

    def __init__(self, device: torch.device, model: retro.RetroModel, dataloader: DataLoader,
                 optimizer: torch.optim.Adam):
        self.optimizer = optimizer
        self.device = device
        self.dataloader = dataloader
        self.model = model
        self.loss_func = nn.CrossEntropyLoss()

    def __call__(self):
        for i, (src, tgt, neighbors) in monit.enum('Train', self.dataloader):
            # Move images to the device
            src, tgt, neighbors = src.to(self.device), tgt.to(self.device), neighbors.to(self.device)

            res = self.model(src, neighbors)
            loss = self.loss_func(res.view(-1, res.shape[-1]), tgt.view(-1))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Save training statistics and increment the global step counter
            tracker.save({'loss.train': loss})
            tracker.add_global_step(len(src))


def train():
    device = torch.device('cuda:0')
    tds = TextFileDataset(
        lab.get_data_path() / 'tiny_shakespeare.txt',
        list,
        url='https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt')

    train_dataset = Dataset(lab.get_data_path() / 'retro_train_dataset.json', tds)
    train_dl = DataLoader(train_dataset, batch_size=4)

    chunk_length = 64
    d_model = 256
    d_ff = 1024
    n_heads = 8
    d_k = 32
    model = RetroModel(tds.n_tokens, d_model, 6, {2, 5}, chunk_length, n_heads, d_k, d_ff,
                       encoder=Encoder(chunk_length, 3, {2}, d_model, n_heads, d_k, d_ff))

    model = model.to(device)

    optimizer = Noam(model.parameters(), lr=1., d_model=d_model, warmup=2_000)

    trainer = Trainer(device, model, train_dl, optimizer)

    for epoch in monit.loop(16):
        trainer()


if __name__ == '__main__':
    train()
