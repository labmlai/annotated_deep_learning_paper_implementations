"""
---
title: RETRO training
summary: >
  Training RETRO model with Tiny Shakespeare dataset
---

# RETRO training

This is the training code for
 [RETRO](index.html).

[![View Run](https://img.shields.io/badge/labml-experiment-brightgreen)](https://app.labml.ai/run/3113dd3ea1e711ec85ee295d18534021)
"""

import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler

from labml import monit, lab, tracker, experiment, logger
from labml.logger import Text
from labml_helpers.datasets.text import TextFileDataset
from labml_nn.optimizers.noam import Noam
from labml_nn.transformers.retro import model as retro
from labml_nn.transformers.retro.dataset import Dataset, RetroIndex
from labml_nn.transformers.retro.model import RetroModel, NearestNeighborEncoder


class Sampler:
    """
    ## Sampler

    This class greedily samples from a model.
    """

    def __init__(self, device: torch.device, model: retro.RetroModel, tds: TextFileDataset, chunk_len: int):
        """
        * `device` is the device of the model
        * `model` is the [Retro mode](retro.html)
        * `tds` is the text dataset (used to get neighbor chunks)
        * `chunk_len` is the length of a chunk
        """
        self.chunk_len = chunk_len
        self.tds = tds
        self.model = model
        self.device = device

        # [Retro index](database.html)
        self.index = RetroIndex()

    def retrieve_nearest_neighbours(self, chunk: str):
        """
        ### Retrieve nearest neighbors of a given chunk
        """

        # Retrieve the offsets of the nearest neighbors
        neighbor_offsets = self.index([chunk], None)

        # Get the neighbors (with neighbor length equal to `chunk_len * 2`)
        text = self.tds.train
        neighbors = [text[j: j + self.chunk_len * 2] for j in neighbor_offsets[0]]

        #
        return neighbors

    def sample(self, prompt: str, sample_len: int):
        """
        ### Sample text from the given prompt
        """

        # To store nearest neighbors as strings
        neighbors_str = []

        # Sampled text
        sampled = ''

        # Sample `sample_len` tokens
        for i in range(sample_len):
            # We need to retrieve neighbors,
            # if there are more sampled chunks than we have already retrieved for
            while len(neighbors_str) < len(prompt) // self.chunk_len:
                # Get the last chunk for which we haven't retrieved neighbors
                off = len(neighbors_str) * self.chunk_len
                chunk = prompt[off: off + self.chunk_len]
                # Retrieve nearest neighbors
                neighbors_str.append(self.retrieve_nearest_neighbours(chunk))

            # Tokenize the input
            src = self.tds.text_to_i(prompt)
            # Tokenize the retrieved neighbors
            neighbors = torch.stack([torch.stack([self.tds.text_to_i(n) for n in chunk]) for chunk in neighbors_str])

            # Move them to the same device as the model
            src = src.to(self.device)
            neighbors = neighbors.to(self.device)

            # Get model output
            res = self.model(src[None, :], neighbors[None, :, :, :])

            # Greedily sample the last token
            token = res[0, -1, :].argmax(dim=-1)

            # Add the sampled token text to the prompt and sample text
            prompt += self.tds.itos[token.item()]
            sampled += self.tds.itos[token.item()]

        #
        return sampled


class Trainer:
    """
    ## Retro trainer
    """

    def __init__(self, device: torch.device, model: retro.RetroModel,
                 dataloader: DataLoader, optimizer: torch.optim.Optimizer):
        """
        * `device` is the device of the model
        * `model` is the [Retro mode](retro.html)
        * `dataloader` is the dataloader for the [dataset with pre-retrieved neighbors](dataset.html)
        * `optimizer` is the optimizer
        """
        self.optimizer = optimizer
        self.device = device
        self.dataloader = dataloader
        self.model = model
        self.loss_func = nn.CrossEntropyLoss()

    def __call__(self):
        """
        ### Train the model for an epoch
        """

        # Iterate through training data
        for i, (src, tgt, neighbors) in monit.enum('Train', self.dataloader):
            # Move data to the device
            src, tgt, neighbors = src.to(self.device), tgt.to(self.device), neighbors.to(self.device)

            # Forward pass
            res = self.model(src, neighbors)
            # Calculate loss
            loss = self.loss_func(res.view(-1, res.shape[-1]), tgt.view(-1))

            # Clear the gradients
            self.optimizer.zero_grad()
            # Backward pass
            loss.backward()
            # Optimize the model
            self.optimizer.step()

            # Save training statistics and increment the global step counter
            tracker.save({'loss.train': loss})
            tracker.add_global_step(len(src))


def train():
    """
    ## Create and train a small model
    """

    # Create an experiment
    experiment.create(name='retro_small')

    # GPU device
    device = torch.device('cuda:0')

    # Load Tiny Shakespeare dataset
    tds = TextFileDataset(
        lab.get_data_path() / 'tiny_shakespeare.txt',
        list,
        url='https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt')

    # Load [Retro dataset](dataset.html)
    train_dataset = Dataset(lab.get_data_path() / 'retro_train_dataset.json', tds)

    # Create dataloader
    train_dl = DataLoader(train_dataset,
                          batch_size=4,
                          sampler=RandomSampler(train_dataset, replacement=True))

    # Hyper-parameters
    chunk_len = 16
    d_model = 128
    d_ff = 512
    n_heads = 16
    d_k = 16

    # Create the nearest neighbor encoder
    nearest_neighbor_encoder = NearestNeighborEncoder(chunk_len, 6, {3}, d_model, n_heads, d_k, d_ff)
    # Create the model
    model = RetroModel(tds.n_tokens, d_model, 6,
                       {3, 5},
                       chunk_len, n_heads, d_k, d_ff,
                       encoder=nearest_neighbor_encoder)
    # Move the model to the device
    model = model.to(device)
    # Create the optimizer
    optimizer = Noam(model.parameters(), lr=1., d_model=d_model, warmup=2_000)
    # Create the `Trainer`
    trainer = Trainer(device, model, train_dl, optimizer)
    # Create the `Sampler`
    sampler = Sampler(device, model, tds, chunk_len)
    #
    prompt = '''Second Citizen:\nOne word, good citizens.\n\nFirst Citizen:'''

    # Set models for saving and loading
    experiment.add_pytorch_models(model=model)

    # Start the experiment
    with experiment.start():
        # Train for `32` epochs
        for epoch in monit.loop(32):
            # Train
            trainer()
            # Print a new line
            tracker.new_line()
            # Sample from the `prompt`
            logger.log([(prompt.replace('\n', '\\n\n'), Text.subtle),
                        (sampler.sample(prompt, 128).replace('\n', '\\n\n'), Text.none)])
            # Save models
            experiment.save_checkpoint()


#
if __name__ == '__main__':
    train()
