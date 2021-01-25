"""
---
title: Gated Linear Units and Variants
summary: >
  Train an auto-regressive transformer with Gated Linear Units and variants
  for the position-wise feedforward network (FFN).
---

# Train Autoregressive Transformer

This trains a simple [transformer](../../) model for auto-regression.
"""
import dataclasses

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from labml import experiment, lab, tracker, monit, logger
from labml.logger import Text
from labml.utils.download import download_file
from labml_nn.experiments.nlp_autoregression import transpose_batch
from labml_nn.optimizers.noam import Noam
from labml_nn.transformers import Encoder, MultiHeadAttention
from labml_nn.transformers.feed_forward import FeedForward
from labml_nn.transformers.models import EmbeddingsWithPositionalEncoding, TransformerLayer
from labml_nn.transformers.utils import subsequent_mask


class AutoregressiveModel(nn.Module):
    """
    ## Auto regressive model
    """

    def __init__(self, src_embed: nn.Module, encoder: Encoder, generator: nn.Module):
        super().__init__()
        # Token embedding module
        self.src_embed = src_embed
        # Transformer based encoder
        self.encoder = encoder
        # Next token generation layer;
        # this give logits of the the next token
        self.generator = generator
        # This will be initialized on the first call
        self.src_mask = None

    def __call__(self, src: torch.Tensor):
        # Create subsequent mask, so that the transformer can only pay attention to past tokens.
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            self.src_mask = subsequent_mask(len(src)).to(src.device)
        # Embed the tokens (`src`) and run it through the the transformer
        res = self.encoder(self.src_embed(src), self.src_mask)
        # Generate logits of the next token
        return self.generator(res)


@dataclasses.dataclass
class Configs:
    d_model: int = 512
    seq_len: int = 128
    batch_size: int = 32
    n_layers: int = 6
    n_heads: int = 8
    dropout: float = 0.1
    d_ff: int = 2048
    glu_variant: str = 'GLU'
    epochs: int = 5
    grad_norm_clip: float = 0.5


class TinyShakespeareDataset(Dataset):
    def __init__(self, seq_len: int):
        path = lab.get_data_path() / 'tiny_shakespeare.txt'
        download_file('https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt', path)
        with open(str(path), 'r') as f:
            text = f.read()

        chars = list(set(text))
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for i, c in enumerate(chars)}
        self.seq_len = seq_len
        self.data = self.text_to_i(text)

    def text_to_i(self, text: str):
        return torch.tensor([self.stoi[c] for c in text], dtype=torch.long)

    def __len__(self):
        return len(self.data) - self.seq_len - 1

    def __getitem__(self, idx):
        return self.data[idx:idx + self.seq_len], self.data[idx + 1:idx + self.seq_len + 1]


class Trainer:
    def __init__(self, configs: Configs):
        self.device = torch.device('cpu')
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
        self.dataset = TinyShakespeareDataset(configs.seq_len)
        self.dataloader = DataLoader(self.dataset, batch_size=configs.batch_size, collate_fn=transpose_batch,
                                     shuffle=True)

        if configs.glu_variant == 'GLU':
            ffn = FeedForward(configs.d_model, configs.d_ff, configs.dropout, nn.Sigmoid(), True, False, False, False)
        elif configs.glu_variant == 'Bilinear':
            ffn = FeedForward(configs.d_model, configs.d_ff, configs.dropout, nn.Identity(), True, False, False, False)
        elif configs.glu_variant == 'ReGLU':
            ffn = FeedForward(configs.d_model, configs.d_ff, configs.dropout, nn.ReLU(), True, False, False, False)
        elif configs.glu_variant == 'GEGLU':
            ffn = FeedForward(configs.d_model, configs.d_ff, configs.dropout, nn.GELU(), True, False, False, False)
        elif configs.glu_variant == 'SwiGLU':
            ffn = FeedForward(configs.d_model, configs.d_ff, configs.dropout, nn.SiLU(), True, False, False, False)
        elif configs.glu_variant == 'ReLU':
            ffn = FeedForward(configs.d_model, configs.d_ff, configs.dropout, nn.ReLU())
        elif configs.glu_variant == 'GELU':
            ffn = FeedForward(configs.d_model, configs.d_ff, configs.dropout, nn.GELU())
        else:
            raise ValueError(f'Unknown variant {configs.glu_variant}')

        n_chars = len(self.dataset.stoi)
        self.model = AutoregressiveModel(EmbeddingsWithPositionalEncoding(configs.d_model, n_chars),
                                         Encoder(TransformerLayer(
                                             d_model=configs.d_model,
                                             self_attn=MultiHeadAttention(configs.n_heads, configs.d_model,
                                                                          configs.dropout),
                                             src_attn=None,
                                             feed_forward=ffn,
                                             dropout_prob=configs.dropout
                                         ), configs.n_layers),
                                         nn.Linear(configs.d_model, n_chars))
        self.model.to(self.device)

        self.optimizer = Noam(self.model.parameters(), lr=1.0, warmup=2_000, d_model=configs.d_model)

        self.loss_func = nn.CrossEntropyLoss()
        self.epochs = configs.epochs
        self.grad_norm_clip = configs.grad_norm_clip

        # Set tracker configurations
        tracker.set_scalar("loss.*", True)

    def sample(self):
        """
        ### Sampling function to generate samples periodically while training
        """

        # Starting prompt
        prompt = 'It is'
        # Collect output for printing
        log = [(prompt, Text.subtle)]
        # Sample 25 tokens
        for i in monit.iterate('Sample', 25):
            # Tokenize the prompt
            data = self.dataset.text_to_i(prompt).unsqueeze(-1)
            data = data.to(self.device)
            # Get the model output
            output = self.model(data)
            # Get the model prediction (greedy)
            output = output.argmax(dim=-1).squeeze()
            # Add the prediction to prompt
            prompt += self.dataset.itos[output[-1].item()]
            # Add the prediction for logging
            log += [(self.dataset.itos[output[-1].item()], Text.value)]

        # Print the sampled output
        logger.log(log)

    def train(self):
        for _ in monit.loop(self.epochs):
            for i, batch in monit.enum('Train', self.dataloader):
                # Move data to the device
                data, target = batch[0].to(self.device), batch[1].to(self.device)

                tracker.add_global_step(data.shape[0] * data.shape[1])

                self.model.train()
                output = self.model(data)

                # Calculate and log loss
                loss = self.loss_func(output.view(-1, output.shape[-1]), target.view(-1))
                tracker.add("loss.train", loss)

                # Calculate gradients
                loss.backward()
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_norm_clip)
                # Take optimizer step
                self.optimizer.step()
                # Log the model parameters and gradients on last batch of every epoch
                if (i + 1) % 100 == 0:
                    tracker.add('model', self.model)
                # Clear the gradients
                self.optimizer.zero_grad()

                if (i + 1) % 100 == 0:
                    self.model.eval()
                    with torch.no_grad():
                        self.sample()

                # Save the tracked metrics
                if (i + 1) % 10 == 0:
                    tracker.save()

            experiment.save_checkpoint()


def main():
    # Create experiment
    experiment.create(name="glu_variants")
    # Create configs
    configs = Configs()
    # Load configurations
    experiment.configs(dataclasses.asdict(configs))

    trainer = Trainer(configs)
    experiment.add_pytorch_models({'model': trainer.model})

    # Start the experiment
    with experiment.start():
        # `TrainValidConfigs.run`
        trainer.train()


if __name__ == '__main__':
    main()
