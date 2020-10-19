"""
Download data from https://github.com/googlecreativelab/quickdraw-dataset

**Acknowledgement**
* Took help from https://github.com/alexis-jacq/Pytorch-Sketch-RNN
"""

import math
from typing import Optional

import einops
import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch import optim
from torch.utils.data import Dataset, DataLoader

from labml import lab, experiment, tracker, monit
from labml.configs import setup, option
from labml.utils import pytorch as pytorch_utils
from labml_helpers.device import DeviceConfigs
from labml_helpers.module import Module
from labml_helpers.optimizer import OptimizerConfigs
from labml_helpers.train_valid import TrainValidConfigs, BatchStepProtocol, hook_model_outputs, \
    MODE_STATE


class StrokesDataset(Dataset):
    def __init__(self, dataset_name: str, max_seq_length):
        path = lab.get_data_path() / 'sketch' / f'{dataset_name}.npz'
        dataset = np.load(str(path), encoding='latin1', allow_pickle=True)
        data = []
        for seq in dataset['train']:
            if seq.shape[0] <= max_seq_length and seq.shape[0] > 10:
                seq = np.minimum(seq, 1000)
                seq = np.maximum(seq, -1000)
                seq = np.array(seq, dtype=np.float32)
                data.append(seq)
        std = np.std(np.concatenate([np.ravel(s[:, 0:2]) for s in data]))
        for s in data:
            s[:, 0:2] /= std
        longest_seq_len = max([len(seq) for seq in data])

        self.data = torch.zeros(len(data), longest_seq_len, 5, dtype=torch.float)
        mask = torch.zeros(len(data), longest_seq_len + 1)

        for i, seq in enumerate(data):
            seq = torch.from_numpy(seq)
            len_seq = len(seq)
            # set x, y
            self.data[i, :len_seq, :2] = seq[:, :2]
            # set pen status
            self.data[i, :len_seq - 1, 2] = 1 - seq[:-1, 2]
            self.data[i, :len_seq - 1, 3] = seq[:-1, 2]
            self.data[i, len_seq - 1:, 4] = 1
            mask[i, :len_seq] = 1

        eos = torch.zeros(len(data), 1, 5)
        self.target = torch.cat([self.data, eos], 1)
        self.mask = mask

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx], self.target[idx], self.mask[idx]


class EncoderRNN(Module):
    def __init__(self, d_z: int, enc_hidden_size: int):
        super().__init__()
        self.lstm = nn.LSTM(5, enc_hidden_size, bidirectional=True)
        self.mu_head = nn.Linear(2 * enc_hidden_size, d_z)
        self.sigma_head = nn.Linear(2 * enc_hidden_size, d_z)

    def __call__(self, inputs: torch.Tensor, state=None):
        _, (hidden, cell) = self.lstm(inputs.float(), state)
        hidden = einops.rearrange(hidden, 'fb b h -> b (fb h)')

        mu = self.mu_head(hidden)
        sigma_hat = self.sigma_head(hidden)
        sigma = torch.exp(sigma_hat / 2.)

        z_size = mu.size()
        z = mu + sigma * torch.normal(mu.new_zeros(z_size), mu.new_ones(z_size))

        return z, mu, sigma_hat


class DecoderRNN(Module):
    def __init__(self, d_z: int, dec_hidden_size: int, n_mixtures: int):
        super().__init__()
        self.dec_hidden_size = dec_hidden_size
        self.init_state = nn.Linear(d_z, 2 * dec_hidden_size)

        self.lstm = nn.LSTM(d_z + 5, dec_hidden_size)

        self.mixtures = nn.Linear(dec_hidden_size, 6 * n_mixtures)
        self.q_head = nn.Linear(dec_hidden_size, 3)
        self.n_mixtures = n_mixtures
        self.q_log_softmax = nn.LogSoftmax(-1)

    def __call__(self, inputs, z: torch.Tensor, state=None):
        if state is None:
            hidden, cell = torch.split(torch.tanh(self.init_state(z)), self.dec_hidden_size, 1)
            state = (hidden.unsqueeze(0).contiguous(), cell.unsqueeze(0).contiguous())

        outputs, (hidden, cell) = self.lstm(inputs, state)
        if not self.training:
            # We dont have to change shape since hidden has shape [1, batch_size, hidden_size]
            # and outputs is of shape [seq_len, batch_size, hidden_size], since we are
            # using a single direction one layer lstm
            outputs = hidden

        q_logits = self.q_log_softmax(self.q_head(outputs))

        pi_logits, mu_x, mu_y, sigma_x, sigma_y, rho_xy = \
            torch.split(self.mixtures(outputs), self.n_mixtures, 2)

        dist = BivariateGaussianMixture(pi_logits, mu_x, mu_y,
                                        torch.exp(sigma_x), torch.exp(sigma_y), torch.tanh(rho_xy))

        return dist, q_logits, (hidden, cell)


class ReconstructionLoss(Module):
    def __call__(self, mask: torch.Tensor, target: torch.Tensor,
                 dist: 'BivariateGaussianMixture', q_logits: torch.Tensor):
        pi, mix = dist.get_distribution()
        xy = target[:, :, 0:2].unsqueeze(-2).expand(-1, -1, dist.n_mixtures, -1)
        probs = torch.sum(pi.probs * torch.exp(mix.log_prob(xy)), 2)
        loss_stroke = -torch.mean(mask * torch.log(1e-5 + probs))
        loss_pen = -torch.mean(target[:, :, 2:] * q_logits)
        return loss_stroke + loss_pen


class KLDivLoss(Module):
    def __call__(self, sigma_hat: torch.Tensor, mu: torch.Tensor):
        return -0.5 * torch.mean(1 + sigma_hat - mu ** 2 - torch.exp(sigma_hat))


class Sampler:
    def __init__(self, encoder: EncoderRNN, decoder: DecoderRNN):
        self.decoder = decoder
        self.encoder = encoder

    def sample(self, data: torch.Tensor, temperature: float):
        z, _, _ = self.encoder(data)
        sos = data.new_tensor([[[0, 0, 1, 0, 0]]])
        seq_len = len(data)
        s = sos
        seq_x = []
        seq_y = []
        seq_z = []
        state = None
        with torch.no_grad():
            for i in range(seq_len):
                data = torch.cat([s, z.unsqueeze(0)], 2)
                dist, q_logits, state = self.decoder(data, z, state)
                s, xy, pen_down, eos = self._sample_step(dist, q_logits, temperature)
                seq_x.append(xy[0].item())
                seq_y.append(xy[1].item())
                seq_z.append(pen_down)
                if eos:
                    break

        x_sample = np.cumsum(seq_x, 0)
        y_sample = np.cumsum(seq_y, 0)
        z_sample = np.array(seq_z)
        sequence = np.stack([x_sample, y_sample, z_sample]).T

        strokes = np.split(sequence, np.where(sequence[:, 2] > 0)[0] + 1)
        for s in strokes:
            plt.plot(s[:, 0], -s[:, 1])
        plt.axis('off')
        plt.show()

    @staticmethod
    def _sample_step(dist: 'BivariateGaussianMixture', q_logits: torch.Tensor, temperature: float):
        dist.set_temperature(temperature)
        pi, mix = dist.get_distribution()
        idx = pi.sample()[0, 0]

        q = torch.distributions.Categorical(logits=q_logits / temperature)
        q_idx = q.sample()[0, 0]

        xy = mix.sample()[0, 0, idx]
        next_pos = q_logits.new_zeros(1, 1, 5)
        next_pos[0, 0, :2] = xy
        next_pos[0, 0, q_idx + 2] = 1
        return next_pos, xy, q_idx == 1, q_idx == 2


class Configs(TrainValidConfigs):
    device: torch.device = DeviceConfigs()
    encoder: EncoderRNN
    decoder: DecoderRNN
    optimizer: optim.Adam = 'setup_all'
    sampler: Sampler

    dataset_name = 'bicycle'
    dataset: StrokesDataset = 'setup_all'
    train_loader = 'setup_all'

    enc_hidden_size = 256
    dec_hidden_size = 512

    batch_step = 'strokes_batch_step'

    batch_size = 100

    d_z = 128
    n_mixtures = 20

    kl_div_loss_weight = 0.5
    grad_clip = 1.
    temperature = 0.4
    max_seq_length = 200

    validator = None
    valid_loader = None
    epochs = 100

    def sample(self):
        data, *_ = self.dataset[np.random.choice(len(self.dataset))]
        data = data.unsqueeze(1).to(self.device)
        self.sampler.sample(data, self.temperature)


@setup([Configs.encoder, Configs.decoder, Configs.optimizer, Configs.sampler,
        Configs.dataset, Configs.train_loader])
def setup_all(self: Configs):
    self.encoder = EncoderRNN(self.d_z, self.enc_hidden_size).to(self.device)
    self.decoder = DecoderRNN(self.d_z, self.dec_hidden_size, self.n_mixtures).to(self.device)

    self.optimizer = OptimizerConfigs()
    self.optimizer.parameters = list(self.encoder.parameters()) + list(self.decoder.parameters())

    self.sampler = Sampler(self.encoder, self.decoder)

    self.dataset = StrokesDataset(self.dataset_name, self.max_seq_length)
    self.train_loader = DataLoader(self.dataset, self.batch_size, shuffle=True)


class StrokesBatchStep(BatchStepProtocol):
    def __init__(self, encoder: EncoderRNN, decoder: DecoderRNN,
                 optimizer: Optional[torch.optim.Adam],
                 kl_div_loss_weight: float, grad_clip: float):
        self.grad_clip = grad_clip
        self.kl_div_loss_weight = kl_div_loss_weight
        self.encoder = encoder
        self.decoder = decoder
        self.optimizer = optimizer
        self.kl_div_loss = KLDivLoss()
        self.reconstruction_loss = ReconstructionLoss()

        hook_model_outputs(self.encoder, 'encoder')
        hook_model_outputs(self.decoder, 'decoder')
        tracker.set_scalar("loss.*", True)
        tracker.set_image("generated", True)

    def prepare_for_iteration(self):
        if MODE_STATE.is_train:
            self.encoder.train()
            self.decoder.train()
        else:
            self.encoder.eval()
            self.decoder.eval()

    def process(self, batch: any, state: any):
        device = self.encoder.device
        data, target, mask = batch
        data = data.to(device).transpose(0, 1)
        target = target.to(device).transpose(0, 1)
        mask = mask.to(device).transpose(0, 1)
        batch_size = data.shape[1]
        seq_len = data.shape[0]

        with monit.section("encoder"):
            z, mu, sigma = self.encoder(data)

        with monit.section("decoder"):
            sos = torch.stack([torch.tensor([0, 0, 1, 0, 0])] * batch_size). \
                unsqueeze(0).to(device)
            batch_init = torch.cat([sos, data], 0)
            z_stack = torch.stack([z] * (seq_len + 1))
            inputs = torch.cat([batch_init, z_stack], 2)
            dist, q_logits, _ = self.decoder(inputs, z)

        with monit.section('loss'):
            kl_loss = self.kl_div_loss(sigma, mu)
            reconstruction_loss = self.reconstruction_loss(mask, target, dist, q_logits)
            loss = self.kl_div_loss_weight * kl_loss + reconstruction_loss

            tracker.add("loss.kl.", kl_loss)
            tracker.add("loss.reconstruction.", reconstruction_loss)
            tracker.add("loss.total.", loss)

        with monit.section('optimize'):
            if MODE_STATE.is_train:
                self.optimizer.zero_grad()
                loss.backward()
                if MODE_STATE.is_log_parameters:
                    pytorch_utils.store_model_indicators(self.encoder, 'encoder')
                    pytorch_utils.store_model_indicators(self.decoder, 'decoder')
                nn.utils.clip_grad_norm_(self.encoder.parameters(), self.grad_clip)
                nn.utils.clip_grad_norm_(self.decoder.parameters(), self.grad_clip)
                self.optimizer.step()

            # tracker.add('generated', generated_images[0:5])

        return {'samples': len(data)}, None


@option(Configs.batch_step)
def strokes_batch_step(c: Configs):
    return StrokesBatchStep(c.encoder, c.decoder, c.optimizer, c.kl_div_loss_weight, c.grad_clip)


class BivariateGaussianMixture:
    def __init__(self, pi_logits: torch.Tensor, mu_x: torch.Tensor, mu_y: torch.Tensor,
                 sigma_x: torch.Tensor, sigma_y: torch.Tensor, rho_xy: torch.Tensor):
        self.pi_logits = pi_logits
        self.mu_x = mu_x
        self.mu_y = mu_y
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.rho_xy = rho_xy

    @property
    def n_mixtures(self):
        return self.pi_logits.shape[-1]

    def set_temperature(self, temperature: float):
        self.pi_logits /= temperature
        self.sigma_x *= math.sqrt(temperature)
        self.sigma_y *= math.sqrt(temperature)

    def get_distribution(self):
        mean = torch.stack([self.mu_x, self.mu_y], -1)
        cov = torch.stack([
            self.sigma_x * self.sigma_x + 1e-6, self.rho_xy * self.sigma_x * self.sigma_y,
            self.rho_xy * self.sigma_x * self.sigma_y, self.sigma_y * self.sigma_y + 1e-6
        ], -1)
        cov = cov.view(*self.sigma_y.shape, 2, 2)

        multi_dist = torch.distributions.MultivariateNormal(mean, covariance_matrix=cov)
        cat_dist = torch.distributions.Categorical(logits=self.pi_logits)

        return cat_dist, multi_dist


def main():
    configs = Configs()
    experiment.create(name="sketch_rnn")
    experiment.configs(configs, {
        'optimizer.optimizer': 'Adam',
        'optimizer.learning_rate': 1e-3
    }, 'run')
    experiment.start()

    configs.run()


if __name__ == "__main__":
    main()
