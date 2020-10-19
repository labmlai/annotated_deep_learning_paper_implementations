"""
This is an annotated implementation of the paper
[A Neural Representation of Sketch Drawings](https://arxiv.org/abs/1704.03477).

Download data from [Quick, Draw! Dataset](https://github.com/googlecreativelab/quickdraw-dataset).
There is a link to download `npz` files in *Sketch-RNN QuickDraw Dataset* section of the readme.
Place the downloaded `npz` file(s) in `data/sketch` folder.
This code is configured to use `bicycle` dataset.
You can change this in configurations.

### Acknowledgements
Took help from [PyTorch Sketch RNN)(https://github.com/alexis-jacq/Pytorch-Sketch-RNN) project by
[Alexis David Jacq](https://github.com/alexis-jacq)
"""

import math
from typing import Optional, Tuple

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
    """
    ## Dataset

    This class load and pre-process the data.
    """

    def __init__(self, dataset: np.array, max_seq_length: int, scale: Optional[float] = None):
        """
        `dataset` is a list of numpy arrays of shape [seq_len, 3].
        It is a sequence of strokes, and each stroke is represented by
        3 integers.
        First two are the displacements along x and y ($\Delta x$, $\Delta y$)
        And the last integer represents the state of the pen - $1$ if it's touching
        the paper and $0$ otherwise.
        """

        data = []
        # We iterate through each of the sequences and filter
        for seq in dataset:
            # Filter if the length of the the sequence of strokes is within our range
            if 10 < len(seq) <= max_seq_length:
                # Clamp $\Delta x$, $\Delta y$ to $[-1000, 1000]$
                seq = np.minimum(seq, 1000)
                seq = np.maximum(seq, -1000)
                # Convert to a floating point array and add to `data`
                seq = np.array(seq, dtype=np.float32)
                data.append(seq)

        # We then calculate the scaling factor which is the
        # standard deviation of ($\Delta x$, $\Delta y$) combined.
        # Paper notes that the mean is not adjusted for simplicity,
        # since the mean is anyway close to $0$.
        if scale is None:
            scale = np.std(np.concatenate([np.ravel(s[:, 0:2]) for s in data]))
        self.scale = scale

        # Get the longest sequence length among all sequences
        longest_seq_len = max([len(seq) for seq in data])

        # We initialize PyTorch data array with two extra steps for start-of-sequence (sos)
        # and end-of-sequence (eos).
        # Each step is a vector $(\Delta x, \Delta y, p_1, p_2, p_3)$.
        # Only one of $p_1, p_2, p_3$ is $1$ and the others are $0$.
        # They represent *pen down*, *pen up* and *end-of-sequence* in that order.
        # $p_1$ is $1$ if the pen touches the paper in the next step.
        # $p_2$ is $1$ if the pen doesn't touch the paper in the next step.
        # $p_2$ is $1$ if it is the end of the drawing.
        self.data = torch.zeros(len(data), longest_seq_len + 2, 5, dtype=torch.float)
        # The mask array is needs only one extra-step since it is for the outputs of the
        # decoder, which takes in `data[:-1]` and predicts next step.
        self.mask = torch.zeros(len(data), longest_seq_len + 1)

        for i, seq in enumerate(data):
            seq = torch.from_numpy(seq)
            len_seq = len(seq)
            # Scale and set $\Delta x, \Delta y$
            self.data[i, 1:len_seq + 1, :2] = seq[:, :2] / scale
            # $p_1$
            self.data[i, 1:len_seq + 1, 2] = 1 - seq[:, 2]
            # $p_2$
            self.data[i, 1:len_seq + 1, 3] = seq[:, 2]
            # $p_3$
            self.data[i, len_seq + 1:, 4] = 1
            # Mask is on until end of sequence
            self.mask[i, :len_seq + 1] = 1

        # Start-of-sequence is $(0, 0, 1, 0, 0)
        self.data[:, 0, 2] = 1

    def __len__(self):
        """Size of the dataset"""
        return len(self.data)

    def __getitem__(self, idx: int):
        """Get a sample"""
        return self.data[idx], self.mask[idx]


class EncoderRNN(Module):
    """
    ## Encoder module

    This consists of a bidirectional LSTM
    """
    def __init__(self, d_z: int, enc_hidden_size: int):
        super().__init__()
        # Create a bidirectional LSTM takes a sequence of
        # $(\Delta x, \Delta y, p_1, p_2, p_3)$ as input.
        self.lstm = nn.LSTM(5, enc_hidden_size, bidirectional=True)
        # Head to get $\mu$
        self.mu_head = nn.Linear(2 * enc_hidden_size, d_z)
        # Head to get $\hat{\sigma}$
        self.sigma_head = nn.Linear(2 * enc_hidden_size, d_z)

    def __call__(self, inputs: torch.Tensor, state=None):
        # The hidden state of the bidirectional LSTM is the concatenation of the
        # output of the last token in the forward direction and
        # and first token in the reverse direction.
        # Which is what we want.
        # $$h_{\rightarrow} = encode_{\rightarrow}(S),
        # h_{\leftarrow} = encode←_{\leftarrow}(S_{reverse}),
        # h = [h_{\rightarrow}; h_{\leftarrow}]$$
        _, (hidden, cell) = self.lstm(inputs.float(), state)
        # The state has shape `[2, batch_size, hidden_size]`
        # where the first dimension is the direction.
        # We rearrange it to get $h = [h_{\rightarrow}; h_{\leftarrow}]$
        hidden = einops.rearrange(hidden, 'fb b h -> b (fb h)')

        # $\mu$
        mu = self.mu_head(hidden)
        # $\hat{\sigma}$
        sigma_hat = self.sigma_head(hidden)
        # $\sigma = \exp(\frac{\hat{\sigma}}{2})$
        sigma = torch.exp(sigma_hat / 2.)

        # Sample $z = \mu + \sigma \cdot \mathcal{N}(0, I)$
        z = mu + sigma * torch.normal(mu.new_zeros(mu.shape), mu.new_ones(mu.shape))

        return z, mu, sigma_hat


class DecoderRNN(Module):
    """
    ## Encoder module

    This consists of a LSTM
    """
    def __init__(self, d_z: int, dec_hidden_size: int, n_distributions: int):
        super().__init__()
        # LSTM takes $[z; (\Delta x, \Delta y, p_1, p_2, p_3)$ as input
        self.lstm = nn.LSTM(d_z + 5, dec_hidden_size)

        # Initial state of the LSTM is $[h_0; c_0] = \tanh(W_{z}z + b_z)$.
        # `init_state` is the linear transformation for this
        self.init_state = nn.Linear(d_z, 2 * dec_hidden_size)

        # This layer produces outputs for each of of the `n_distributions`.
        # Each distribution needs six parameters
        # $(\hat{\Pi_i}, \mu_{x_i}, \mu_{y_i}, \hat{\sigma_{x_i}}, \hat{\sigma_{y_i}} \hat{\rho_{xy_i}})$
        self.mixtures = nn.Linear(dec_hidden_size, 6 * n_distributions)

        # This head is for the logits $(\hat{q_1}, \hat{q_2}, \hat{q_3})$
        self.q_head = nn.Linear(dec_hidden_size, 3)
        # This is to calculate $\log(q_k)$ where
        # $$q_k = \frac{\exp(\hat{q_k}}{\sum_{j = 1}^3 \exp(\hat{q_j}}$$
        self.q_log_softmax = nn.LogSoftmax(-1)

        # These parameters are stored for future reference
        self.n_distributions = n_distributions
        self.dec_hidden_size = dec_hidden_size

    def __call__(self, x: torch.Tensor, z: torch.Tensor, state: Optional[Tuple[torch.Tensor, torch.Tensor]]):
        # Calculate the initial state
        if state is None:
            # $[h_0; c_0] = \tanh(W_{z}z + b_z)$
            h, c = torch.split(torch.tanh(self.init_state(z)), self.dec_hidden_size, 1)
            # `h` and `c` have shapes `[batch_size, lstm_size]`. We want to make them
            # to shape `[1, batch_size, lstm_size]` because that's the shape used in LSTM.
            state = (h.unsqueeze(0).contiguous(), c.unsqueeze(0).contiguous())

        # Run the LSTM
        outputs, state = self.lstm(x, state)

        # Get $\log(q)$
        q_logits = self.q_log_softmax(self.q_head(outputs))

        # Get $(\hat{\Pi_i}, \mu_{x_i}, \mu_{y_i}, \hat{\sigma_{x_i}},
        # \hat{\sigma_{y_i}} \hat{\rho_{xy_i}})$.
        # `torch.split` splits the output into 6 tensors of size `self.n_distribution`
        # across dimension `2`.
        pi_logits, mu_x, mu_y, sigma_x, sigma_y, rho_xy = \
            torch.split(self.mixtures(outputs), self.n_distributions, 2)

        # Create a bivariate gaussian mixture
        dist = BivariateGaussianMixture(pi_logits, mu_x, mu_y,
                                        torch.exp(sigma_x), torch.exp(sigma_y), torch.tanh(rho_xy))

        return dist, q_logits, state


class ReconstructionLoss(Module):
    def __call__(self, mask: torch.Tensor, target: torch.Tensor,
                 dist: 'BivariateGaussianMixture', q_logits: torch.Tensor):
        pi, mix = dist.get_distribution()
        xy = target[:, :, 0:2].unsqueeze(-2).expand(-1, -1, dist.n_distributions, -1)
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
        sos = data.new_tensor([0, 0, 1, 0, 0])
        seq_len = len(data)
        s = sos
        seq = [s]
        state = None
        with torch.no_grad():
            for i in range(seq_len):
                data = torch.cat([s.view(1, 1, -1), z.unsqueeze(0)], 2)
                dist, q_logits, state = self.decoder(data, z, state)
                s = self._sample_step(dist, q_logits, temperature)
                seq.append(s)
                if s[4] == 1:
                    break

        seq = torch.stack(seq)

        self.plot(seq)

    @staticmethod
    def plot(seq: torch.Tensor):
        seq[:, 0:2] = torch.cumsum(seq[:, 0:2], dim=0)
        seq[:, 2] = seq[:, 3] == 1
        seq = seq[:, 0:3].detach().cpu().numpy()

        strokes = np.split(seq, np.where(seq[:, 2] > 0)[0] + 1)
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
        next_pos = q_logits.new_zeros(5)
        next_pos[:2] = xy
        next_pos[q_idx + 2] = 1
        return next_pos


class Configs(TrainValidConfigs):
    device: torch.device = DeviceConfigs()
    encoder: EncoderRNN
    decoder: DecoderRNN
    optimizer: optim.Adam = 'setup_all'
    sampler: Sampler

    dataset_name: str
    train_loader = 'setup_all'
    valid_loader = 'setup_all'
    train_dataset: StrokesDataset
    valid_dataset: StrokesDataset

    enc_hidden_size = 256
    dec_hidden_size = 512

    batch_step = 'strokes_batch_step'

    batch_size = 100

    d_z = 128
    n_distributions = 20

    kl_div_loss_weight = 0.5
    grad_clip = 1.
    temperature = 0.4
    max_seq_length = 200

    epochs = 100

    def sample(self):
        data, *_ = self.train_dataset[np.random.choice(len(self.train_dataset))]
        data = data.unsqueeze(1).to(self.device)
        self.sampler.sample(data, self.temperature)


@setup([Configs.encoder, Configs.decoder, Configs.optimizer, Configs.sampler,
        Configs.train_dataset, Configs.train_loader,
        Configs.valid_dataset, Configs.valid_loader])
def setup_all(self: Configs):
    self.encoder = EncoderRNN(self.d_z, self.enc_hidden_size).to(self.device)
    self.decoder = DecoderRNN(self.d_z, self.dec_hidden_size, self.n_distributions).to(self.device)

    self.optimizer = OptimizerConfigs()
    self.optimizer.parameters = list(self.encoder.parameters()) + list(self.decoder.parameters())

    self.sampler = Sampler(self.encoder, self.decoder)
    # `npz` file path is `data/sketch/[DATASET NAME].npz`
    path = lab.get_data_path() / 'sketch' / f'{self.dataset_name}.npz'
    # Load the numpy file.
    dataset = np.load(str(path), encoding='latin1', allow_pickle=True)

    self.train_dataset = StrokesDataset(dataset['train'], self.max_seq_length)
    self.valid_dataset = StrokesDataset(dataset['valid'], self.max_seq_length, self.train_dataset.scale)

    self.train_loader = DataLoader(self.train_dataset, self.batch_size, shuffle=True)
    self.valid_loader = DataLoader(self.valid_dataset, self.batch_size, shuffle=True)


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
        tracker.set_scalar("loss.total.*", True)

    def prepare_for_iteration(self):
        if MODE_STATE.is_train:
            self.encoder.train()
            self.decoder.train()
        else:
            self.encoder.eval()
            self.decoder.eval()

    def process(self, batch: any, state: any):
        device = self.encoder.device
        data, mask = batch
        data = data.to(device).transpose(0, 1)
        mask = mask.to(device).transpose(0, 1)

        with monit.section("encoder"):
            z, mu, sigma = self.encoder(data)

        with monit.section("decoder"):
            z_stack = z.unsqueeze(0).expand(data.shape[0] - 1, -1, -1)
            inputs = torch.cat([data[:-1], z_stack], 2)
            dist, q_logits, _ = self.decoder(inputs, z, None)

        with monit.section('loss'):
            kl_loss = self.kl_div_loss(sigma, mu)
            reconstruction_loss = self.reconstruction_loss(mask, data[1:], dist, q_logits)
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
    def n_distributions(self):
        return self.pi_logits.shape[-1]

    def set_temperature(self, temperature: float):
        self.pi_logits /= temperature
        self.sigma_x *= math.sqrt(temperature)
        self.sigma_y *= math.sqrt(temperature)

    def get_distribution(self):
        sigma_x = torch.clamp_min(self.sigma_x, 1e-5)
        sigma_y = torch.clamp_min(self.sigma_y, 1e-5)
        rho_xy = torch.clamp(self.rho_xy, -1 + 1e-5, 1 - 1e-5)

        mean = torch.stack([self.mu_x, self.mu_y], -1)
        cov = torch.stack([
            sigma_x * sigma_x, rho_xy * sigma_x * sigma_y,
            rho_xy * sigma_x * sigma_y, sigma_y * sigma_y
        ], -1)
        cov = cov.view(*sigma_y.shape, 2, 2)

        multi_dist = torch.distributions.MultivariateNormal(mean, covariance_matrix=cov)
        cat_dist = torch.distributions.Categorical(logits=self.pi_logits)

        return cat_dist, multi_dist


def main():
    configs = Configs()
    experiment.create(name="sketch_rnn")
    experiment.configs(configs, {
        'optimizer.optimizer': 'Adam',
        'optimizer.learning_rate': 1e-3,
        'dataset_name': 'bicycle',
        'inner_iterations': 10
    }, 'run')
    experiment.start()

    configs.run()


if __name__ == "__main__":
    main()
