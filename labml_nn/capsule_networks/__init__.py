"""
This is an implementation of paper
"""

import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

from labml import experiment, tracker
from labml.configs import option
from labml.utils.pytorch import get_device
from labml_helpers.datasets.mnist import MNISTConfigs
from labml_helpers.device import DeviceConfigs
from labml_helpers.module import Module
from labml_helpers.train_valid import TrainValidConfigs, BatchStep


class Squash(Module):
    """
    This is **squashing** function from paper.
    """

    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon

    def __call__(self, s: torch.Tensor):
        # shape: batch, caps, features
        s2 = (s ** 2).sum(dim=-1, keepdims=True)
        return (s2 / (1 + s2)) * (s / torch.sqrt(s2 + self.epsilon))


class Router(Module):
    """
    The routing mechanism
    """

    def __init__(self, in_caps: int, out_caps: int, in_d: int, out_d: int,
                 iterations: int):
        super().__init__()
        self.in_caps = in_caps
        self.out_caps = out_caps
        self.iterations = iterations
        self.weight = nn.Parameter(torch.randn(in_caps, out_caps, in_d, out_d))
        self.softmax = nn.Softmax(dim=1)
        self.squash = Squash()

    def __call__(self, u: torch.Tensor):
        # batch, in_caps, in_d
        u_hat = torch.einsum('ijnm,bin->bijm', self.weight, u)
        b = u.new_zeros(u.shape[0], self.in_caps, self.out_caps)
        v = None
        for i in range(self.iterations):
            c = self.softmax(b)
            s = torch.einsum('bij,bijm->bjm', c, u_hat)
            v = self.squash(s)
            a = torch.einsum('bjm,bijm->bij', v, u_hat)
            b = b + a

        return v


class MarginLoss(Module):
    def __init__(self, *, n_labels: int, lambda_: float = 0.5, m_positive: float = 0.9, m_negative: float = 0.1):
        super().__init__()

        self.m_negative = m_negative
        self.m_positive = m_positive
        self.lambda_ = lambda_
        self.n_labels = n_labels

    def __call__(self, v: torch.Tensor, labels: torch.Tensor):
        v_norm = torch.sqrt((v ** 2).sum(dim=-1))

        labels = torch.eye(self.n_labels, device=labels.device)[labels]
        loss = labels * F.relu(self.m_positive - v_norm) + \
               self.lambda_ * (1.0 - labels) * F.relu(v_norm - self.m_negative)
        loss = loss.sum(dim=-1).mean()

        return loss


class MNISTCapsuleNetworkModel(Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=9, stride=1)
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=32 * 8, kernel_size=9, stride=2, padding=0)
        self.squash = Squash()

        # self.digit_capsules = DigitCaps()
        self.digit_capsules = Router(32 * 6 * 6, 10, 8, 16, 3)
        self.reconstruct = nn.Sequential(
            nn.Linear(16 * 10, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 784),
            nn.Sigmoid()
        )

        self.mse_loss = nn.MSELoss()

    def forward(self, data):
        x = F.relu(self.conv1(data))
        caps = self.conv2(x).view(x.shape[0], 8, 32 * 6 * 6).permute(0, 2, 1)
        caps = self.squash(caps)
        caps = self.digit_capsules(caps)

        with torch.no_grad():
            pred = (caps ** 2).sum(-1).argmax(-1)
            masked = torch.eye(10, device=x.device)[pred]

        reconstructions = self.reconstruct((caps * masked[:, :, None]).view(x.shape[0], -1))
        reconstructions = reconstructions.view(-1, 1, 28, 28)

        return caps, reconstructions, pred


class CapsuleNetworkBatchStep(BatchStep):
    def __init__(self, *, model, optimizer):
        super().__init__(model=model, optimizer=optimizer, loss_func=None, accuracy_func=None)
        self.reconstruction_loss = nn.MSELoss()
        self.margin_loss = MarginLoss(n_labels=10)

    def calculate_loss(self, batch: any, state: any):
        device = get_device(self.model)
        data, target = batch
        data, target = data.to(device), target.to(device)
        stats = {'samples': len(data)}

        caps, reconstructions, pred = self.model(data)

        loss = self.margin_loss(caps, target) + 0.0005 * self.reconstruction_loss(reconstructions, data)

        stats['correct'] = pred.eq(target).sum().item()
        stats['loss'] = loss.detach().item() * stats['samples']
        tracker.add("loss.", loss)

        return loss, stats, None


class Configs(MNISTConfigs, TrainValidConfigs):
    batch_step = 'capsule_network_batch_step'
    device: torch.device = DeviceConfigs()
    epochs: int = 10

    loss_func = None
    accuracy_func = None


@option(Configs.model)
def model(c: Configs):
    return MNISTCapsuleNetworkModel().to(c.device)


@option(Configs.batch_step)
def capsule_network_batch_step(c: TrainValidConfigs):
    return CapsuleNetworkBatchStep(model=c.model, optimizer=c.optimizer)


def main():
    conf = Configs()
    experiment.create(name='mnist_latest', writers={})
    experiment.configs(conf, {'optimizer.optimizer': 'Adam',
                              'device.cuda_device': 1},
                       'run')
    experiment.add_pytorch_models(dict(model=conf.model))
    with experiment.start():
        conf.run()


if __name__ == '__main__':
    main()
