from typing import Any

import torch.nn as nn
import torch.utils.data

from labml import tracker, experiment
from labml.configs import option, calculate
from labml_helpers.module import Module
from labml_helpers.schedule import Schedule, RelativePiecewise
from labml_helpers.train_valid import BatchIndex
from labml_nn.experiments.mnist import MNISTConfigs
from labml_nn.uncertainty.evidence import KLDivergenceLoss, TrackStatistics, MaximumLikelihoodLoss, \
    CrossEntropyBayesRisk, SquaredErrorBayesRisk


class Model(Module):
    def __init__(self, dropout: float):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5)
        self.act2 = nn.ReLU()
        self.max_pool1 = nn.MaxPool2d(2, 2)
        self.max_pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(50 * 4 * 4, 500)
        self.act3 = nn.ReLU()
        self.fc2 = nn.Linear(500, 10)
        self.dropout = nn.Dropout(p=dropout)

    def __call__(self, x: torch.Tensor):
        x = self.max_pool1(self.act1(self.conv1(x)))
        x = self.max_pool2(self.act2(self.conv2(x)))
        x = x.view(x.shape[0], -1)
        x = self.act3(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


class Configs(MNISTConfigs):
    model: Module
    epochs: int = 10
    # dataset_transforms = transforms.ToTensor()
    annealing_coef = [(0, 0), (1, 1)]
    loss_func: Module
    kl_div_loss = KLDivergenceLoss()
    stats = TrackStatistics()
    inner_iterations = 1
    kl_div_coef: Schedule
    kl_div_coef_schedule = [(0, 0.), (0.2, 0.01), (1, 1.)]
    dropout: float = 0.5
    logits_to_evidence = nn.ReLU()

    def init(self):
        """
        ### Initialization
        """
        # Set tracker configurations
        tracker.set_scalar("loss.*", True)
        tracker.set_scalar("accuracy.*", True)

        tracker.set_histogram('u.*', False)
        tracker.set_histogram('prob.*', True)
        tracker.set_scalar('annealing_coef.*', False)

        self.state_modules = []

    def step(self, batch: Any, batch_idx: BatchIndex):
        self.model.train(self.mode.is_train)
        data, target = batch[0].to(self.device), batch[1].to(self.device)
        eye = torch.eye(10).to(torch.float).to(self.device)
        target = eye[target]

        if self.mode.is_train:
            tracker.add_global_step(len(data))

        logits = self.model(data)
        evidence = self.logits_to_evidence(logits)

        loss = self.loss_func(evidence, target)
        kl_div_loss = self.kl_div_loss(evidence, target)
        tracker.add("loss.", loss)
        tracker.add("kl_div_loss.", kl_div_loss)

        annealing_coef = min(1., self.kl_div_coef(tracker.get_global_step()))
        tracker.add("annealing_coef.", annealing_coef)
        loss = loss + annealing_coef * kl_div_loss

        self.stats(evidence, target)

        if self.mode.is_train:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            tracker.save()


@option(Configs.model)
def mnist_model(c: Configs):
    return Model(c.dropout).to(c.device)


@option(Configs.kl_div_coef)
def kl_div_coef(c: Configs):
    return RelativePiecewise(c.kl_div_coef_schedule, c.epochs * len(c.train_dataset))


calculate(Configs.loss_func, 'max_likelihood_loss', lambda: MaximumLikelihoodLoss())
calculate(Configs.loss_func, 'cross_entropy_bayes_risk', lambda: CrossEntropyBayesRisk())
calculate(Configs.loss_func, 'squared_error_bayes_risk', lambda: SquaredErrorBayesRisk())

calculate(Configs.logits_to_evidence, 'relu', lambda: nn.ReLU())
calculate(Configs.logits_to_evidence, 'softplus', lambda: nn.Softplus())


def main():
    experiment.create(name='evidence_mnist')

    conf = Configs()
    experiment.configs(conf, {
        'optimizer.optimizer': 'Adam',
        'optimizer.learning_rate': 0.001,
        'optimizer.weight_decay': 0.005,

        # 'loss_func': 'max_likelihood_loss',
        # 'loss_func': 'cross_entropy_bayes_risk',
        'loss_func': 'squared_error_bayes_risk',

        'logits_to_evidence': 'softplus',

        # High dropout is important
        'dropout': 0.5,
    })
    with experiment.start():
        conf.run()


if __name__ == '__main__':
    main()
