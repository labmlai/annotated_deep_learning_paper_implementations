import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

from labml import experiment, tracker
from labml.configs import option
from labml_helpers.datasets.mnist import MNISTConfigs
from labml_helpers.device import DeviceConfigs
from labml_helpers.metrics.accuracy import Accuracy
from labml_helpers.module import Module
from labml_helpers.seed import SeedConfigs
from labml_helpers.train_valid import TrainValidConfigs, BatchIndex, hook_model_outputs
from labml_nn.normalization.batch_norm import BatchNorm


class Net(Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.bn1 = BatchNorm(20)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.bn2 = BatchNorm(50)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.bn3 = BatchNorm(500)
        self.fc2 = nn.Linear(500, 10)

    def __call__(self, x: torch.Tensor):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.bn3(self.fc1(x)))
        return self.fc2(x)


class Configs(MNISTConfigs, TrainValidConfigs):
    optimizer: torch.optim.Adam
    model: nn.Module
    set_seed = SeedConfigs()
    device: torch.device = DeviceConfigs()
    epochs: int = 10

    is_save_models = True
    model: nn.Module
    inner_iterations = 10

    accuracy_func = Accuracy()
    loss_func = nn.CrossEntropyLoss()

    def init(self):
        tracker.set_queue("loss.*", 20, True)
        tracker.set_scalar("accuracy.*", True)
        hook_model_outputs(self.mode, self.model, 'model')
        self.state_modules = [self.accuracy_func]

    def step(self, batch: any, batch_idx: BatchIndex):
        data, target = batch[0].to(self.device), batch[1].to(self.device)

        if self.mode.is_train:
            tracker.add_global_step(len(data))

        with self.mode.update(is_log_activations=batch_idx.is_last):
            output = self.model(data)

        loss = self.loss_func(output, target)
        self.accuracy_func(output, target)
        tracker.add("loss.", loss)

        if self.mode.is_train:
            loss.backward()

            self.optimizer.step()
            if batch_idx.is_last:
                tracker.add('model', self.model)
            self.optimizer.zero_grad()

        tracker.save()


@option(Configs.model)
def model(c: Configs):
    return Net().to(c.device)


@option(Configs.optimizer)
def _optimizer(c: Configs):
    from labml_helpers.optimizer import OptimizerConfigs
    opt_conf = OptimizerConfigs()
    opt_conf.parameters = c.model.parameters()
    return opt_conf


def main():
    conf = Configs()
    experiment.create(name='mnist_labml_helpers')
    experiment.configs(conf, {'optimizer.optimizer': 'Adam'})
    conf.set_seed.set()
    experiment.add_pytorch_models(dict(model=conf.model))
    with experiment.start():
        conf.run()


if __name__ == '__main__':
    main()
