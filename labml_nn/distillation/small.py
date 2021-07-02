import torch.nn as nn

from labml import experiment
from labml.configs import option
from labml_nn.experiments.cifar10 import CIFAR10Configs, CIFAR10VGGModel
from labml_nn.normalization.batch_norm import BatchNorm


class Configs(CIFAR10Configs):
    pass


class SmallModel(CIFAR10VGGModel):
    """
    ### VGG model for CIFAR-10 classification

    This derives from the [generic VGG style architecture](../experiments/cifar10.html).
    """

    def conv_block(self, in_channels, out_channels) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            BatchNorm(out_channels, track_running_stats=False),
            nn.ReLU(inplace=True),
        )

    def __init__(self):
        super().__init__([[32, 32], [64, 64], [128], [128], [128]])


@option(Configs.model)
def _small_model(c: Configs):
    """
    ### Create model
    """
    return SmallModel().to(c.device)


def main():
    # Create experiment
    experiment.create(name='cifar10', comment='small model')
    # Create configurations
    conf = Configs()
    # Load configurations
    experiment.configs(conf, {
        'device.cuda_device': 1,
        'optimizer.optimizer': 'Adam',
        'optimizer.learning_rate': 2.5e-4,
    })
    experiment.add_pytorch_models({'model': conf.model})

    # Start the experiment and run the training loop
    with experiment.start():
        conf.run()


#
if __name__ == '__main__':
    main()
