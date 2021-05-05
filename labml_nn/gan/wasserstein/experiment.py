# We import the [simple gan experiment]((simple_mnist_experiment.html) and change the
# generator and discriminator networks
from labml import experiment

from labml.configs import calculate
from labml_nn.gan.dcgan import Configs
from labml_nn.gan.wasserstein import GeneratorLoss, DiscriminatorLoss

calculate(Configs.generator_loss, 'wasserstein', lambda c: GeneratorLoss())
calculate(Configs.discriminator_loss, 'wasserstein', lambda c: DiscriminatorLoss())


def main():
    conf = Configs()
    experiment.create(name='mnist_wassertein_dcgan', comment='test')
    experiment.configs(conf,
                       {
                           'discriminator': 'cnn',
                           'generator': 'cnn',
                           'label_smoothing': 0.01,
                           'generator_loss': 'wasserstein',
                           'discriminator_loss': 'wasserstein',
                       })
    with experiment.start():
        conf.run()


if __name__ == '__main__':
    main()
