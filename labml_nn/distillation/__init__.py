import torch
from torch import nn
import torch.nn.functional

from labml_helpers.train_valid import BatchIndex

from labml import experiment, tracker
from labml.configs import option
from labml_nn.distillation.large import LargeModel, Configs as LargeConfigs
from labml_nn.distillation.small import SmallModel
from labml_nn.experiments.cifar10 import CIFAR10Configs


class Configs(CIFAR10Configs):
    large: LargeModel
    kl_div_loss = nn.KLDivLoss(log_target=True)
    kl_div_loss_weight = 16 * 5
    nll_loss_weight = 0.5
    temperature = 5

    def step(self, batch: any, batch_idx: BatchIndex):
        """
        ### Training or validation step
        """

        # Training/Evaluation mode
        self.model.train(self.mode.is_train)
        self.large.eval()

        # Move data to the device
        data, target = batch[0].to(self.device), batch[1].to(self.device)

        # Update global step (number of samples processed) when in training mode
        if self.mode.is_train:
            tracker.add_global_step(len(data))

        with torch.no_grad():
            large_logits = self.large(data)

        # Whether to capture model outputs
        with self.mode.update(is_log_activations=batch_idx.is_last):
            # Get model outputs.
            output = self.model(data)

        # Calculate and log loss
        kl_div_loss = self.kl_div_loss(nn.functional.log_softmax(output / self.temperature, dim=-1),
                                       nn.functional.log_softmax(large_logits / self.temperature, dim=-1))
        nll_loss = self.loss_func(output, target)
        loss = self.kl_div_loss_weight * kl_div_loss + self.nll_loss_weight * nll_loss
        tracker.add("loss.kl_div.", kl_div_loss)
        tracker.add("loss.nll", nll_loss)
        tracker.add("loss.", loss)

        # Calculate and log accuracy
        self.accuracy(output, target)
        self.accuracy.track()

        # Train the model
        if self.mode.is_train:
            # Calculate gradients
            loss.backward()
            # Take optimizer step
            self.optimizer.step()
            # Log the model parameters and gradients on last batch of every epoch
            if batch_idx.is_last:
                tracker.add('model', self.model)
            # Clear the gradients
            self.optimizer.zero_grad()

        # Save the tracked metrics
        tracker.save()


@option(Configs.large)
def _large_model(c: Configs):
    """
    ### Create model
    """
    return LargeModel().to(c.device)


@option(Configs.model)
def _small_student_model(c: Configs):
    """
    ### Create model
    """
    return SmallModel().to(c.device)


def get_saved_model(run_uuid: str, checkpoint: int):
    experiment.evaluate()
    conf = LargeConfigs()
    experiment.configs(conf, experiment.load_configs(run_uuid))
    experiment.add_pytorch_models({'model': conf.model})
    experiment.load(run_uuid, checkpoint)
    experiment.start()

    return conf.model


def main(run_uuid: str, checkpoint: int):
    large_model = get_saved_model(run_uuid, checkpoint)
    # Create experiment
    experiment.create(name='distillation', comment='cifar10')
    # Create configurations
    conf = Configs()
    conf.large = large_model
    # Load configurations
    experiment.configs(conf, {
        'optimizer.optimizer': 'Adam',
        'optimizer.learning_rate': 2.5e-4,
        'model': '_small_student_model',
    })
    experiment.add_pytorch_models({'model': conf.model})
    experiment.load(None, None)
    # Start the experiment and run the training loop
    with experiment.start():
        conf.run()


#
if __name__ == '__main__':
    main('85b4be18da6811eba4852143f828ac51', 1_000_000)
