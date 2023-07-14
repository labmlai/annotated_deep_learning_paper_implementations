"""
---
title: Transformer Auto-Regression Experiment with [Sophia-G optimizer](../../optimizers/sophia.html)
summary: >
  This trains a simple transformer model on NLP auto-regression with Sophia-G optimizer.
---

# Transformer Auto-Regression Experiment with [Sophia-G optimizer](../../optimizers/sophia.html)

This trains a simple transformer introduced in [Attention Is All You Need](https://papers.labml.ai/paper/1706.03762)
on an NLP auto-regression task (with Tiny Shakespeare dataset) with [Sophia-G optimizer](../../optimizers/sophia.html).
"""
import torch

from labml import experiment, tracker
from labml_helpers.train_valid import BatchIndex
from labml_nn.optimizers.sophia import Sophia
from labml_nn.transformers.basic.autoregressive_experiment import Configs as TransformerAutoRegressionConfigs


class Configs(TransformerAutoRegressionConfigs):
    """
    ## Configurations

    This inherits from [`Configs`](autoregressive_experiment.html)
    """

    hess_interval: int = 10

    optimizer: Sophia

    def step(self, batch: any, batch_idx: BatchIndex):
        """
        ### Training or validation step with Gauss-Newton-Bartlett (GNB) Hessian diagonal estimator
        """

        # Set training/eval mode
        self.model.train(self.mode.is_train)

        # Move data to the device
        data, target = batch[0].to(self.device), batch[1].to(self.device)

        # Estimate the Hessian diagonal every $k$ steps
        if isinstance(self.optimizer, Sophia) and self.mode.is_train and batch_idx.idx % self.hess_interval == 0:
            # Get model outputs
            output, *_ = self.model(data)

            # Create a categorical distribution from logits
            samp_dist = torch.distributions.Categorical(logits=output)
            # Sample $\hat{y}$
            y_sample = samp_dist.sample()

            # Calculate and log loss
            loss = self.loss_func(output, y_sample)
            tracker.add("loss.hess.", loss)

            # Calculate gradients
            loss.backward()
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_norm_clip)
            # Update EMA Hessian diagonal
            #
            # \begin{align}
            # \hat{h}_t &= B \cdot \nabla_\theta \hat{L} (\theta) \odot \nabla_\theta \hat{L} (\theta) \\
            # h_t &= \beta_2 h_{t-k} + (1 - \beta_2) \hat{h}_t
            # \end{align}
            self.optimizer.update_hessian(data.numel())
            # Clear the gradients
            self.optimizer.zero_grad()
        else:
            # Move data to the device
            data, target = batch[0].to(self.device), batch[1].to(self.device)

            # Update global step (number of tokens processed) when in training mode
            if self.mode.is_train:
                tracker.add_global_step(data.shape[0] * data.shape[1])

            # Whether to capture model outputs
            with self.mode.update(is_log_activations=batch_idx.is_last and self.is_log_model_activations):
                # Get model outputs.
                # It's returning a tuple for states when using RNNs.
                # This is not implemented yet. 😜
                output, *_ = self.model(data)

            # Calculate and log loss
            loss = self.loss_func(output, target)
            tracker.add("loss.", loss)

            # Calculate and log accuracy
            self.accuracy(output, target)
            self.accuracy.track()

            self.other_metrics(output, target)

            # Train the model
            if self.mode.is_train:
                # Calculate gradients
                loss.backward()
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_norm_clip)
                # Take optimizer step
                self.optimizer.step()
                # Log the model parameters and gradients on last batch of every epoch
                if batch_idx.is_last and self.is_log_model_params_grads:
                    tracker.add('model', self.model)
                # Clear the gradients
                self.optimizer.zero_grad()

            # Save the tracked metrics
            tracker.save()


def main():
    # Create experiment
    experiment.create(name="transformer")
    # Create configs
    conf = Configs()
    # Override configurations
    experiment.configs(conf, {
        # Use character level tokenizer
        'tokenizer': 'character',
        # Prompt separator is blank
        'prompt_separator': '',
        # Starting prompt for sampling
        'prompt': 'It is ',
        # Use Tiny Shakespeare dataset
        'text': 'tiny_shakespeare',

        # Use a context size of $256$
        'seq_len': 512,
        # Train for 32 epochs
        'epochs': 32,
        # Batch size $32$
        'batch_size': 16,
        # Switch between training and validation for $10$ times
        # per epoch
        'inner_iterations': 10,

        # Model size
        'd_model': 256,
        'transformer.n_heads': 16,
        'transformer.ffn.d_ff': 1024,

        # Use [Sophia optimizer](../../optimizers/sophia.html)
        'optimizer.optimizer': 'Sophia',
        'optimizer.learning_rate': 3e-4,
        'optimizer.rho': 0.03,
    })

    # Set models for saving and loading
    experiment.add_pytorch_models({'model': conf.model})

    # Start the experiment
    with experiment.start():
        # Run training
        conf.run()


#
if __name__ == '__main__':
    main()
