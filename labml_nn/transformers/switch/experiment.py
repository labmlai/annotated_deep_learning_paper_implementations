"""
---
title: Switch Transformer Experiment
summary: This experiment trains a small switch transformer on tiny Shakespeare dataset.
---

# Switch Transformer Experiment

This is an annotated PyTorch experiment to train a switch transformer.
"""

import torch
import torch.nn as nn

from labml import experiment, tracker
from labml.configs import option
from labml_helpers.module import Module
from labml_helpers.train_valid import BatchIndex
from labml_nn.experiments.nlp_autoregression import NLPAutoRegressionConfigs


class AutoregressiveModel(Module):
    """
    ## Auto regressive model
    """

    def __init__(self, n_vocab: int, d_model: int, transformer: Module):
        super().__init__()
        # Token embedding module
        self.src_embed = nn.Embedding(n_vocab, d_model)
        # Transformer
        self.transformer = transformer
        # Final layer
        self.generator = nn.Linear(d_model, n_vocab)
        self.mask = None

    def forward(self, x: torch.Tensor):
        # Initialize the subsequent mask
        if self.mask is None or self.mask.size(0) != len(x):
            from labml_nn.transformers.utils import subsequent_mask
            self.mask = subsequent_mask(len(x)).to(x.device)
        # Token embeddings
        x = self.src_embed(x)
        # Run it through the transformer
        res, counts, route_prob, n_dropped = self.transformer(x, self.mask)
        # Generate logits of the next token
        res = self.generator(res)
        #
        return res, counts, route_prob, n_dropped


class Configs(NLPAutoRegressionConfigs):
    """
    ## Configurations

    The default configs can and will be over-ridden when we start the experiment
    """

    model: AutoregressiveModel
    transformer: Module

    # Token embedding size
    d_model: int = 128
    # Number of attention heads
    heads: int = 4
    # Dropout probability
    dropout: float = 0.0
    # Number of features in FFN hidden layer
    d_ff: int = 256
    # Number of transformer layers
    n_layers: int = 6
    # Number of experts
    n_experts: int = 4
    # Load balancing coefficient
    load_balancing_loss_ceof = 0.01
    # Whether to scale the chosen expert outputs by the routing probability
    is_scale_prob: bool = True
    # Whether to drop tokens
    drop_tokens: bool = False
    # Capacity factor to determine capacity of each model
    capacity_factor: float = 1.0

    def init(self):
        super().init()
        # Initialize tracking indicators
        tracker.set_scalar("lb_loss.*", False)
        tracker.set_scalar("route.*", False)
        tracker.set_scalar("dropped.*", False)

    def step(self, batch: any, batch_idx: BatchIndex):
        """
        ### Training or validation step
        """

        # Move data to the device
        data, target = batch[0].to(self.device), batch[1].to(self.device)

        # Update global step (number of tokens processed) when in training mode
        if self.mode.is_train:
            tracker.add_global_step(data.shape[0] * data.shape[1])

        # Whether to capture model outputs
        with self.mode.update(is_log_activations=batch_idx.is_last):
            # Get model outputs.
            output, counts, route_prob, n_dropped = self.model(data)

        # Calculate and cross entropy loss
        cross_entropy_loss = self.loss_func(output, target)
        # Total number of tokens processed, $T$, in the current batch $\mathscr{B}$
        total = counts.sum(dim=-1, keepdims=True)
        # Fraction of tokens routed to each expert
        # $$f_i = \frac{1}{T} \sum_{x \in \mathscr{B}} \unicode{x1D7D9} \{ \mathop{argmax} p(x), i \}$$
        # $f_i$ is the count of tokens where the argmax of $p(x)$ is equal to $i$.
        route_frac = counts / total
        # Mean routing probability
        # $$P_i = \frac{1}{T} \sum_{x \in \mathscr{B}} p_i (x)$$
        route_prob = route_prob / total
        # Load balancing loss
        # $$\mathscr{L} = N \sum_{i=1}^N f_i \cdot P_i$$
        load_balancing_loss = self.n_experts * (route_frac * route_prob).sum()

        # Track stats
        tracker.add('dropped.', total.new_tensor(n_dropped) / total)
        tracker.add('route.min.', route_frac.min())
        tracker.add('route.max.', route_frac.max())
        tracker.add('route.std.', route_frac.std())
        tracker.add("loss.", cross_entropy_loss)
        tracker.add("lb_loss.", load_balancing_loss)

        # Combined loss.
        # The load balancing loss is multiplied by a coefficient $\alpha$ which is
        # set to something small like $\alpha = 0.01$.
        loss = cross_entropy_loss + self.load_balancing_loss_ceof * load_balancing_loss

        # Calculate and log accuracy
        self.accuracy(output, target)
        self.accuracy.track()

        # Train the model
        if self.mode.is_train:
            # Calculate gradients
            loss.backward()
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_norm_clip)
            # Take optimizer step
            self.optimizer.step()
            # Log the model parameters and gradients on last batch of every epoch
            if batch_idx.is_last:
                tracker.add('model', self.model)
            # Clear the gradients
            self.optimizer.zero_grad()

        # Save the tracked metrics
        tracker.save()


@option(Configs.model)
def autoregressive_model(c: Configs):
    """
    ### Initialize the auto-regressive model
    """
    m = AutoregressiveModel(c.n_tokens, c.d_model, c.transformer)
    return m.to(c.device)


@option(Configs.transformer)
def switch_transformer(c: Configs):
    """
    ### Initialize the switch transformer
    """
    from labml_nn.transformers.switch import SwitchTransformer, SwitchTransformerLayer, SwitchFeedForward
    from labml_nn.transformers import MultiHeadAttention
    from labml_nn.transformers.feed_forward import FeedForward

    return SwitchTransformer(
        SwitchTransformerLayer(d_model=c.d_model,
                               attn=MultiHeadAttention(c.heads, c.d_model, c.dropout),
                               feed_forward=SwitchFeedForward(capacity_factor=c.capacity_factor,
                                                              drop_tokens=c.drop_tokens,
                                                              is_scale_prob=c.is_scale_prob,
                                                              n_experts=c.n_experts,
                                                              expert=FeedForward(c.d_model, c.d_ff, c.dropout),
                                                              d_model=c.d_model),
                               dropout_prob=c.dropout),
        c.n_layers)


def main():
    """
    ### Run the experiment
    """
    # Create experiment
    experiment.create(name="switch_transformer", comment='')
    # Create configs
    conf = Configs()
    # Load configurations
    experiment.configs(conf,
                       # A dictionary of configurations to override
                       {'tokenizer': 'character',
                        'text': 'tiny_shakespeare',
                        'optimizer.learning_rate': 1.,
                        'optimizer.optimizer': 'Noam',
                        'prompt': 'It is',
                        'prompt_separator': '',

                        'transformer': 'switch_transformer',
                        'is_scale_prob': False,
                        'n_experts': 4,

                        'drop_tokens': True,
                        'capacity_factor': 1.2,

                        'train_loader': 'shuffled_train_loader',
                        'valid_loader': 'shuffled_valid_loader',

                        'seq_len': 64,
                        'epochs': 128,
                        'batch_size': 32,
                        'inner_iterations': 25,
                        })

    # Set models for saving and loading
    experiment.add_pytorch_models({'model': conf.model})

    # Start the experiment
    with experiment.start():
        # `TrainValidConfigs.run`
        conf.run()


#
if __name__ == '__main__':
    main()
