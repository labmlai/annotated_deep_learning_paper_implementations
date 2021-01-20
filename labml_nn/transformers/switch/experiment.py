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
        self.transformer = transformer
        self.generator = nn.Linear(d_model, n_vocab)
        self.mask = None

    def __call__(self, x: torch.Tensor):
        if self.mask is None or self.mask.size(0) != len(x):
            from labml_nn.transformers.utils import subsequent_mask
            self.mask = subsequent_mask(len(x)).to(x.device)
        x = self.src_embed(x)
        # Embed the tokens (`src`) and run it through the the transformer
        res, counts, route_prob, n_dropped = self.transformer(x, self.mask)
        # Generate logits of the next token
        return self.generator(res), counts, route_prob, n_dropped


class Configs(NLPAutoRegressionConfigs):
    """
    ## Configurations

    The default configs can and will be over-ridden when we start the experiment
    """

    model: AutoregressiveModel
    transformer: Module

    d_model: int = 128
    heads: int = 4
    dropout: float = 0.0
    d_ff: int = 256
    n_layers: int = 6
    n_experts: int = 4
    load_balancing_loss_ceof = 0.01
    is_scale_prob: bool = True
    drop_tokens: bool = False
    capacity_factor: float = 1.0

    def init(self):
        super().init()
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
            # It's returning a tuple for states when using RNNs.
            # This is not implemented yet. 😜
            output, counts, route_prob, n_dropped = self.model(data)

        # Calculate and log loss
        loss = self.loss_func(output, target)
        total = counts.sum(dim=-1, keepdims=True)
        route_frac = counts / total
        route_prob = route_prob / total
        tracker.add('dropped.', total.new_tensor(n_dropped) / total)
        tracker.add('route.min.', route_frac.min())
        tracker.add('route.max.', route_frac.max())
        tracker.add('route.std.', route_frac.std())
        # for i in range(self.n_switches):
        #     tracker.add(f'route.{i}', route_frac[:, i].mean())
        load_balancing_loss = self.n_experts * (route_frac * route_prob).sum()
        tracker.add("loss.", loss)
        tracker.add("lb_loss.", loss)
        loss = loss + self.load_balancing_loss_ceof * load_balancing_loss

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
    m = AutoregressiveModel(c.n_tokens, c.d_model, c.transformer)
    return m.to(c.device)


@option(Configs.transformer)
def switch_transformer(c: Configs):
    from labml_nn.transformers.switch import SwitchTransformer, SwitchTransformerLayer, SwitchFeedForward
    from labml_nn.transformers import MultiHeadAttention

    return SwitchTransformer(
        SwitchTransformerLayer(d_model=c.d_model,
                               attn=MultiHeadAttention(c.heads, c.d_model, c.dropout),
                               feed_forward=SwitchFeedForward(capacity_factor=c.capacity_factor,
                                                              drop_tokens=c.drop_tokens,
                                                              is_scale_prob=c.is_scale_prob,
                                                              n_experts=c.n_experts,
                                                              d_model=c.d_model,
                                                              d_ff=c.d_ff,
                                                              dropout=c.dropout),
                               dropout_prob=c.dropout),
        c.n_layers)


def main():
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
                        'inner_iterations': 25})

    # Set models for saving and loading
    experiment.add_pytorch_models({'model': conf.model})

    conf.init()
    # Start the experiment
    with experiment.start():
        # `TrainValidConfigs.run`
        conf.run()


if __name__ == '__main__':
    main()
