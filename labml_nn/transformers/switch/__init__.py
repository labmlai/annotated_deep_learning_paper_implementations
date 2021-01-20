import torch
from torch import nn

from labml_helpers.module import Module
from labml_nn.transformers.mha import MultiHeadAttention
from labml_nn.transformers.models import FeedForward
from labml_nn.utils import clone_module_list


class SwitchFeedForward(Module):
    """
    ## Position-wise feed-forward network with hidden layer
    """

    def __init__(self, *,
                 capacity_factor: float,
                 drop_tokens: bool,
                 is_scale_prob: bool,
                 n_switches: int,
                 d_model: int,
                 d_ff: int,
                 dropout: float = 0.1):

        super().__init__()
        self.capacity_factor = capacity_factor
        self.is_scale_prob = is_scale_prob
        self.units = nn.ModuleList([FeedForward(d_model, d_ff, dropout) for _ in range(n_switches)])
        self.switch = nn.Linear(d_model, n_switches)
        self.softmax = nn.Softmax(dim=-1)
        self.n_switches = n_switches
        self.drop_tokens = drop_tokens

    def __call__(self, x: torch.Tensor):
        seq_len, bs, d_model = x.shape
        x = x.view(-1, d_model)

        route_prob = self.softmax(self.switch(x))
        route_prob_max, routes = torch.max(route_prob, dim=-1)

        if self.is_scale_prob:
            factor = route_prob_max
        else:
            factor = route_prob_max / route_prob_max.detach()
        x = x * factor.view(-1, 1)

        # Get indexes of vectors going to each route
        indexes_list = [torch.eq(routes, i).nonzero(as_tuple=True)[0] for i in range(self.n_switches)]

        # Tensor to store outputs
        final_output = x.new_zeros(x.shape)

        # Capacity of a route
        capacity = int(self.capacity_factor * len(x) / self.n_switches)
        # Number of tokens going to each route
        counts = x.new_tensor([len(indexes_list[i]) for i in range(self.n_switches)])

        # Drop tokens
        dropped = []
        if self.drop_tokens:
            for i in range(self.n_switches):
                if len(indexes_list[i]) <= capacity:
                    continue
                dropped.append(indexes_list[i][capacity:])
                indexes_list[i] = indexes_list[i][:capacity]

        route_outputs = [self.units[i](x[indexes_list[i], :]) for i in range(self.n_switches)]

        # Assign to final output
        for i in range(self.n_switches):
            final_output[indexes_list[i], :] = route_outputs[i]

        # Pass through the dropped tokens
        if dropped:
            dropped = torch.cat(dropped)
            final_output[dropped, :] = x[dropped, :]

        # Change the shape of the final output
        final_output = final_output.view(seq_len, bs, d_model)

        return final_output, counts, route_prob.sum(0), len(dropped)


class SwitchTransformerLayer(Module):
    def __init__(self, *,
                 d_model: int,
                 attn: MultiHeadAttention,
                 feed_forward: SwitchFeedForward,
                 dropout_prob: float):
        super().__init__()
        self.size = d_model
        self.attn = attn
        self.feed_forward = feed_forward
        self.dropout = nn.Dropout(dropout_prob)
        self.norm_self_attn = nn.LayerNorm([d_model])
        self.norm_ff = nn.LayerNorm([d_model])

    def __call__(self, *,
                 x: torch.Tensor,
                 mask: torch.Tensor):
        # Normalize the vectors before doing self attention
        z = self.norm_self_attn(x)
        # Run through self attention, i.e. keys and values are from self
        self_attn = self.attn(query=z, key=z, value=z, mask=mask)
        # Add the self attention results
        x = x + self.dropout(self_attn)

        # Normalize for feed-forward
        z = self.norm_ff(x)
        # Pass through the feed-forward network
        ff, counts, route_prob, n_dropped = self.feed_forward(z)
        # Add the feed-forward results back
        x = x + self.dropout(ff)

        return x, counts, route_prob, n_dropped


class SwitchTransformer(Module):
    """
    <a id="Encoder">
    ## Transformer Encoder
    </a>
    """

    def __init__(self, layer: SwitchTransformerLayer, n_layers: int):
        super().__init__()
        # Make copies of the transformer layer
        self.layers = clone_module_list(layer, n_layers)
        self.norm = nn.LayerNorm([layer.size])

    def __call__(self, x: torch.Tensor, mask: torch.Tensor):
        # Run through each transformer layer
        counts, route_prob, n_dropped = [], [], []
        for layer in self.layers:
            x, f, p, n_d = layer(x=x, mask=mask)
            counts.append(f)
            route_prob.append(p)
            n_dropped.append(n_d)
        # Finally, normalize the vectors
        return self.norm(x), torch.stack(counts), torch.stack(route_prob), n_dropped
