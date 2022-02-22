import torch
from torch import nn

from labml.logger import inspect
from labml_nn.transformers.mha import MultiHeadAttention


class RotaryPositionalEmbeddings(nn.Module):
    def __init__(self, d_k: int, base: int = 10_000):
        super().__init__()
        self.theta = nn.Parameter(1. / (base ** (torch.arange(0, d_k, 2).float() / d_k)), requires_grad=False)

    def forward(self, x: torch.Tensor):
        seq_len, batch_size, n_heads, d_k = x.shape

        seq_idx = torch.arange(seq_len, device=x.device).type_as(self.theta)

        idx_theta = torch.einsum('n,d->nd', seq_idx, self.theta)

        idx_theta2 = torch.cat([idx_theta, idx_theta], dim=1)

        h_d_k = d_k // 2
        neg_half_x = torch.cat([-x[:, :, :, h_d_k:], x[:, :, :, :h_d_k]], dim=-1)

        rx = (x * idx_theta2.cos()[:, None, None, :]) + (neg_half_x * idx_theta2.sin()[:, None, None, :])

        return rx


class RotaryPEMultiHeadAttention(MultiHeadAttention):
    def __init__(self, heads: int, d_model: int, dropout_prob: float = 0.1):
        # The linear transformations do not need a bias since we
        # explicitly include it when calculating scores.
        # However having a bias for `value` might make sense.
        super().__init__(heads, d_model, dropout_prob, bias=False)

        self.query_rotary_pe = RotaryPositionalEmbeddings(self.d_k)
        self.key_rotary_pe = RotaryPositionalEmbeddings(self.d_k)

    def get_scores(self, query: torch.Tensor, key: torch.Tensor):
        """
        ### Calculate scores between queries and keys

        This method can be overridden for other variations like relative attention.
        """

        # Calculate $Q K^\top$ or $S_{ijbh} = \sum_d Q_{ibhd} K_{jbhd}$
        return torch.einsum('ibhd,jbhd->ijbh', self.query_rotary_pe(query), self.key_rotary_pe(key))


def _test_rotary():
    x = torch.tensor([[1, 2, 3, 4], [4, 5, 6, 7], [7, 8, 9, 10]], dtype=torch.float)
    x = x[:, None, None, :]
    inspect(x)

    rotary_pe = RotaryPositionalEmbeddings(3)
    inspect(rotary_pe(x))


if __name__ == '__main__':
    _test_rotary()
