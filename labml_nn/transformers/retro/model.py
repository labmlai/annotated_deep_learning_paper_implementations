import math
from typing import Set

import torch
from labml.logger import inspect
from torch import nn


class SelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k, is_causal: bool):
        super().__init__()

        self.is_causal = is_causal
        self.n_heads = n_heads
        self.d_k = d_k
        self.scale = 1 / math.sqrt(self.d_k)

        self.query = nn.Linear(d_model, n_heads * d_k)
        self.key = nn.Linear(d_model, n_heads * d_k)
        self.value = nn.Linear(d_model, n_heads * d_k)

        self.norm = nn.LayerNorm(d_model)

        self.softmax = nn.Softmax(dim=-1)

        self.output = nn.Linear(n_heads * d_k, d_model)

    def mask_attention(self, attn: torch.Tensor):
        if not self.is_causal:
            return attn

        mask = torch.tril(attn.new_ones(attn.shape[-2:]))
        return attn.masked_fill(mask == 0, float('-inf'))

    def forward(self, h: torch.Tensor):
        """
        h [batch, seq, d_model]
        """

        h_res = h
        h = self.norm(h)

        mh_shape = (*h.shape[:-1], self.n_heads, self.d_k)
        q = self.query(h).view(mh_shape)
        k = self.key(h).view(mh_shape)
        v = self.value(h).view(mh_shape)

        attn = torch.einsum('bihd,bjhd->bhij', q, k)
        attn = attn * self.scale

        attn = self.mask_attention(attn)

        attn = self.softmax(attn)

        h = torch.einsum("bhij,bjhd->bihd", attn, v)

        h = h.reshape(*h.shape[:-2], -1)

        h = self.output(h)

        return h + h_res


class CrossAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k):
        super().__init__()

        self.n_heads = n_heads
        self.d_k = d_k
        self.scale = 1 / math.sqrt(self.d_k)

        self.query = nn.Linear(d_model, n_heads * d_k)
        self.key = nn.Linear(d_model, n_heads * d_k)
        self.value = nn.Linear(d_model, n_heads * d_k)

        self.norm = nn.LayerNorm(d_model)

        self.softmax = nn.Softmax(dim=-1)

        self.output = nn.Linear(n_heads * d_k, d_model)

    def forward(self, e: torch.Tensor, h: torch.Tensor):
        """
        e [batch_size, chunks, neighbors, seq, d_model]
        h [batch_size, chunks, seq, d_model]
        """

        e_res = e
        e = self.norm(e)

        q = self.query(e).view(*e.shape[:-1], self.n_heads, self.d_k)
        k = self.key(h).view(*h.shape[:-1], self.n_heads, self.d_k)
        v = self.value(h).view(*h.shape[:-1], self.n_heads, self.d_k)

        attn = torch.einsum('bcnihd,bcjhd->bcnhij', q, k)
        attn = attn * self.scale

        attn = self.softmax(attn)

        e = torch.einsum("bcnhij,bcjhd->bcnihd", attn, v)

        e = e.reshape(*e.shape[:-2], -1)

        e = self.output(e)

        return e + e_res


class ChunkedCrossAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k, chunk_len):
        super().__init__()

        self.chunk_len = chunk_len
        self.n_heads = n_heads
        self.d_k = d_k
        self.scale = 1 / math.sqrt(self.d_k)

        self.query = nn.Linear(d_model, n_heads * d_k)
        self.key = nn.Linear(d_model, n_heads * d_k)
        self.value = nn.Linear(d_model, n_heads * d_k)

        self.norm = nn.LayerNorm(d_model)

        self.softmax = nn.Softmax(dim=-1)

        self.output = nn.Linear(d_model, d_model)

    def forward(self, h: torch.Tensor, e: torch.Tensor):
        """
        h [batch_size, seq, d_model]
        e [batch_size, chunks, neighbors, seq, d_model]
        """

        batch_size, chunks, neighbors, neighbor_len, d_model = e.shape

        if chunks == 0:
            return h

        h_res = h

        h = h[:, self.chunk_len - 1:]
        h = self.norm(h)
        if h.shape[1] < chunks * self.chunk_len:
            h = torch.cat((h, h.new_zeros(batch_size, chunks * self.chunk_len - h.shape[1], d_model)), dim=1)
        h = h.reshape(batch_size, chunks, self.chunk_len, d_model)

        q = self.query(h).view(*h.shape[:-1], self.n_heads, self.d_k)
        k = self.key(e).view(*e.shape[:-1], self.n_heads, self.d_k)
        v = self.value(e).view(*e.shape[:-1], self.n_heads, self.d_k)

        attn = torch.einsum('bcihd,bcnjhd->bchinj', q, k)
        attn = attn * self.scale

        attn = self.softmax(attn.view(*attn.shape[:-2], -1)).view(attn.shape)

        h = torch.einsum("bchinj,bcnjhd->bcihd", attn, v)

        h = h.reshape(batch_size, chunks * self.chunk_len, d_model)

        h = self.output(h)

        h = torch.cat((h.new_zeros(batch_size, self.chunk_len - 1, d_model), h), dim=1)

        return h[:, :h_res.shape[1], :] + h_res


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()

        self.lin1 = nn.Linear(d_model, d_ff)
        self.lin2 = nn.Linear(d_ff, d_model)

        self.norm = nn.LayerNorm(d_model)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor):
        x_res = x
        x = self.norm(x)
        x = self.lin1(x)
        x = self.act(x)
        x = self.lin2(x)

        return x + x_res


class Encoder(nn.Module):
    def __init__(self, chunk_len: int, n_layers: int, ca_layers: Set[int],
                 d_model: int, n_heads: int, d_k: int, d_ff: int):
        super().__init__()
        self.ca_layers = ca_layers
        self.chunk_len = chunk_len
        self.ca = nn.ModuleList([CrossAttention(d_model, n_heads, d_k) for _ in range(len(ca_layers))])
        self.attn = nn.ModuleList([SelfAttention(d_model, n_heads, d_k, is_causal=False) for _ in range(n_layers)])
        self.ffw = nn.ModuleList([FeedForward(d_model, d_ff) for _ in range(n_layers)])

        self.norm_h = nn.LayerNorm(d_model)

    def forward(self, e: torch.Tensor, h: torch.Tensor):
        """
        e [batch_size, chunks, neighbors, seq, d_model]
        h [batch_size, seq, d_model]
        """

        batch_size, chunks, neighbors, neighbor_len, d_model = e.shape
        h_split = h[:, :self.chunk_len * chunks, :].reshape(batch_size, chunks, self.chunk_len, d_model)
        h_split = self.norm_h(h_split)

        p_ca = 0
        for p in range(len(self.attn)):
            e = self.attn[p](e.view(-1, neighbor_len, d_model)).view(e.shape)

            if p in self.ca_layers:
                e = self.ca[p_ca](e, h_split)
                p_ca += 1

            e = self.ffw[p](e)

        return e


class Model(nn.Module):
    def __init__(self, n_vocab: int, d_model: int, n_layers: int, ca_layers: Set[int], chunk_length: int,
                 n_heads: int, d_k: int, d_ff: int,
                 encoder: Encoder):
        super().__init__()

        self.ca_layers = ca_layers
        self.emb = nn.Embedding(n_vocab, d_model)
        self.encoder = encoder
        self.cca = nn.ModuleList(
            [ChunkedCrossAttention(d_model, n_heads, d_k, chunk_length) for _ in range(len(ca_layers))])
        self.attn = nn.ModuleList([SelfAttention(d_model, n_heads, d_k, is_causal=True) for _ in range(n_layers)])
        self.ffw = nn.ModuleList([FeedForward(d_model, d_ff) for _ in range(n_layers)])
        self.read = nn.Linear(d_model, n_vocab)

        self.norm_e = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, ret: torch.Tensor):
        """
        x [batch_size, seq]
        e [batch_size, chunks, neighbors, seq]
        """

        h = self.emb(x)

        ret_emb = self.emb(ret)

        p_ca = 0
        for p in range(len(self.attn)):
            h = self.attn[p](h)

            if p == min(self.ca_layers):
                e = self.encoder(ret_emb, h)
                e = self.norm_e(e)

            if p in self.ca_layers:
                h = self.cca[p_ca](h, e)
                p_ca += 1

            h = self.ffw[p](h)

        return self.read(h)


def _test():
    chunk_length = 4
    d_model = 8
    d_ff = 32
    n_heads = 2
    d_k = 4
    m = Model(5, d_model, 6, {2, 5}, chunk_length, n_heads, d_k, d_ff,
              encoder=Encoder(chunk_length, 2, {1}, d_model, n_heads, d_k, d_ff))

    x = [1, 2, 4, 4, 0, 1, 2, 3, 4, 3]
    ret = [
        [[0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1]],
        [[0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1]],
    ]
    res = m(torch.tensor([x]), torch.tensor([ret]))

    inspect(res)


if __name__ == '__main__':
    _test()
