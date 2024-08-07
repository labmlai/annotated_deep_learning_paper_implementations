import torch
import torch.nn as nn
from labml_nn.lora import Linear, Embedding


class FFN(nn.Module):
    def __init__(self, dim: int, n_embed: int, r: int):
        super().__init__()
        self.c_fc = Linear(n_embed, dim, r=r, bias=True)
        self.c_proj = Linear(dim, n_embed, r=r, bias=True)
        self.act = nn.functional.gelu

    def forward(self, hidden_states):
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        return hidden_states


class MultiHeadAttention(nn.Module):
    def __init__(self, n_embed: int, r: int):
        super().__init__()
        self.embed_dim = n_embed
        self.num_heads = n_embed
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim

        self.c_att = Linear(n_embed, n_embed * 3, r=r, bias=True)
        self.c_proj = Linear(n_embed, n_embed, r=r, bias=True)

    def _split_heads(self, tensor, num_heads, attn_head_size):
        """
        Splits hidden_size dim into attn_head_size and num_heads
        """
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def forward(self, hidden_states):
        batch_size, seq_length, _ = hidden_states.size()

        query, key, value = self.c_att(hidden_states).split(self.split_size, dim=2)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,  # for the triangular mask
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_length, self.embed_dim)

        attn_output = self.c_proj(attn_output)

        return attn_output


class Block(nn.Module):
    def __init__(self, n_embed: int, layer_norm_epsilon: float, r: int):
        super().__init__()
        self.pre_norm = nn.LayerNorm(n_embed, eps=layer_norm_epsilon)
        self.attn = MultiHeadAttention(n_embed, r)
        self.post_norm = nn.LayerNorm(n_embed, eps=layer_norm_epsilon)
        self.ffn = FFN(n_embed * 4, n_embed, r)

    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = self.pre_norm(hidden_states)

        attn_output = self.attn(hidden_states)

        hidden_states = attn_output + residual
        residual = hidden_states
        hidden_states = self.post_norm(hidden_states)
        feed_forward_output = self.ffn(hidden_states)
        hidden_states = feed_forward_output + residual

        return hidden_states


class GPTModel(nn.Module):
    def __init__(self, layer_norm_epsilon: float, n_embd: int, n_layer: int, n_positions: int,
                 vocab_size: int, r: int, device: torch.device):
        super().__init__()

        self.token_embedding = Embedding(vocab_size, n_embd, r=r)
        self.position_embedding = Embedding(n_positions, n_embd, r=r)

        self.blocks = nn.ModuleList([Block(n_embd, layer_norm_epsilon, r=r)
                                     for _ in range(n_layer)])

        self.final_norm = nn.LayerNorm(n_embd, eps=layer_norm_epsilon)

        self.lm_head = Linear(n_embd, vocab_size, r=r, bias=False)

        self.device = device

    def forward(self, input_ids):
        batch_size, input_shape = input_ids.size()

        token_embeddings = self.token_embedding(input_ids)  # B T C
        position_ids = torch.arange(input_shape, device=self.device)  # T C
        position_embeddings = self.position_embedding(position_ids)  # B T C

        hidden_states = token_embeddings + position_embeddings

        for block in self.blocks:
            hidden_states = block(hidden_states)

        hidden_states = self.final_norm(hidden_states)

        logits = self.lm_head(hidden_states)

        return logits
