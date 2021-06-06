from typing import List

import torch


class MLM:
    def __init__(self, *,
                 masking_prob: float, randomize_prob: float, no_change_prob: float,
                 padding_token: int, mask_token: int, no_mask_tokens: List[int],
                 n_tokens: int,
                 ):
        self.n_tokens = n_tokens
        self.no_change_prob = no_change_prob
        self.randomize_prob = randomize_prob
        self.masking_prob = masking_prob
        self.no_mask_tokens = no_mask_tokens + [padding_token, mask_token]
        self.padding_token = padding_token
        self.mask_token = mask_token

    def __call__(self, x: torch.Tensor):
        full_mask = torch.rand(x.shape, device=x.device) < self.masking_prob
        for t in self.no_mask_tokens:
            full_mask &= x != t

        random_token_mask = full_mask & (torch.rand(x.shape, device=x.device) < self.randomize_prob)
        unchanged = full_mask & (torch.rand(x.shape, device=x.device) < self.no_change_prob)
        mask = full_mask & ~random_token_mask & ~unchanged
        random_token_idx = torch.nonzero(random_token_mask, as_tuple=True)
        random_tokens = torch.randint(0, self.n_tokens, (len(random_token_idx[0]),), device=x.device)

        y = x.clone()

        x.masked_fill_(mask, self.mask_token)
        x[random_token_idx] = random_tokens
        y.masked_fill_(~full_mask, self.padding_token)

        return x, y
