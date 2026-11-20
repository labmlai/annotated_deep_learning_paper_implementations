"""Golden-vector properties for rotary positional embeddings."""

import unittest

import torch

from labml_nn.transformers.rope import RotaryPositionalEmbeddings


class TestRotaryPositionalEmbeddings(unittest.TestCase):
    D = 32
    BASE = 10000
    BATCH = 2
    SEQ = 8
    HEADS = 2

    def setUp(self):
        torch.manual_seed(0)
        self.rope = RotaryPositionalEmbeddings(self.D, base=self.BASE)
        self.x = torch.randn(self.SEQ, self.BATCH, self.HEADS, self.D, dtype=torch.float32)

    def test_l2_norm_preserved_per_token(self):
        y = self.rope(self.x)
        x_norm = torch.linalg.vector_norm(self.x, dim=-1)
        y_norm = torch.linalg.vector_norm(y, dim=-1)
        torch.testing.assert_close(y_norm, x_norm, rtol=1e-5, atol=1e-5)

    def test_relative_position_scores_match_at_offset(self):
        q = torch.zeros(self.SEQ, 1, 1, self.D, dtype=torch.float32)
        k = torch.zeros(self.SEQ, 1, 1, self.D, dtype=torch.float32)
        q[..., 0] = 1.0
        k[..., 0] = 1.0
        q_rot = self.rope(q).squeeze(1).squeeze(1)
        k_rot = self.rope(k).squeeze(1).squeeze(1)
        score_03 = (q_rot[0] * k_rot[3]).sum()
        score_25 = (q_rot[2] * k_rot[5]).sum()
        torch.testing.assert_close(score_03, score_25, rtol=1e-5, atol=1e-5)

    def test_invertible_via_negated_sin(self):
        y = self.rope(self.x)
        inv = RotaryPositionalEmbeddings(self.D, base=self.BASE)
        inv.cos_cached = self.rope.cos_cached
        inv.sin_cached = -self.rope.sin_cached
        recovered = inv(y)
        torch.testing.assert_close(recovered, self.x, rtol=1e-5, atol=1e-5)

    def test_neg_half_pairs_first_half_with_second(self):
        x_rope = self.x[..., : self.D]
        half = self.rope._neg_half(x_rope)
        d2 = self.D // 2
        torch.testing.assert_close(half[..., :d2], -x_rope[..., d2:], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(half[..., d2:], x_rope[..., :d2], rtol=1e-5, atol=1e-5)

    def test_cache_rebuilds_when_sequence_grows(self):
        self.rope(self.x)
        cached_id = id(self.rope.cos_cached)
        x_long = torch.randn(16, self.BATCH, self.HEADS, self.D, dtype=torch.float32)
        self.rope(x_long)
        self.assertGreaterEqual(self.rope.cos_cached.shape[0], 16)
        self.assertNotEqual(id(self.rope.cos_cached), cached_id)
        if torch.cuda.is_available():
            rope_cuda = RotaryPositionalEmbeddings(self.D, base=self.BASE).cuda()
            x_cuda = self.x.cuda()
            rope_cuda(x_cuda)
            cuda_id = id(rope_cuda.cos_cached)
            rope_cuda(x_long.cuda())
            self.assertGreaterEqual(rope_cuda.cos_cached.shape[0], 16)
            self.assertNotEqual(id(rope_cuda.cos_cached), cuda_id)


if __name__ == '__main__':
    unittest.main()
