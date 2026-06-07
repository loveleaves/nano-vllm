"""
RotaryEmbedding (RoPE) 单元测试
"""
import pytest
import torch

from nanovllm.layers.rotary_embedding import RotaryEmbedding, apply_rotary_emb, get_rope


class TestApplyRotaryEmb:

    @pytest.mark.unit
    def test_identity_when_cos_one_sin_zero(self):
        head_dim = 8
        N = 4
        x = torch.randn(N, head_dim)
        cos = torch.ones(N, head_dim // 2)
        sin = torch.zeros(N, head_dim // 2)
        y = apply_rotary_emb(x, cos, sin)
        assert torch.allclose(x.float(), y.float(), atol=1e-6)

    @pytest.mark.unit
    def test_output_shape_preserved(self):
        x = torch.randn(6, 16)
        cos = torch.randn(6, 8)
        sin = torch.randn(6, 8)
        y = apply_rotary_emb(x, cos, sin)
        assert y.shape == x.shape


class TestRotaryEmbedding:

    @pytest.mark.unit
    def test_output_shape(self):
        rope = RotaryEmbedding(head_size=8, rotary_dim=8,
                               max_position_embeddings=128, base=10000)
        positions = torch.arange(5)
        q = torch.randn(5, 2, 8)
        k = torch.randn(5, 2, 8)
        q_out, k_out = rope(positions, q, k)
        assert q_out.shape == q.shape
        assert k_out.shape == k.shape

    @pytest.mark.unit
    def test_different_positions_give_different_output(self):
        rope = RotaryEmbedding(head_size=8, rotary_dim=8,
                               max_position_embeddings=64, base=10000)
        q = torch.ones(2, 1, 8)
        q0, _ = rope(torch.tensor([0, 0]), q.clone(), q.clone())
        q1, _ = rope(torch.tensor([0, 1]), q.clone(), q.clone())
        assert not torch.allclose(q0[1], q1[1])

    @pytest.mark.unit
    def test_lru_cache_returns_same_instance(self):
        get_rope.cache_clear()
        r1 = get_rope(8, 8, 64, 10000.0)
        r2 = get_rope(8, 8, 64, 10000.0)
        assert r1 is r2

    @pytest.mark.unit
    def test_cos_sin_cache_shape(self):
        rope = RotaryEmbedding(head_size=16, rotary_dim=16,
                               max_position_embeddings=128, base=10000)
        assert rope.cos_sin_cache.shape == (128, 1, 16)

    @pytest.mark.unit
    def test_norm_preserved_approx(self):
        rope = RotaryEmbedding(head_size=8, rotary_dim=8,
                               max_position_embeddings=32, base=10000)
        q = torch.randn(4, 1, 8)
        q_norm_before = q.norm(dim=-1)
        q_out, _ = rope(torch.arange(4), q, q.clone())
        q_norm_after = q_out.norm(dim=-1)
        assert torch.allclose(q_norm_before.float(), q_norm_after.float(), atol=1e-5)
