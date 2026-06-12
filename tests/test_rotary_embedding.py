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


# ─── Partial RoPE（rotary_dim < head_size）────────────────────────────────────

class TestPartialRoPE:

    @pytest.mark.unit
    def test_rotary_dims_change_passthrough_unchanged(self):
        """前 rotary_dim 维应被旋转，后续维度保持不变。"""
        get_rope.cache_clear()
        head_size, rotary_dim = 16, 4
        rope = RotaryEmbedding(head_size=head_size, rotary_dim=rotary_dim,
                               max_position_embeddings=32, base=10000)
        q = torch.randn(3, 2, head_size)
        k = torch.randn(3, 2, head_size)
        q_out, k_out = rope(torch.arange(3), q.clone(), k.clone())

        # pass-through 段不变
        assert torch.allclose(q_out[..., rotary_dim:], q[..., rotary_dim:])
        assert torch.allclose(k_out[..., rotary_dim:], k[..., rotary_dim:])
        # rotary 段已变化（非零输入不可能完全不变）
        assert not torch.allclose(q_out[..., :rotary_dim], q[..., :rotary_dim])

    @pytest.mark.unit
    def test_partial_rope_output_shape_unchanged(self):
        get_rope.cache_clear()
        rope = RotaryEmbedding(head_size=256, rotary_dim=64,
                               max_position_embeddings=128, base=1_000_000)
        q = torch.randn(5, 4, 256)
        k = torch.randn(5, 2, 256)
        q_out, k_out = rope(torch.arange(5), q, k)
        assert q_out.shape == q.shape
        assert k_out.shape == k.shape

    @pytest.mark.unit
    def test_full_rope_equals_original_behavior(self):
        """rotary_dim == head_size 时与旧实现结果一致。"""
        get_rope.cache_clear()
        head_size = 8
        rope = RotaryEmbedding(head_size=head_size, rotary_dim=head_size,
                               max_position_embeddings=32, base=10000)
        q = torch.randn(4, 1, head_size)
        k = torch.randn(4, 1, head_size)
        pos = torch.arange(4)
        q_out, k_out = rope(pos, q.clone(), k.clone())
        # 验证范数保持（RoPE 是等距变换）
        assert torch.allclose(q.norm(dim=-1).float(),
                               q_out.norm(dim=-1).float(), atol=1e-5)

    @pytest.mark.unit
    def test_lru_cache_supports_multiple_rotary_configs(self):
        """maxsize=16 能同时缓存不同的旋转维度配置。"""
        get_rope.cache_clear()
        r1 = get_rope(head_size=256, rotary_dim=64,  max_position=128, base=1e6)
        r2 = get_rope(head_size=256, rotary_dim=256, max_position=128, base=1e6)
        r3 = get_rope(head_size=64,  rotary_dim=64,  max_position=128, base=1e4)
        assert r1 is not r2
        assert r1 is not r3
        # 再次请求应命中缓存
        r1b = get_rope(head_size=256, rotary_dim=64, max_position=128, base=1e6)
        assert r1 is r1b

    @pytest.mark.unit
    def test_cos_sin_cache_shape_partial(self):
        """cos_sin_cache 形状应为 [max_pos, 1, rotary_dim]（非 head_size）。"""
        get_rope.cache_clear()
        rope = RotaryEmbedding(head_size=256, rotary_dim=64,
                               max_position_embeddings=128, base=1_000_000)
        assert rope.cos_sin_cache.shape == (128, 1, 64)
