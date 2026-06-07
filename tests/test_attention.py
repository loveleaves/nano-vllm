"""
Attention 层单元测试（SDPA prefill，CPU 可运行）
"""
import pytest
import torch

from nanovllm.utils.context import set_context, reset_context
from nanovllm.layers.attention import Attention


class TestAttentionPrefill:

    @pytest.mark.unit
    def test_mha_output_shape(self):
        reset_context()
        set_context(is_prefill=True, cu_seqlens_q=torch.tensor([0, 5]))
        attn = Attention(num_heads=4, head_dim=8, scale=8 ** -0.5, num_kv_heads=4)
        N = 5
        q = torch.randn(N, 4, 8)
        k = torch.randn(N, 4, 8)
        v = torch.randn(N, 4, 8)
        o = attn(q, k, v)
        assert o.shape == (N, 4, 8)
        reset_context()

    @pytest.mark.unit
    def test_gqa_output_shape(self):
        reset_context()
        set_context(is_prefill=True)
        attn = Attention(num_heads=4, head_dim=8, scale=8 ** -0.5, num_kv_heads=2)
        N = 6
        q = torch.randn(N, 4, 8)
        k = torch.randn(N, 2, 8)
        v = torch.randn(N, 2, 8)
        o = attn(q, k, v)
        assert o.shape == (N, 4, 8)
        reset_context()

    @pytest.mark.unit
    def test_causal_mask_shape(self):
        reset_context()
        set_context(is_prefill=True)
        attn = Attention(num_heads=1, head_dim=4, scale=4 ** -0.5, num_kv_heads=1)
        N = 3
        q = torch.randn(N, 1, 4)
        k = torch.randn(N, 1, 4)
        v = torch.eye(N, 4).unsqueeze(1)
        o = attn(q, k, v)
        assert o.shape == (N, 1, 4)
        reset_context()

    @pytest.mark.unit
    def test_single_token_prefill(self):
        reset_context()
        set_context(is_prefill=True, cu_seqlens_q=torch.tensor([0, 1]))
        attn = Attention(num_heads=2, head_dim=4, scale=4 ** -0.5, num_kv_heads=2)
        q = torch.randn(1, 2, 4)
        k = torch.randn(1, 2, 4)
        v = torch.randn(1, 2, 4)
        o = attn(q, k, v)
        assert o.shape == (1, 2, 4)
        reset_context()
