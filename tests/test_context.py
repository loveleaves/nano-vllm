"""
Context 工具函数单元测试
"""
import pytest
import torch

from nanovllm.utils.context import Context, set_context, get_context, reset_context


class TestContext:

    @pytest.mark.unit
    def test_default_context_is_decode(self):
        reset_context()
        ctx = get_context()
        assert not ctx.is_prefill
        assert ctx.cu_seqlens_q is None
        assert ctx.slot_mapping is None
        assert ctx.max_seqlen_q == 0

    @pytest.mark.unit
    def test_set_prefill_context(self):
        cu_q = torch.tensor([0, 3, 7], dtype=torch.int32)
        cu_k = torch.tensor([0, 3, 7], dtype=torch.int32)
        sm = torch.arange(7, dtype=torch.int32)
        set_context(True, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
                    max_seqlen_q=4, max_seqlen_k=4, slot_mapping=sm)
        ctx = get_context()
        assert ctx.is_prefill
        assert ctx.max_seqlen_q == 4
        assert torch.equal(ctx.cu_seqlens_q, cu_q)
        reset_context()

    @pytest.mark.unit
    def test_set_decode_context(self):
        sm = torch.tensor([10, 20], dtype=torch.int32)
        set_context(False, slot_mapping=sm)
        ctx = get_context()
        assert not ctx.is_prefill
        assert torch.equal(ctx.slot_mapping, sm)
        reset_context()

    @pytest.mark.unit
    def test_reset_clears_all_fields(self):
        set_context(True, max_seqlen_q=99)
        reset_context()
        ctx = get_context()
        assert not ctx.is_prefill
        assert ctx.max_seqlen_q == 0
        assert ctx.cu_seqlens_q is None

    @pytest.mark.unit
    def test_context_is_global_singleton(self):
        set_context(True, max_seqlen_q=42)
        ctx1 = get_context()
        ctx2 = get_context()
        assert ctx1 is ctx2
        reset_context()
