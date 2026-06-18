"""
AttentionMetadata 数据类单元测试
"""
import pytest
import torch

from nanovllm.utils.context import AttentionMetadata


class TestAttentionMetadata:

    @pytest.mark.unit
    def test_default_is_empty(self):
        md = AttentionMetadata()
        assert md.query_start_loc is None
        assert md.cu_seqlens_k is None
        assert md.slot_mapping is None
        assert md.block_table is None
        assert md.max_query_len == 0
        assert md.max_seq_len == 0

    @pytest.mark.unit
    def test_prefill_like_metadata(self):
        # 两个 prefill 序列：长度 3 和 4
        qsl = torch.tensor([0, 3, 7], dtype=torch.int32)
        cu_k = torch.tensor([0, 3, 7], dtype=torch.int32)
        sm = torch.arange(7, dtype=torch.int32)
        md = AttentionMetadata(
            query_start_loc=qsl, cu_seqlens_k=cu_k,
            max_query_len=4, max_seq_len=4, slot_mapping=sm,
        )
        assert md.max_query_len == 4
        assert torch.equal(md.query_start_loc, qsl)
        assert torch.equal(md.cu_seqlens_k, cu_k)
        assert not md.is_decode_only

    @pytest.mark.unit
    def test_decode_only_metadata(self):
        # 三个 decode 序列：query 长度均为 1，KV 累计长度 [0,10,30,60]
        qsl = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
        cu_k = torch.tensor([0, 10, 30, 60], dtype=torch.int32)
        sm = torch.tensor([10, 20, 30], dtype=torch.int32)
        md = AttentionMetadata(
            query_start_loc=qsl, cu_seqlens_k=cu_k,
            max_query_len=1, max_seq_len=30, slot_mapping=sm,
        )
        assert md.is_decode_only
        assert torch.equal(md.slot_mapping, sm)

    @pytest.mark.unit
    def test_mixed_batch_not_decode_only(self):
        # 混合批：一个 prefill chunk(5) + 两个 decode(1,1)
        md = AttentionMetadata(max_query_len=5)
        assert not md.is_decode_only

    @pytest.mark.unit
    def test_block_table_none_means_no_cache(self):
        md = AttentionMetadata(max_query_len=4)
        assert md.block_table is None
