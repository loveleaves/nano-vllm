"""
Attention 层单元测试（统一 SDPA 路径，CPU 可运行）

统一连续批后 Attention.forward(q, k, v, attn_md) 不再区分 prefill/decode：
  - block_table 为 None：裸 k/v varlen（warmup / 无缓存的整段 prefill）
  - block_table 非 None：从分页 KV cache 读历史（decode / 前缀缓存 / chunk 续算）
"""
import pytest
import torch

from nanovllm.utils.context import AttentionMetadata
from nanovllm.layers.attention import Attention


def _prefill_md(cu_q):
    """构造无缓存 prefill 元数据（block_table=None，KV 长度==query 长度）。"""
    cu_q = torch.tensor(cu_q, dtype=torch.int32)
    seglen = (cu_q[1:] - cu_q[:-1])
    return AttentionMetadata(
        query_start_loc=cu_q,
        cu_seqlens_k=cu_q.clone(),
        max_query_len=int(seglen.max()),
        max_seq_len=int(seglen.max()),
        slot_mapping=None,
        block_table=None,
    )


class TestAttentionPrefill:

    @pytest.mark.unit
    def test_mha_output_shape(self):
        attn = Attention(num_heads=4, head_dim=8, scale=8 ** -0.5, num_kv_heads=4)
        N = 5
        q, k, v = (torch.randn(N, 4, 8) for _ in range(3))
        o = attn(q, k, v, _prefill_md([0, 5]))
        assert o.shape == (N, 4, 8)

    @pytest.mark.unit
    def test_gqa_output_shape(self):
        attn = Attention(num_heads=4, head_dim=8, scale=8 ** -0.5, num_kv_heads=2)
        N = 6
        q = torch.randn(N, 4, 8)
        k = torch.randn(N, 2, 8)
        v = torch.randn(N, 2, 8)
        o = attn(q, k, v, _prefill_md([0, 6]))
        assert o.shape == (N, 4, 8)

    @pytest.mark.unit
    def test_single_token_prefill(self):
        attn = Attention(num_heads=2, head_dim=4, scale=4 ** -0.5, num_kv_heads=2)
        q, k, v = (torch.randn(1, 2, 4) for _ in range(3))
        o = attn(q, k, v, _prefill_md([0, 1]))
        assert o.shape == (1, 2, 4)

    @pytest.mark.unit
    def test_causal_correctness(self):
        # 第 0 个 query 只能看到自己；用 value=单位行验证不泄露未来
        torch.manual_seed(0)
        attn = Attention(num_heads=1, head_dim=4, scale=4 ** -0.5, num_kv_heads=1)
        N = 3
        q = torch.randn(N, 1, 4)
        k = torch.randn(N, 1, 4)
        v = torch.arange(N * 4, dtype=torch.float32).reshape(N, 1, 4)
        o = attn(q, k, v, _prefill_md([0, N]))
        # query0 只 attend key0 → 输出必须等于 v[0]
        assert torch.allclose(o[0], v[0], atol=1e-5)


class TestAttentionMixedBatch:

    @pytest.mark.unit
    def test_cross_sequence_isolation(self):
        # 两个独立 prefill 序列拼在同一批，输出必须与各自单独计算一致
        torch.manual_seed(1)
        attn = Attention(num_heads=2, head_dim=4, scale=4 ** -0.5, num_kv_heads=2)
        a = [torch.randn(3, 2, 4) for _ in range(3)]   # 序列 A，长 3
        b = [torch.randn(2, 2, 4) for _ in range(3)]   # 序列 B，长 2
        qa, ka, va = a
        qb, kb, vb = b
        oa = attn(qa, ka, va, _prefill_md([0, 3]))
        ob = attn(qb, kb, vb, _prefill_md([0, 2]))
        # 拼批
        q = torch.cat([qa, qb]); k = torch.cat([ka, kb]); v = torch.cat([va, vb])
        o = attn(q, k, v, _prefill_md([0, 3, 5]))
        assert torch.allclose(o[:3], oa, atol=1e-5)   # A 不被 B 污染
        assert torch.allclose(o[3:], ob, atol=1e-5)   # B 不 attend A


class TestAttentionDecode:

    @pytest.mark.unit
    def test_decode_reads_from_cache(self):
        # 构造一个手工 KV cache，decode（query_len=1）从 cache 读全历史
        torch.manual_seed(2)
        block_size = 256
        attn = Attention(num_heads=2, head_dim=4, scale=4 ** -0.5, num_kv_heads=2)
        # 单块缓存，前 4 个 token 是历史
        ctx_len = 4
        kc = torch.zeros(1, block_size, 2, 4)
        vc = torch.zeros(1, block_size, 2, 4)
        hist_k = torch.randn(ctx_len, 2, 4)
        hist_v = torch.randn(ctx_len, 2, 4)
        kc[0, :ctx_len] = hist_k
        vc[0, :ctx_len] = hist_v
        attn.k_cache, attn.v_cache = kc, vc
        # decode：1 个新 token，slot 写入位置 ctx_len-1（这里历史已含当前，简化为读 ctx_len）
        q = torch.randn(1, 2, 4)
        md = AttentionMetadata(
            query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
            cu_seqlens_k=torch.tensor([0, ctx_len], dtype=torch.int32),
            max_query_len=1, max_seq_len=ctx_len,
            slot_mapping=torch.tensor([-1], dtype=torch.int32),  # 不写 cache
            block_table=torch.tensor([[0]], dtype=torch.int32),
        )
        o = attn(q, torch.randn(1, 2, 4), torch.randn(1, 2, 4), md)
        # 参考：q attend 完整历史（右下对齐 causal，query 在最后一行 → 全可见）
        ref = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(0, 1).unsqueeze(0),
            hist_k.transpose(0, 1).unsqueeze(0),
            hist_v.transpose(0, 1).unsqueeze(0),
            scale=4 ** -0.5,
        ).squeeze(0).transpose(0, 1)
        assert o.shape == (1, 2, 4)
        assert torch.allclose(o, ref, atol=1e-5)

    @pytest.mark.unit
    def test_decode_gqa_expansion(self):
        # GQA：cache 中 kv_heads=1，q_heads=2，应正确广播
        torch.manual_seed(3)
        block_size = 256
        attn = Attention(num_heads=2, head_dim=4, scale=4 ** -0.5, num_kv_heads=1)
        ctx_len = 3
        kc = torch.zeros(1, block_size, 1, 4)
        vc = torch.zeros(1, block_size, 1, 4)
        kc[0, :ctx_len] = torch.randn(ctx_len, 1, 4)
        vc[0, :ctx_len] = torch.randn(ctx_len, 1, 4)
        attn.k_cache, attn.v_cache = kc, vc
        q = torch.randn(1, 2, 4)
        md = AttentionMetadata(
            query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
            cu_seqlens_k=torch.tensor([0, ctx_len], dtype=torch.int32),
            max_query_len=1, max_seq_len=ctx_len,
            slot_mapping=torch.tensor([-1], dtype=torch.int32),
            block_table=torch.tensor([[0]], dtype=torch.int32),
        )
        o = attn(q, torch.randn(1, 1, 4), torch.randn(1, 1, 4), md)
        assert o.shape == (1, 2, 4)
