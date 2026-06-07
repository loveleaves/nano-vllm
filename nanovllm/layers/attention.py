import torch
from torch import nn

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

try:
    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False

import torch.nn.functional as F
from nanovllm.utils.context import get_context


if HAS_TRITON:
    @triton.jit
    def store_kvcache_kernel(
        key_ptr,
        key_stride,
        value_ptr,
        value_stride,
        k_cache_ptr,
        v_cache_ptr,
        slot_mapping_ptr,
        D: tl.constexpr,
    ):
        """
        Triton kernel：将新计算的 K/V 写入 KV cache 的指定 slot。

        每个 program 处理一个 token：
          - 从 slot_mapping 读取目标 slot
          - slot == -1 表示无效（CUDA graph dummy token），跳过
          - 向量化读写 D 个元素
        """
        idx = tl.program_id(0)
        slot = tl.load(slot_mapping_ptr + idx)
        if slot == -1:
            return
        key_offsets = idx * key_stride + tl.arange(0, D)
        value_offsets = idx * value_stride + tl.arange(0, D)
        key = tl.load(key_ptr + key_offsets)
        value = tl.load(value_ptr + value_offsets)
        cache_offsets = slot * D + tl.arange(0, D)
        tl.store(k_cache_ptr + cache_offsets, key)
        tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache_triton(key: torch.Tensor, value: torch.Tensor,
                         k_cache: torch.Tensor, v_cache: torch.Tensor,
                         slot_mapping: torch.Tensor):
    """使用 Triton kernel 将 key/value 写入 KV cache。"""
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    store_kvcache_kernel[(N,)](
        key, key.stride(0), value, value.stride(0),
        k_cache, v_cache, slot_mapping, D,
    )


def store_kvcache_naive(key: torch.Tensor, value: torch.Tensor,
                        k_cache: torch.Tensor, v_cache: torch.Tensor,
                        slot_mapping: torch.Tensor):
    """朴素 Python scatter（Triton 不可用时的 fallback）。"""
    block_size = k_cache.shape[1]
    for idx in range(key.shape[0]):
        slot = slot_mapping[idx].item()
        if slot < 0:
            continue
        block_id = slot // block_size
        offset = slot % block_size
        k_cache[block_id, offset] = key[idx]
        v_cache[block_id, offset] = value[idx]


def store_kvcache(key: torch.Tensor, value: torch.Tensor,
                  k_cache: torch.Tensor, v_cache: torch.Tensor,
                  slot_mapping: torch.Tensor):
    """将 K/V 写入 KV cache，优先使用 Triton kernel。"""
    if HAS_TRITON and key.is_cuda:
        store_kvcache_triton(key, value, k_cache, v_cache, slot_mapping)
    else:
        store_kvcache_naive(key, value, k_cache, v_cache, slot_mapping)


class Attention(nn.Module):
    """
    PagedAttention 层（Phase 4：FlashAttention + Triton KV 写入）。

    k_cache / v_cache：
      初始为空张量，由 ModelRunner.allocate_kv_cache 替换为全局 KV cache 对应层切片。
      形状：[num_blocks, block_size, num_kv_heads, head_dim]

    forward 分两路：
      prefill: flash_attn_varlen_func（可变长，causal）
      decode:  flash_attn_with_kvcache（分页读 KV cache）

    KV 写入在注意力计算前执行，确保当前 token 参与自注意力。
    """

    def __init__(self, num_heads: int, head_dim: int, scale: float, num_kv_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.num_kv_groups = num_heads // num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        输入：
          q: [N, num_heads, head_dim]
          k: [N, num_kv_heads, head_dim]
          v: [N, num_kv_heads, head_dim]
        返回：
          o: [N, num_heads, head_dim]
        """
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache

        # 先将当前 K/V 写入 KV cache
        if k_cache.numel() and v_cache.numel() and context.slot_mapping is not None:
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)

        if context.is_prefill:
            if HAS_FLASH_ATTN:
                if context.block_tables is not None:
                    # 有前缀缓存，从 KV cache 读历史
                    k_fa, v_fa = k_cache, v_cache
                else:
                    k_fa, v_fa = k, v
                o = flash_attn_varlen_func(
                    q, k_fa, v_fa,
                    max_seqlen_q=context.max_seqlen_q,
                    cu_seqlens_q=context.cu_seqlens_q,
                    max_seqlen_k=context.max_seqlen_k,
                    cu_seqlens_k=context.cu_seqlens_k,
                    softmax_scale=self.scale,
                    causal=True,
                    block_table=context.block_tables,
                )
            else:
                # FlashAttention 不可用，退回 SDPA
                if self.num_kv_groups > 1:
                    k = k.repeat_interleave(self.num_kv_groups, dim=1)
                    v = v.repeat_interleave(self.num_kv_groups, dim=1)
                q_t = q.transpose(0, 1).unsqueeze(0)
                k_t = k.transpose(0, 1).unsqueeze(0)
                v_t = v.transpose(0, 1).unsqueeze(0)
                o = F.scaled_dot_product_attention(q_t, k_t, v_t, scale=self.scale, is_causal=True)
                o = o.squeeze(0).transpose(0, 1)
        else:
            if HAS_FLASH_ATTN:
                # q: [bs, num_heads, head_dim] → [bs, 1, num_heads, head_dim]
                o = flash_attn_with_kvcache(
                    q.unsqueeze(1), k_cache, v_cache,
                    cache_seqlens=context.context_lens,
                    block_table=context.block_tables,
                    softmax_scale=self.scale,
                    causal=True,
                )
                # [bs, 1, num_heads, head_dim] → [bs, num_heads, head_dim]
                o = o.squeeze(1)
            else:
                # Fallback: 从 KV cache 手动收集
                bs = q.size(0)
                outputs = []
                for i in range(bs):
                    seq_len = context.context_lens[i].item()
                    block_size = k_cache.shape[1]
                    num_blocks_needed = (seq_len + block_size - 1) // block_size
                    blocks = context.block_tables[i, :num_blocks_needed]
                    k_hist = torch.cat([k_cache[b] for b in blocks], dim=0)[:seq_len]
                    v_hist = torch.cat([v_cache[b] for b in blocks], dim=0)[:seq_len]
                    if self.num_kv_groups > 1:
                        k_hist = k_hist.repeat_interleave(self.num_kv_groups, dim=1)
                        v_hist = v_hist.repeat_interleave(self.num_kv_groups, dim=1)
                    qi = q[i].unsqueeze(1)
                    ki = k_hist.transpose(0, 1)
                    vi = v_hist.transpose(0, 1)
                    oi = F.scaled_dot_product_attention(
                        qi.unsqueeze(0), ki.unsqueeze(0), vi.unsqueeze(0), scale=self.scale
                    ).squeeze(0).squeeze(1)
                    outputs.append(oi)
                o = torch.stack(outputs, dim=0)

        return o
