"""FlashAttention 后端：统一 varlen 调用（覆盖 prefill / decode / 前缀缓存）。"""
import torch

try:
    from flash_attn import flash_attn_varlen_func
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False

from nanovllm.layers.attention.backend import (
    AttentionBackend, AttentionImpl, AttentionMetadataBuilder,
)
from nanovllm.layers.attention.common import CommonAttentionMetadata
from nanovllm.layers.attention.kv_ops import store_kvcache


class FlashAttentionMetadataBuilder(AttentionMetadataBuilder):
    """nano FlashAttn 直接消费 CommonAttentionMetadata，build 为恒等。"""

    def build(self, common: CommonAttentionMetadata) -> CommonAttentionMetadata:
        return common


class FlashAttentionImpl(AttentionImpl):
    """
    统一单一 flash_attn_varlen_func 调用，无 prefill/decode 分支。
      block_table 非 None：从分页 KV cache 读历史（decode = query_len 1 退化）
      block_table 为 None：裸 k/v（仅 warmup，无 cache）
    flash_attn 2.8.x 无 seqused_k，故用 cu_seqlens_k（累计 KV 长度）+ block_table。
    """

    def __init__(self, num_heads: int, head_dim: int, scale: float, num_kv_heads: int):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads

    def forward(self, q, k, v, k_cache, v_cache, md: CommonAttentionMetadata):
        if k_cache.numel() and v_cache.numel() and md.slot_mapping is not None:
            store_kvcache(k, v, k_cache, v_cache, md.slot_mapping)
        if md.block_table is not None:
            k_fa, v_fa, block_table = k_cache, v_cache, md.block_table
        else:
            k_fa, v_fa, block_table = k, v, None
        return flash_attn_varlen_func(
            q, k_fa, v_fa,
            cu_seqlens_q=md.query_start_loc, max_seqlen_q=md.max_query_len,
            cu_seqlens_k=md.cu_seqlens_k, max_seqlen_k=md.max_seq_len,
            softmax_scale=self.scale, causal=True, block_table=block_table,
        )


class FlashAttentionBackend(AttentionBackend):

    @staticmethod
    def get_name() -> str:
        return "flash_attn"

    @staticmethod
    def get_impl_cls() -> type[AttentionImpl]:
        return FlashAttentionImpl

    @staticmethod
    def get_builder_cls() -> type[AttentionMetadataBuilder]:
        return FlashAttentionMetadataBuilder
