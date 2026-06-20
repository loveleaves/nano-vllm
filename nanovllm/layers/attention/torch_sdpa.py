"""Torch SDPA 后端：逐序列 scaled_dot_product_attention（flash_attn 不可用/CPU 时）。"""
import torch
import torch.nn.functional as F

from nanovllm.layers.attention.backend import (
    AttentionBackend, AttentionImpl, AttentionMetadataBuilder,
)
from nanovllm.layers.attention.common import CommonAttentionMetadata
from nanovllm.layers.attention.kv_ops import store_kvcache


class TorchSDPAMetadataBuilder(AttentionMetadataBuilder):
    def build(self, common: CommonAttentionMetadata) -> CommonAttentionMetadata:
        return common


class TorchSDPAImpl(AttentionImpl):
    """
    统一 SDPA：按 query_start_loc 逐序列独立计算（避免跨序列污染），
    decode 即 seqlen_q==1 退化。seqlen_k>seqlen_q（decode/前缀缓存/chunk 续算）时
    历史 K/V 从分页 cache 收集，causal mask 按右下对齐。
    """

    def __init__(self, num_heads: int, head_dim: int, scale: float, num_kv_heads: int):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.num_kv_groups = num_heads // num_kv_heads

    def forward(self, q, k, v, k_cache, v_cache, md: CommonAttentionMetadata):
        if k_cache.numel() and v_cache.numel() and md.slot_mapping is not None:
            store_kvcache(k, v, k_cache, v_cache, md.slot_mapping)

        cu_q = md.query_start_loc
        if cu_q is None:
            cu_q = torch.tensor([0, q.size(0)])
        cu_k = md.cu_seqlens_k if md.cu_seqlens_k is not None else cu_q
        outputs = []
        for s in range(cu_q.numel() - 1):
            q0, q1 = cu_q[s].item(), cu_q[s + 1].item()
            seqlen_q = q1 - q0
            seqlen_k = cu_k[s + 1].item() - cu_k[s].item()
            q_s = q[q0:q1]
            if seqlen_k > seqlen_q:
                # 历史前缀在 KV cache 中，按 block_table 收集完整 K/V
                block_size = k_cache.shape[1]
                num_blocks = (seqlen_k + block_size - 1) // block_size
                blocks = md.block_table[s, :num_blocks]
                k_s = torch.cat([k_cache[b] for b in blocks], dim=0)[:seqlen_k]
                v_s = torch.cat([v_cache[b] for b in blocks], dim=0)[:seqlen_k]
            else:
                k_s, v_s = k[q0:q1], v[q0:q1]
            if self.num_kv_groups > 1:
                k_s = k_s.repeat_interleave(self.num_kv_groups, dim=1)
                v_s = v_s.repeat_interleave(self.num_kv_groups, dim=1)
            q_t = q_s.transpose(0, 1).unsqueeze(0)
            k_t = k_s.transpose(0, 1).unsqueeze(0)
            v_t = v_s.transpose(0, 1).unsqueeze(0)
            if seqlen_k > seqlen_q:
                # 右下对齐 causal：query i 可见 key j ≤ i + (seqlen_k - seqlen_q)
                mask = torch.ones(seqlen_q, seqlen_k, dtype=torch.bool,
                                  device=q.device).tril(seqlen_k - seqlen_q)
                o_s = F.scaled_dot_product_attention(
                    q_t, k_t, v_t, scale=self.scale, attn_mask=mask)
            else:
                o_s = F.scaled_dot_product_attention(
                    q_t, k_t, v_t, scale=self.scale, is_causal=True)
            outputs.append(o_s.squeeze(0).transpose(0, 1))
        return torch.cat(outputs, dim=0)


class TorchSDPABackend(AttentionBackend):

    # SDPA 兜底后端：任意平台可用、不限 head_size、支持 fp16/bf16/fp32
    supported_dtypes = [torch.float16, torch.bfloat16, torch.float32]

    @staticmethod
    def get_name() -> str:
        return "torch_sdpa"

    @staticmethod
    def get_impl_cls() -> type[AttentionImpl]:
        return TorchSDPAImpl

    @staticmethod
    def get_builder_cls() -> type[AttentionMetadataBuilder]:
        return TorchSDPAMetadataBuilder
