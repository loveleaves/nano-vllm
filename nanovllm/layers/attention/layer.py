"""Attention 层：持有后端 impl + builder，forward 委派（不含 kernel 细节）。"""
import torch
from torch import nn

from nanovllm.layers.attention.common import CommonAttentionMetadata
from nanovllm.layers.attention.selector import get_attn_backend


class Attention(nn.Module):
    """
    PagedAttention 层（多后端可插拔）。

    k_cache / v_cache：
      初始为空张量，由 ModelRunner.allocate_kv_cache 替换为全局 KV cache 对应层切片。
      形状：[num_blocks, block_size, num_kv_heads, head_dim]

    后端在 __init__ 时绑定（forward 期不再 dispatch，保证 CUDA graph 捕获稳定）：
      impl    — 实际 kernel（FlashAttn / SDPA）
      builder — 通用元数据 → 后端专属元数据
    """

    def __init__(self, num_heads: int, head_dim: int, scale: float, num_kv_heads: int):
        super().__init__()
        # 按 head_size/dtype/平台筛选后端（当前默认设备与 dtype 即模型构建环境）
        backend = get_attn_backend(
            head_size=head_dim, dtype=torch.get_default_dtype(),
            device_type=torch.get_default_device().type)
        self.impl = backend.get_impl_cls()(num_heads, head_dim, scale, num_kv_heads)
        self.builder = backend.get_builder_cls()()
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                common_md: CommonAttentionMetadata) -> torch.Tensor:
        """
        输入 q:[N,num_heads,head_dim]  k/v:[N,num_kv_heads,head_dim]
        返回 o:[N,num_heads,head_dim]
        """
        md = self.builder.build(common_md)
        return self.impl.forward(q, k, v, self.k_cache, self.v_cache, md)
