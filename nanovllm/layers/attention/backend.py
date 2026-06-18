"""
注意力后端抽象（对齐 vLLM V1 的 AttentionBackend / AttentionImpl / AttentionMetadataBuilder）。

三件套职责：
  AttentionBackend          — 静态工厂：暴露 impl / builder 类与 KV cache 形状
  AttentionMetadataBuilder  — 把 CommonAttentionMetadata 转成后端专属元数据
  AttentionImpl             — 执行实际 attention kernel（含 KV 写入）

nano 仅 FlashAttn + SDPA 两后端，能力查询精简为最小集，接口为未来后端预留。
"""
from abc import ABC, abstractmethod
import torch

from nanovllm.layers.attention.common import CommonAttentionMetadata


class AttentionBackend(ABC):
    """后端静态工厂。"""

    @staticmethod
    @abstractmethod
    def get_name() -> str:
        ...

    @staticmethod
    @abstractmethod
    def get_impl_cls() -> type["AttentionImpl"]:
        ...

    @staticmethod
    @abstractmethod
    def get_builder_cls() -> type["AttentionMetadataBuilder"]:
        ...

    @staticmethod
    def get_kv_cache_shape(num_blocks: int, block_size: int,
                           num_kv_heads: int, head_dim: int) -> tuple:
        """nano 默认 KV 布局：[2(k/v), num_blocks, block_size, num_kv_heads, head_dim]。"""
        return (2, num_blocks, block_size, num_kv_heads, head_dim)


class AttentionMetadataBuilder(ABC):
    """通用元数据 → 后端专属元数据。nano 两后端共享字段，build 为恒等。"""

    @abstractmethod
    def build(self, common: CommonAttentionMetadata):
        ...


class AttentionImpl(ABC):
    """后端 kernel 实现。持有 num_heads/head_dim/scale/num_kv_heads。"""

    num_heads: int
    head_dim: int
    scale: float
    num_kv_heads: int

    @abstractmethod
    def __init__(self, num_heads: int, head_dim: int, scale: float, num_kv_heads: int):
        ...

    @abstractmethod
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                k_cache: torch.Tensor, v_cache: torch.Tensor,
                attn_md: CommonAttentionMetadata) -> torch.Tensor:
        ...
