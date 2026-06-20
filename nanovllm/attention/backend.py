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

from nanovllm.attention.common import CommonAttentionMetadata


class AttentionBackend(ABC):
    """后端静态工厂 + 能力查询（对齐 V1：按 head_size/dtype/平台筛选可用后端）。"""

    # 支持的计算 dtype（空 = 不限）；子类按 kernel 约束覆盖
    supported_dtypes: list[torch.dtype] = [torch.float16, torch.bfloat16, torch.float32]

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

    # ── 能力查询（对齐 V1 AttentionBackend.supports_*）─────────────────────────
    @classmethod
    def is_available(cls, device_type: str) -> bool:
        """该后端在当前平台是否可用（如 flash 需 cuda + 已安装）。默认始终可用。"""
        return True

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        """支持的 head_size 列表（空 = 不限）。"""
        return []

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        sizes = cls.get_supported_head_sizes()
        return (not sizes) or head_size in sizes

    @classmethod
    def supports_dtype(cls, dtype: torch.dtype) -> bool:
        return (not cls.supported_dtypes) or dtype in cls.supported_dtypes


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
