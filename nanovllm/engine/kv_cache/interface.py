"""
KV cache 规格抽象（对齐 vLLM V1 `vllm/v1/kv_cache_interface.py`）。

KVCacheSpec 描述"一层 KV cache 需要怎样的存储"，把块字节数 / 显存→块数 的计算从
ModelRunner 内联逻辑收敛到规格对象。nano 只有 full-attention 同构层，故仅实现
FullAttentionSpec；V1 的 SlidingWindow / Mamba / MLA / Encoder 等异构 Spec 不在范围内
（无对应模型层）。
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class KVCacheSpec(ABC):
    """单层 KV cache 规格基类。"""
    block_size: int

    @property
    @abstractmethod
    def page_size_bytes(self) -> int:
        """单层、单个块（page）的字节数。"""
        ...

    @abstractmethod
    def kv_cache_shape(self, num_blocks: int) -> tuple:
        """单层 KV cache 张量形状（与 AttentionBackend.get_kv_cache_shape 对齐）。"""
        ...


@dataclass(frozen=True)
class FullAttentionSpec(KVCacheSpec):
    """标准全注意力层的 KV cache 规格。"""
    num_kv_heads: int
    head_dim: int
    dtype: torch.dtype

    @property
    def page_size_bytes(self) -> int:
        # 一个块需同时存 K 与 V：2 * block_size * num_kv_heads * head_dim * itemsize
        return (2 * self.block_size * self.num_kv_heads
                * self.head_dim * self.dtype.itemsize)

    def kv_cache_shape(self, num_blocks: int) -> tuple:
        return (2, num_blocks, self.block_size, self.num_kv_heads, self.head_dim)

    def num_blocks_for_memory(self, available_bytes: int, num_layers: int) -> int:
        """给定可用显存与层数，反推可容纳的块数（全层共享同一块数）。"""
        return int(available_bytes) // (self.page_size_bytes * num_layers)
