"""
注意力子系统（多后端抽象）。

向后兼容：保持 `from nanovllm.layers.attention import Attention` 可用。
"""
from nanovllm.layers.attention.common import CommonAttentionMetadata
from nanovllm.layers.attention.backend import (
    AttentionBackend, AttentionImpl, AttentionMetadataBuilder,
)
from nanovllm.layers.attention.flash_attn import (
    FlashAttentionBackend, FlashAttentionImpl, HAS_FLASH_ATTN,
)
from nanovllm.layers.attention.torch_sdpa import TorchSDPABackend, TorchSDPAImpl
from nanovllm.layers.attention.selector import get_attn_backend
from nanovllm.layers.attention.kv_ops import store_kvcache, HAS_TRITON
from nanovllm.layers.attention.layer import Attention

__all__ = [
    "Attention",
    "CommonAttentionMetadata",
    "AttentionBackend", "AttentionImpl", "AttentionMetadataBuilder",
    "FlashAttentionBackend", "FlashAttentionImpl",
    "TorchSDPABackend", "TorchSDPAImpl",
    "get_attn_backend",
    "store_kvcache", "HAS_FLASH_ATTN", "HAS_TRITON",
]
