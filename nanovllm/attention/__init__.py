"""
注意力子系统（多后端抽象）。

向后兼容：保持 `from nanovllm.attention import Attention` 可用。
"""
from nanovllm.attention.common import CommonAttentionMetadata
from nanovllm.attention.backend import (
    AttentionBackend, AttentionImpl, AttentionMetadataBuilder,
)
from nanovllm.attention.flash_attn import (
    FlashAttentionBackend, FlashAttentionImpl, HAS_FLASH_ATTN,
)
from nanovllm.attention.torch_sdpa import TorchSDPABackend, TorchSDPAImpl
from nanovllm.attention.registry import (
    AttentionBackendEnum, register_backend,
)
from nanovllm.attention.selector import get_attn_backend
from nanovllm.attention.kv_ops import store_kvcache, HAS_TRITON
from nanovllm.attention.layer import Attention

__all__ = [
    "Attention",
    "CommonAttentionMetadata",
    "AttentionBackend", "AttentionImpl", "AttentionMetadataBuilder",
    "FlashAttentionBackend", "FlashAttentionImpl",
    "TorchSDPABackend", "TorchSDPAImpl",
    "AttentionBackendEnum", "register_backend",
    "get_attn_backend",
    "store_kvcache", "HAS_FLASH_ATTN", "HAS_TRITON",
]
