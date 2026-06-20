"""向后兼容垫片：BlockManager 已拆分进 `nanovllm.engine.kv_cache` 包（对齐 V1）。

  BlockManager == KVCacheManager（每请求编排），其块级原语下沉到 BlockPool。
  旧 import `from nanovllm.engine.block_manager import BlockManager` / `Block` 仍可用。
"""
from nanovllm.engine.kv_cache import Block, BlockManager, BlockPool, KVCacheManager

__all__ = ["BlockManager", "KVCacheManager", "BlockPool", "Block"]
