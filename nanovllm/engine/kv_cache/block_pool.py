"""
物理块池（对齐 vLLM V1 `v1/core/block_pool.py::BlockPool`）。

把"物理块 + 空闲队列 + 前缀缓存哈希表 + 引用计数"从 KVCacheManager 拆出：
BlockPool 只管块级原语（分配/释放/复用缓存块/注册哈希），不感知 Sequence；
KVCacheManager 在其上做每请求编排（见 kv_cache_manager.py）。
"""
from collections import deque

import numpy as np
import xxhash


class KVCacheBlock:
    """
    KV cache 物理块（对齐 V1 `KVCacheBlock`，原 nano `Block`）。

    字段：
      block_id  — 物理块编号
      ref_count — 引用计数；为 0 时可被回收（FIFO 空闲队列）
      hash      — 该块 token 序列的链式哈希（-1 表示未哈希）
      token_ids — 存储的 token 序列（用于哈希碰撞完整性校验）

    生命周期：
      分配 → reset() 清除上次哈希/token_ids，ref_count 置 1
      引用 → ref_count++（多个 seq 共享前缀块）
      释放 → ref_count--，归零时归还空闲队列（hash 保留供后续前缀命中）
      重分配 → 从哈希表删除旧 hash（避免脏命中）
    """

    def __init__(self, block_id: int):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids: list[int] = []

    def update(self, h: int, token_ids: list[int]):
        self.hash = h
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


# 向后兼容别名（旧代码/测试用 Block）
Block = KVCacheBlock


class BlockPool:
    """
    物理块池：分页 KV cache 的块级原语 + 前缀缓存哈希表。

    数据结构：
      blocks            — 所有物理块列表，下标即 block_id
      hash_to_block_id  — hash → block_id，前缀缓存核心查找表
      free_block_ids    — 空闲块 ID FIFO deque（左取右还）
      used_block_ids    — 当前被引用的块 ID 集合

    前缀缓存原理：
      对已满块计算链式 xxhash（含前一块的哈希值），形成唯一链式标识。
      释放时保留 hash（块留在 free 队列末尾），下次分配该块时再删除旧 hash。
    """

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[KVCacheBlock] = [KVCacheBlock(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = {}
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1) -> int:
        """
        计算一个块的链式哈希：
          - 将前一块的哈希值拼入，使同一块 token 在不同前缀下产生不同哈希
          - xxhash（非加密哈希，极快）
        """
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids, dtype=np.int64).tobytes())
        return h.intdigest()

    # ── 查询 ──────────────────────────────────────────────────────────────────
    def get_num_free_blocks(self) -> int:
        return len(self.free_block_ids)

    def get_block(self, block_id: int) -> KVCacheBlock:
        return self.blocks[block_id]

    def is_used(self, block_id: int) -> bool:
        return block_id in self.used_block_ids

    def cached_block_id(self, h: int) -> int:
        """按链式哈希查前缀缓存命中的 block_id；未命中返回 -1。"""
        return self.hash_to_block_id.get(h, -1)

    # ── 分配 / 释放 ────────────────────────────────────────────────────────────
    def get_new_block(self) -> int:
        """从空闲队列取一个块并初始化，清除其旧哈希记录。"""
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        assert block.ref_count == 0
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]
        block.reset()
        self.used_block_ids.add(block_id)
        return block_id

    def reuse_cached_block(self, block_id: int):
        """复用一个前缀缓存命中的块：已在用则 ref++，否则从空闲取出并置 ref=1。"""
        block = self.blocks[block_id]
        if block_id in self.used_block_ids:
            block.ref_count += 1
        else:
            block.ref_count = 1
            self.free_block_ids.remove(block_id)
            self.used_block_ids.add(block_id)

    def deref_block(self, block_id: int) -> bool:
        """引用计数 -1；归零则归还空闲队列（FIFO 末尾，延迟复用以延长前缀缓存寿命）。
        返回是否真正释放。"""
        block = self.blocks[block_id]
        block.ref_count -= 1
        if block.ref_count == 0:
            self.used_block_ids.remove(block_id)
            self.free_block_ids.append(block_id)
            return True
        return False

    def register_hash(self, block_id: int, h: int, token_ids: list[int]):
        """登记一个已填满块的链式哈希（供后续前缀命中）。"""
        self.blocks[block_id].update(h, token_ids)
        self.hash_to_block_id[h] = block_id
