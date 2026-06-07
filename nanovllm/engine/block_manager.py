from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:
    """
    KV cache 物理块。

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


class BlockManager:
    """
    KV cache 分页管理器（Phase 4：含前缀缓存）。

    数据结构：
      blocks            — 所有物理块列表，下标即 block_id
      hash_to_block_id  — hash → block_id，前缀缓存核心查找表
      free_block_ids    — 空闲块 ID FIFO deque（左取右还）
      used_block_ids    — 当前被引用的块 ID 集合

    前缀缓存原理：
      对已满块计算链式 xxhash（含前一块的哈希值），形成唯一链式标识。
      后续相同前缀的请求 can_allocate 时探测命中，allocate 时直接复用。
      释放时保留 hash（块留在 free 队列末尾），下次分配时再删除。
    """

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
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

    def _allocate_block(self) -> int:
        """从空闲队列取一个块并初始化，清除旧哈希记录。"""
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        assert block.ref_count == 0
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]
        block.reset()
        self.used_block_ids.add(block_id)
        return block_id

    def _deallocate_block(self, block_id: int):
        """将引用计数归零的块归还空闲队列（FIFO 末尾，延迟复用延长前缀缓存有效期）。"""
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> int:
        """
        检查能否为 seq 分配 KV 块，同时探测前缀缓存命中数。

        返回值：
          -1               — 空闲块不足，无法调度
          num_cached_blocks — 可以调度，且有这多块命中前缀缓存（0 = 全部新分配）

        实现：遍历除最后一块外的所有满块（最后块可能不满，不参与哈希缓存）：
          - 计算链式哈希 → 查 hash_to_block_id → 校验 token_ids（防碰撞）
          - 命中且在 used_block_ids → num_new_blocks-- （共享引用，无需新分配）
          - 未命中 → 停止（后续块无法命中）
        """
        h = -1
        num_cached_blocks = 0
        num_new_blocks = seq.num_blocks
        for i in range(seq.num_blocks - 1):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                break
            num_cached_blocks += 1
            if block_id in self.used_block_ids:
                num_new_blocks -= 1
        if len(self.free_block_ids) < num_new_blocks:
            return -1
        return num_cached_blocks

    def allocate(self, seq: Sequence, num_cached_blocks: int = 0):
        """
        为 seq 正式分配 KV 块：
          - 前 num_cached_blocks 块：直接引用（ref_count++）
          - 其余块：从空闲队列新分配
          - seq.num_cached_tokens = num_cached_blocks * block_size（跳过 prefill）
        """
        assert not seq.block_table
        h = -1
        for i in range(num_cached_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id[h]
            block = self.blocks[block_id]
            if block_id in self.used_block_ids:
                block.ref_count += 1
            else:
                block.ref_count = 1
                self.free_block_ids.remove(block_id)
                self.used_block_ids.add(block_id)
            seq.block_table.append(block_id)
        for _ in range(num_cached_blocks, seq.num_blocks):
            seq.block_table.append(self._allocate_block())
        seq.num_cached_tokens = num_cached_blocks * self.block_size

    def deallocate(self, seq: Sequence):
        """释放 seq 的所有 KV 块（逆序，让最新块最先进空闲队列）。"""
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        """decode 步是否有足够块追加（仅当 len%block_size==1 时需要新块）。"""
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        """decode 步按需分配新块。"""
        if len(seq) % self.block_size == 1:
            seq.block_table.append(self._allocate_block())

    def hash_blocks(self, seq: Sequence):
        """
        每步推理后，对本步新填满的块计算并注册哈希（供后续前缀命中）。

        范围：从上次缓存边界到本步缓存结束的已满块（最后块可能未满，跳过）。
        """
        start = seq.num_cached_tokens // self.block_size
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
        if start == end:
            return
        h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
        for i in range(start, end):
            block = self.blocks[seq.block_table[i]]
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block.block_id
