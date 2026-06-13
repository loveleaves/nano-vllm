from collections import deque

from nanovllm.engine.sequence import Sequence


class Block:
    """
    KV cache 物理块。

    字段：
      block_id  — 物理块编号
      ref_count — 引用计数；为 0 时可被回收
    """

    def __init__(self, block_id: int):
        self.block_id = block_id
        self.ref_count = 0

    def reset(self):
        """分配给新 seq 前清零状态（ref_count 置 1）。"""
        self.ref_count = 1


class BlockManager:
    """
    KV cache 分页管理器（阶段一：不含前缀缓存）。

    数据结构：
      blocks         — 所有物理块列表，下标即 block_id
      free_block_ids — 空闲块 ID 的 FIFO deque（左取右还）
      used_block_ids — 当前被引用的块 ID 集合
    """

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    def _allocate_block(self) -> int:
        """从空闲队列取一个块并初始化。"""
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        assert block.ref_count == 0
        block.reset()
        self.used_block_ids.add(block_id)
        return block_id

    def _deallocate_block(self, block_id: int):
        """将引用计数归零的块归还空闲队列。"""
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> int:
        """
        检查能否为 seq 分配所需 KV 块。

        返回值：
          -1 — 空闲块不足
           0 — 可以分配（阶段一无前缀缓存，始终返回 0）
        """
        if len(self.free_block_ids) < seq.num_blocks:
            return -1
        return 0

    def allocate(self, seq: Sequence, num_cached_blocks: int = 0):
        """为 seq 分配 KV 块（阶段一：全部新分配，num_cached_blocks 恒为 0）。"""
        assert not seq.block_table
        for _ in range(seq.num_blocks):
            seq.block_table.append(self._allocate_block())
        seq.num_cached_tokens = 0

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
        """
        检查 decode 步是否有足够空闲块追加。
        仅当 seq 最后一个 token 恰好是某个新块的第一个 token
        （即上一块刚被填满、需要开新块）时，才需要新分配一个块。
        """
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        """decode 步按需分配新块（最后一 token 为新块首 token，即 len % block_size == 1 时分配）。"""
        if len(seq) % self.block_size == 1:
            seq.block_table.append(self._allocate_block())
