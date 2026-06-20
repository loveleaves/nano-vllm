"""
KV cache 管理器（对齐 vLLM V1 `v1/core/kv_cache_manager.py::KVCacheManager`）。

每请求（Sequence）的块编排：前缀缓存命中探测、分配/释放、decode 追加块、哈希注册。
块级原语下沉到 BlockPool；本类只做"序列 → 物理块"的映射与策略。

> 范围：nano 仅有同构 full-attention 层，故单一 BlockPool 即可；不引入 V1 的
> KVCacheCoordinator / 多组（hybrid：sliding window / Mamba / MLA）编排。详见
> docs/arch_kvcache/design.md。
"""
from collections import deque

from nanovllm.engine.sequence import Sequence
from nanovllm.engine.kv_cache.block_pool import BlockPool


class KVCacheManager:

    def __init__(self, num_blocks: int, block_size: int, num_swap_blocks: int = 0):
        self.block_size = block_size
        self.block_pool = BlockPool(num_blocks, block_size)
        # CPU swap 区：空闲槽位 + 已换出序列的槽位映射（seq_id → 逻辑块序的 swap_slot 列表）
        self.num_swap_blocks = num_swap_blocks
        self.free_swap_slots: deque[int] = deque(range(num_swap_blocks))
        self.swapped_slots: dict[int, list[int]] = {}

    # ── 每请求块管理 ──────────────────────────────────────────────────────────
    def can_allocate(self, seq: Sequence) -> int:
        """
        检查能否为 seq 分配 KV 块，同时探测前缀缓存命中数。

        返回值：
          -1                — 空闲块不足，无法调度
          num_cached_blocks — 可以调度，且有这多块命中前缀缓存（0 = 全部新分配）

        实现：遍历除最后一块外的所有满块（最后块可能不满，不参与哈希缓存）：
          - 计算链式哈希 → 查命中 → 校验 token_ids（防碰撞）
          - 命中且在 used → num_new_blocks-- （共享引用，无需新分配）
          - 未命中 → 停止（后续块无法命中）
        """
        pool = self.block_pool
        h = -1
        num_cached_blocks = 0
        num_new_blocks = seq.num_blocks
        for i in range(seq.num_blocks - 1):
            token_ids = seq.block(i)
            h = pool.compute_hash(token_ids, h)
            block_id = pool.cached_block_id(h)
            if block_id == -1 or pool.blocks[block_id].token_ids != token_ids:
                break
            num_cached_blocks += 1
            if pool.is_used(block_id):
                num_new_blocks -= 1
        if pool.get_num_free_blocks() < num_new_blocks:
            return -1
        return num_cached_blocks

    def allocate(self, seq: Sequence, num_cached_blocks: int = 0):
        """
        为 seq 正式分配 KV 块：
          - 前 num_cached_blocks 块：复用前缀缓存块（ref++ / 从空闲取出）
          - 其余块：从空闲队列新分配
          - seq.num_cached_tokens = num_cached_blocks * block_size（跳过 prefill）
        """
        assert not seq.block_table
        pool = self.block_pool
        h = -1
        for i in range(num_cached_blocks):
            token_ids = seq.block(i)
            h = pool.compute_hash(token_ids, h)
            block_id = pool.cached_block_id(h)
            pool.reuse_cached_block(block_id)
            seq.block_table.append(block_id)
        for _ in range(num_cached_blocks, seq.num_blocks):
            seq.block_table.append(pool.get_new_block())
        seq.num_cached_tokens = num_cached_blocks * self.block_size

    def deallocate(self, seq: Sequence):
        """释放 seq 的所有 KV 块（逆序，让最新块最先进空闲队列）。"""
        for block_id in reversed(seq.block_table):
            self.block_pool.deref_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    # ── swap（抢占换出/换入，对齐 V1 swap_out/swap_in）────────────────────────
    def can_swap_out(self, seq: Sequence) -> bool:
        return len(self.free_swap_slots) >= seq.num_blocks

    def swap_out(self, seq: Sequence) -> list[tuple[int, int]]:
        """把 seq 全部逻辑块搬到 CPU swap 区：分配 swap_slots、释放 GPU 块（**保留**
        num_cached_tokens，故恢复后续 decode 而非重算）。返回 [(gpu_block_id, swap_slot)]
        （按逻辑块序），供 ModelRunner 做 D2H 拷贝。"""
        block_ids = list(seq.block_table)
        slots = [self.free_swap_slots.popleft() for _ in block_ids]
        self.swapped_slots[seq.seq_id] = slots
        mapping = list(zip(block_ids, slots))
        for block_id in reversed(block_ids):
            self.block_pool.deref_block(block_id)
        seq.block_table.clear()         # GPU 块已释放；num_cached_tokens 不动
        return mapping

    def can_swap_in(self, seq: Sequence) -> bool:
        return self.block_pool.get_num_free_blocks() >= seq.num_blocks

    def swap_in(self, seq: Sequence) -> list[tuple[int, int]]:
        """为 seq 重新分配 GPU 块并从 CPU swap 区恢复：返回 [(gpu_block_id, swap_slot)]
        供 ModelRunner 做 H2D 拷贝；归还 swap_slots，恢复 seq.block_table。"""
        slots = self.swapped_slots.pop(seq.seq_id)
        block_ids = [self.block_pool.get_new_block() for _ in slots]
        seq.block_table = block_ids
        for s in slots:
            self.free_swap_slots.append(s)
        return list(zip(block_ids, slots))

    def can_append(self, seq: Sequence) -> bool:
        """decode 步是否有足够块追加（仅当 len%block_size==1 时需要新块）。"""
        return self.block_pool.get_num_free_blocks() >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        """decode 步按需分配新块。"""
        if len(seq) % self.block_size == 1:
            seq.block_table.append(self.block_pool.get_new_block())

    def truncate_blocks(self, seq: Sequence):
        """投机解码回滚：释放超出当前 num_tokens 所需的**尾部**块（前部块保留，其 KV 不动）。

        只 pop/deref 末尾多余块，已分配的前部块物理位置不变——故 verify 写入的、被接受 token
        对应的 KV 完好（被拒绝 token 的尾部块释放，其 KV 随块回收）。
        """
        needed = (len(seq) + self.block_size - 1) // self.block_size
        while len(seq.block_table) > needed:
            self.block_pool.deref_block(seq.block_table.pop())

    def hash_blocks(self, seq: Sequence):
        """
        每步推理后，对本步新填满的块计算并注册哈希（供后续前缀命中）。

        范围：从上次缓存边界到本步缓存结束的已满块（最后块可能未满，跳过）。
        """
        pool = self.block_pool
        start = seq.num_cached_tokens // self.block_size
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
        if start == end:
            return
        h = pool.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
        for i in range(start, end):
            token_ids = seq.block(i)
            h = pool.compute_hash(token_ids, h)
            pool.register_hash(seq.block_table[i], h, token_ids)
