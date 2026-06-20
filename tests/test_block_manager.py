"""
BlockManager 单元测试

Phase 1：基础块分配（FIFO）
Phase 4：追加前缀缓存测试（compute_hash / hash_blocks / 共享块）
"""
import pytest

from nanovllm.engine.kv_cache import BlockPool, KVCacheManager
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams


def _make_seq(num_tokens: int, block_size: int = 4, max_tokens: int = 10) -> Sequence:
    Sequence.block_size = block_size
    return Sequence(list(range(num_tokens)), SamplingParams(max_tokens=max_tokens))


# ─── Phase 1：基础块分配 ────────────────────────────────────────────────────────


class TestBlockManagerBasic:

    def setup_method(self):
        Sequence.block_size = 4
        self.bm = KVCacheManager(num_blocks=10, block_size=4)

    @pytest.mark.unit
    def test_initial_state(self):
        assert len(self.bm.block_pool.free_block_ids) == 10
        assert len(self.bm.block_pool.used_block_ids) == 0

    @pytest.mark.unit
    def test_allocate_fills_block_table(self):
        seq = _make_seq(5)   # 需要 2 个逻辑块
        assert self.bm.can_allocate(seq) == 0
        self.bm.allocate(seq)
        assert len(seq.block_table) == 2
        assert len(self.bm.block_pool.free_block_ids) == 8
        assert len(self.bm.block_pool.used_block_ids) == 2

    @pytest.mark.unit
    def test_allocate_exact_capacity(self):
        seq = _make_seq(40)  # 10 块，刚好占满
        assert self.bm.can_allocate(seq) == 0
        self.bm.allocate(seq)
        assert len(self.bm.block_pool.free_block_ids) == 0

    @pytest.mark.unit
    def test_can_allocate_returns_negative_when_oom(self):
        seq = _make_seq(44)  # 11 块，超出容量
        assert self.bm.can_allocate(seq) == -1

    @pytest.mark.unit
    def test_deallocate_restores_blocks(self):
        seq = _make_seq(5)
        self.bm.allocate(seq)
        self.bm.deallocate(seq)
        assert len(self.bm.block_pool.free_block_ids) == 10
        assert seq.block_table == []
        assert seq.num_cached_tokens == 0

    @pytest.mark.unit
    def test_block_ids_unique_across_seqs(self):
        seqs = [_make_seq(2) for _ in range(4)]
        for s in seqs:
            self.bm.allocate(s)
        all_ids = [bid for s in seqs for bid in s.block_table]
        assert len(all_ids) == len(set(all_ids))

    @pytest.mark.unit
    def test_can_append_no_new_block_needed(self):
        seq = _make_seq(3)   # 3%4≠1，无需新块
        self.bm.allocate(seq)
        assert self.bm.can_append(seq)

    @pytest.mark.unit
    def test_can_append_new_block_needed(self):
        seq = _make_seq(4)
        self.bm.allocate(seq)
        seq.num_tokens = 5   # 5%4=1 → 需要新块
        assert self.bm.can_append(seq)

    @pytest.mark.unit
    def test_may_append_adds_block_when_full(self):
        seq = _make_seq(4)
        self.bm.allocate(seq)
        seq.num_tokens = 5
        self.bm.may_append(seq)
        assert len(seq.block_table) == 2

    @pytest.mark.unit
    def test_may_append_no_op_when_not_needed(self):
        seq = _make_seq(3)
        self.bm.allocate(seq)
        before = len(seq.block_table)
        self.bm.may_append(seq)
        assert len(seq.block_table) == before

    @pytest.mark.unit
    def test_fifo_reuse_order(self):
        seq1 = _make_seq(4)
        self.bm.allocate(seq1)
        first_block = seq1.block_table[0]
        self.bm.deallocate(seq1)
        seq2 = _make_seq(40)   # 占满所有块
        self.bm.allocate(seq2)
        assert first_block in seq2.block_table


# ─── Phase 4：前缀缓存哈希 ──────────────────────────────────────────────────────


def _make_seq_p4(num_tokens: int, block_size: int = 4) -> Sequence:
    Sequence.block_size = block_size
    from nanovllm.sampling_params import SamplingParams
    return Sequence(list(range(num_tokens)), SamplingParams(max_tokens=10))


class TestBlockManagerHash:

    @pytest.mark.unit
    def test_compute_hash_deterministic(self):
        tokens = [1, 2, 3, 4]
        assert BlockPool.compute_hash(tokens) == BlockPool.compute_hash(tokens)

    @pytest.mark.unit
    def test_compute_hash_different_prefix(self):
        tokens = [1, 2, 3, 4]
        h0 = BlockPool.compute_hash(tokens)
        ha = BlockPool.compute_hash(tokens, prefix=42)
        hb = BlockPool.compute_hash(tokens, prefix=99)
        assert h0 != ha and h0 != hb and ha != hb

    @pytest.mark.unit
    def test_compute_hash_different_tokens(self):
        assert BlockPool.compute_hash([1, 2, 3, 4]) != BlockPool.compute_hash([1, 2, 3, 5])

    @pytest.mark.unit
    def test_compute_hash_returns_int(self):
        assert isinstance(BlockPool.compute_hash([0, 1, 2, 3]), int)


class TestBlockManagerPrefixCache:

    @pytest.mark.unit
    def test_can_allocate_no_cache(self):
        bm = KVCacheManager(10, 4)
        seq = _make_seq_p4(8)
        assert bm.can_allocate(seq) == 0

    @pytest.mark.unit
    def test_can_allocate_insufficient_blocks(self):
        bm = KVCacheManager(1, 4)
        seq = _make_seq_p4(8)
        assert bm.can_allocate(seq) == -1

    @pytest.mark.unit
    def test_prefix_cache_hit_after_hash_blocks(self):
        bm = KVCacheManager(10, 4)
        seq1 = _make_seq_p4(8)
        bm.allocate(seq1, 0)
        seq1.num_scheduled_tokens = 8
        bm.hash_blocks(seq1)

        seq2 = _make_seq_p4(8)
        assert bm.can_allocate(seq2) == 1

    @pytest.mark.unit
    def test_allocate_with_prefix_cache_shares_block(self):
        bm = KVCacheManager(10, 4)
        seq1 = _make_seq_p4(8)
        bm.allocate(seq1, 0)
        seq1.num_scheduled_tokens = 8
        bm.hash_blocks(seq1)

        seq2 = _make_seq_p4(8)
        num_cached = bm.can_allocate(seq2)
        bm.allocate(seq2, num_cached)
        assert seq1.block_table[0] == seq2.block_table[0]

    @pytest.mark.unit
    def test_deallocate_preserves_hash(self):
        bm = KVCacheManager(10, 4)
        seq = _make_seq_p4(8)
        bm.allocate(seq, 0)
        seq.num_scheduled_tokens = 8
        bm.hash_blocks(seq)
        hashes_before = dict(bm.block_pool.hash_to_block_id)
        bm.deallocate(seq)
        for h in hashes_before:
            assert h in bm.block_pool.hash_to_block_id

    @pytest.mark.unit
    def test_reallocate_clears_old_hash(self):
        bm = KVCacheManager(1, 4)
        seq1 = _make_seq_p4(4)
        bm.allocate(seq1, 0)
        seq1.num_scheduled_tokens = 4
        bm.hash_blocks(seq1)
        old_hash = bm.block_pool.blocks[seq1.block_table[0]].hash
        assert old_hash in bm.block_pool.hash_to_block_id

        bm.deallocate(seq1)
        assert old_hash in bm.block_pool.hash_to_block_id  # 延迟复用

        seq2 = _make_seq_p4(4)
        seq2.token_ids = [10, 11, 12, 13]
        bm.allocate(seq2, 0)
        assert old_hash not in bm.block_pool.hash_to_block_id

    @pytest.mark.unit
    def test_ref_count_shared_block(self):
        bm = KVCacheManager(10, 4)
        seq1 = _make_seq_p4(8)
        bm.allocate(seq1, 0)
        seq1.num_scheduled_tokens = 8
        bm.hash_blocks(seq1)

        seq2 = _make_seq_p4(8)
        num_cached = bm.can_allocate(seq2)
        bm.allocate(seq2, num_cached)

        shared = seq1.block_table[0]
        assert bm.block_pool.blocks[shared].ref_count == 2
        bm.deallocate(seq1)
        assert bm.block_pool.blocks[shared].ref_count == 1
        assert shared in bm.block_pool.used_block_ids
        bm.deallocate(seq2)
        assert bm.block_pool.blocks[shared].ref_count == 0
        assert shared not in bm.block_pool.used_block_ids

    @pytest.mark.unit
    def test_hash_blocks_registers_full_blocks_only(self):
        bm = KVCacheManager(10, 4)
        seq = _make_seq_p4(5)
        bm.allocate(seq, 0)
        seq.num_scheduled_tokens = 5
        bm.hash_blocks(seq)
        assert bm.block_pool.blocks[seq.block_table[0]].hash != -1
        assert bm.block_pool.blocks[seq.block_table[1]].hash == -1
