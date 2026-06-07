"""
BlockManager 单元测试

Phase 1：基础块分配（FIFO）
Phase 4：追加前缀缓存测试（compute_hash / hash_blocks / 共享块）
"""
import pytest

from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams


def _make_seq(num_tokens: int, block_size: int = 4, max_tokens: int = 10) -> Sequence:
    Sequence.block_size = block_size
    return Sequence(list(range(num_tokens)), SamplingParams(max_tokens=max_tokens))


# ─── Phase 1：基础块分配 ────────────────────────────────────────────────────────


class TestBlockManagerBasic:

    def setup_method(self):
        Sequence.block_size = 4
        self.bm = BlockManager(num_blocks=10, block_size=4)

    @pytest.mark.unit
    def test_initial_state(self):
        assert len(self.bm.free_block_ids) == 10
        assert len(self.bm.used_block_ids) == 0

    @pytest.mark.unit
    def test_allocate_fills_block_table(self):
        seq = _make_seq(5)   # 需要 2 个逻辑块
        assert self.bm.can_allocate(seq) == 0
        self.bm.allocate(seq)
        assert len(seq.block_table) == 2
        assert len(self.bm.free_block_ids) == 8
        assert len(self.bm.used_block_ids) == 2

    @pytest.mark.unit
    def test_allocate_exact_capacity(self):
        seq = _make_seq(40)  # 10 块，刚好占满
        assert self.bm.can_allocate(seq) == 0
        self.bm.allocate(seq)
        assert len(self.bm.free_block_ids) == 0

    @pytest.mark.unit
    def test_can_allocate_returns_negative_when_oom(self):
        seq = _make_seq(44)  # 11 块，超出容量
        assert self.bm.can_allocate(seq) == -1

    @pytest.mark.unit
    def test_deallocate_restores_blocks(self):
        seq = _make_seq(5)
        self.bm.allocate(seq)
        self.bm.deallocate(seq)
        assert len(self.bm.free_block_ids) == 10
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
