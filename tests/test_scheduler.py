"""
Scheduler 单元测试

Phase 1：FCFS 调度（prefill 优先、内存约束、终止条件）
Phase 4：追加 Chunked Prefill 和抢占调度测试
"""
import pytest

from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.sampling_params import SamplingParams


def _make_sched(num_blocks=20, block_size=4, max_num_seqs=8,
                max_num_batched_tokens=64, eos=999) -> Scheduler:
    Sequence.block_size = block_size
    return Scheduler(num_blocks, block_size, max_num_seqs, max_num_batched_tokens, eos)


def _make_seq(num_tokens: int, block_size: int = 4, max_tokens: int = 10) -> Sequence:
    Sequence.block_size = block_size
    return Sequence(list(range(num_tokens)), SamplingParams(max_tokens=max_tokens))


# ─── Phase 1：FCFS 基础调度 ────────────────────────────────────────────────────


class TestSchedulerFCFS:

    @pytest.mark.unit
    def test_initial_state_is_finished(self):
        assert _make_sched().is_finished()

    @pytest.mark.unit
    def test_schedule_prefill_first(self):
        sched = _make_sched()
        seq = _make_seq(3)
        sched.add(seq)
        seqs, is_prefill = sched.schedule()
        assert is_prefill
        assert seq in seqs
        assert seq.status == SequenceStatus.RUNNING
        assert seq.num_scheduled_tokens == 3

    @pytest.mark.unit
    def test_prefill_then_decode_transition(self):
        sched = _make_sched()
        seq = _make_seq(3)
        sched.add(seq)
        seqs, p = sched.schedule()
        sched.postprocess(seqs, [10], p)
        assert seq.num_tokens == 4

        seqs2, p2 = sched.schedule()
        assert not p2
        assert seq in seqs2

    @pytest.mark.unit
    def test_eos_terminates_sequence(self):
        sched = _make_sched(eos=999)
        seq = _make_seq(2)
        sched.add(seq)
        seqs, p = sched.schedule()
        sched.postprocess(seqs, [999], p)
        assert seq.is_finished and sched.is_finished()

    @pytest.mark.unit
    def test_max_tokens_terminates_sequence(self):
        sched = _make_sched()
        seq = Sequence([1, 2, 3], SamplingParams(max_tokens=3))
        sched.add(seq)
        for token in [10, 20, 30]:
            seqs, p = sched.schedule()
            sched.postprocess(seqs, [token], p)
        assert seq.is_finished and sched.is_finished()

    @pytest.mark.unit
    def test_ignore_eos_flag(self):
        sched = _make_sched(eos=999)
        seq = Sequence([1], SamplingParams(max_tokens=3, ignore_eos=True))
        sched.add(seq)
        for _ in range(2):
            seqs, p = sched.schedule()
            sched.postprocess(seqs, [999], p)
            assert not seq.is_finished
        seqs, p = sched.schedule()
        sched.postprocess(seqs, [999], p)
        assert seq.is_finished   # max_tokens=3 触发

    @pytest.mark.unit
    def test_token_budget_limits_prefill_batch(self):
        sched = _make_sched(max_num_batched_tokens=10)
        sched.add(_make_seq(8))
        sched.add(_make_seq(8))
        seqs, is_prefill = sched.schedule()
        assert is_prefill and len(seqs) == 1

    @pytest.mark.unit
    def test_max_num_seqs_limits_batch(self):
        sched = _make_sched(max_num_seqs=2)
        for _ in range(5):
            sched.add(_make_seq(2))
        seqs, _ = sched.schedule()
        assert len(seqs) <= 2

    @pytest.mark.unit
    def test_memory_full_returns_empty_list(self):
        sched = _make_sched(num_blocks=2)
        sched.add(_make_seq(12))   # 需要 3 块
        seqs, is_prefill = sched.schedule()
        assert seqs == [] and is_prefill

    @pytest.mark.unit
    def test_full_lifecycle(self):
        sched = _make_sched()
        seq = Sequence([10, 20], SamplingParams(max_tokens=3))
        sched.add(seq)
        for token in [100, 200, 300]:
            seqs, p = sched.schedule()
            sched.postprocess(seqs, [token], p)
        assert sched.is_finished()
        assert seq.completion_token_ids == [100, 200, 300]

    @pytest.mark.unit
    def test_concurrent_seqs(self):
        sched = _make_sched()
        sp = SamplingParams(max_tokens=2)
        seq1 = Sequence([1, 2], sp)
        seq2 = Sequence([3, 4], sp)
        sched.add(seq1)
        sched.add(seq2)
        seqs, p = sched.schedule()
        assert len(seqs) == 2
        sched.postprocess(seqs, [10, 20], p)
        seqs, p = sched.schedule()
        sched.postprocess(seqs, [11, 21], p)
        assert seq1.is_finished and seq2.is_finished

    @pytest.mark.unit
    def test_blocks_released_on_finish(self):
        sched = _make_sched()
        initial_free = len(sched.block_manager.free_block_ids)
        seq = Sequence([1, 2, 3], SamplingParams(max_tokens=1))
        sched.add(seq)
        seqs, p = sched.schedule()
        sched.postprocess(seqs, [10], p)
        assert sched.is_finished()
        assert len(sched.block_manager.free_block_ids) == initial_free
