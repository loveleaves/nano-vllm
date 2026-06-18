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
        seqs, num_scheduled = sched.schedule()
        assert num_scheduled[seq.seq_id] == 3   # prefill chunk 调度 3 token
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
        # 第二步为纯 decode：每个 seq 仅调度 1 token
        assert all(n == 1 for n in p2.values())
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
    def test_token_budget_fills_across_seqs(self):
        # 统一连续批：任意 seq 可分块，预算跨 seq 填满（不再限队首）
        sched = _make_sched(max_num_batched_tokens=10)
        sched.add(_make_seq(8))
        sched.add(_make_seq(8))
        seqs, num_scheduled = sched.schedule()
        assert sum(num_scheduled.values()) == 10   # 预算恰好填满
        assert num_scheduled[seqs[0].seq_id] == 8   # seq1 完整 prefill
        assert num_scheduled[seqs[1].seq_id] == 2   # seq2 分块 2 token

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
        seqs, num_scheduled = sched.schedule()
        assert seqs == [] and not num_scheduled

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


# ─── Phase 4：Chunked Prefill ───────────────────────────────────────────────────


def _make_sched_p4(num_blocks=10, block_size=4, max_seqs=8,
                   max_batched_tokens=16, eos=-1):
    Sequence.block_size = block_size
    return Scheduler(num_blocks, block_size, max_seqs, max_batched_tokens, eos)


def _make_seq_p4(num_tokens, block_size=4, max_tokens=10):
    Sequence.block_size = block_size
    return Sequence(list(range(num_tokens)), SamplingParams(max_tokens=max_tokens))


class TestSchedulerChunkedPrefill:

    @pytest.mark.unit
    def test_chunked_prefill_two_rounds(self):
        sched = _make_sched_p4(num_blocks=10, max_batched_tokens=4)
        seq = _make_seq_p4(8)
        sched.add(seq)

        seqs, is_prefill = sched.schedule()
        assert is_prefill and len(seqs) == 1
        assert seqs[0].num_scheduled_tokens == 4

        sched.postprocess(seqs, [99], is_prefill)
        assert seq.num_cached_tokens == 4
        assert seq.num_tokens == 8

        seqs2, is_prefill2 = sched.schedule()
        assert is_prefill2 and seqs2[0].num_scheduled_tokens == 4

        sched.postprocess(seqs2, [42], is_prefill2)
        assert seq.num_cached_tokens == 8
        assert seq.num_tokens == 9

    @pytest.mark.unit
    def test_only_first_seq_chunked(self):
        sched = _make_sched_p4(num_blocks=20, max_batched_tokens=4)
        seq1 = _make_seq_p4(8)
        seq2 = _make_seq_p4(4)
        sched.add(seq1)
        sched.add(seq2)

        seqs, is_prefill = sched.schedule()
        assert is_prefill and len(seqs) == 1
        assert seqs[0] is seq1

    @pytest.mark.unit
    def test_chunked_seq_stays_in_waiting(self):
        sched = _make_sched_p4(num_blocks=10, max_batched_tokens=4)
        seq = _make_seq_p4(8)
        sched.add(seq)

        seqs, is_prefill = sched.schedule()
        sched.postprocess(seqs, [0], is_prefill)

        assert seq.status == SequenceStatus.WAITING
        assert seq in sched.waiting


# ─── Phase 4：抢占调度 ──────────────────────────────────────────────────────────


class TestSchedulerPreemption:

    @pytest.mark.unit
    def test_preempt_running_seq_when_no_free_blocks(self):
        # 2 块全被占满，decode 需要新块时触发抢占
        sched = _make_sched_p4(num_blocks=2, max_batched_tokens=8)
        seq1 = _make_seq_p4(4)   # 占 1 块
        seq2 = _make_seq_p4(4)   # 占 1 块
        sched.add(seq1)
        sched.add(seq2)

        # 第一步：两个 seq 同批 prefill 完成，各占 1 块（块用尽）
        seqs, ns = sched.schedule()
        assert len(seqs) == 2
        sched.postprocess(seqs, [1, 2], ns)

        # 第二步：两个 seq decode，各自第 5 个 token 需新块，但无空闲块 → 抢占
        seqs, ns = sched.schedule()
        assert all(n == 1 for n in ns.values())   # decode 步
        # 一个 seq 被抢占回 waiting（块不足以同时容纳两者增长）
        assert len(sched.waiting) >= 1

    @pytest.mark.unit
    def test_preempt_restores_seq_to_waiting_head(self):
        sched = _make_sched_p4(num_blocks=10, max_batched_tokens=8)
        seq = _make_seq_p4(4)
        sched.add(seq)

        seqs, is_prefill = sched.schedule()
        sched.postprocess(seqs, [1], is_prefill)
        assert seq.status == SequenceStatus.RUNNING

        sched.preempt(seq)
        assert seq.status == SequenceStatus.WAITING
        assert seq.is_prefill is True
        assert seq.block_table == []
        assert sched.waiting[0] is seq

    @pytest.mark.unit
    def test_memory_full_deadlock_protection(self):
        sched = _make_sched_p4(num_blocks=1, max_batched_tokens=16)
        seq = _make_seq_p4(8)   # 需要 2 块，只有 1 块 → 无法调度
        sched.add(seq)

        seqs, num_scheduled = sched.schedule()
        assert seqs == [] and not num_scheduled


# ─── 统一连续批：prefill 与 decode 混排同批 ──────────────────────────────────────


class TestSchedulerContinuousBatching:

    @pytest.mark.unit
    def test_mixed_prefill_decode_batch(self):
        # 一个 seq 先 prefill 完成进入 decode，再来一个新 seq；
        # 下一步调度应在同一批里同时含 decode(seq1) 与 prefill(seq2)
        sched = _make_sched(max_num_batched_tokens=64)
        seq1 = _make_seq(3)
        sched.add(seq1)
        seqs, ns = sched.schedule()           # 第一步：seq1 prefill
        sched.postprocess(seqs, [10], ns)
        assert seq1.status == SequenceStatus.RUNNING

        seq2 = _make_seq(5)
        sched.add(seq2)
        seqs, ns = sched.schedule()           # 第二步：seq1 decode + seq2 prefill

        assert seq1 in seqs and seq2 in seqs
        assert ns[seq1.seq_id] == 1           # seq1 decode：1 token
        assert ns[seq2.seq_id] == 5           # seq2 prefill：整段
        # 批内同时存在 query 长度 1 与 >1 → 混合批
        assert min(ns.values()) == 1 and max(ns.values()) > 1

    @pytest.mark.unit
    def test_decode_only_batch_all_ones(self):
        sched = _make_sched()
        s1, s2 = _make_seq(2), _make_seq(2)
        sched.add(s1)
        sched.add(s2)
        seqs, ns = sched.schedule()           # prefill 两者
        sched.postprocess(seqs, [10, 20], ns)
        seqs, ns = sched.schedule()           # 纯 decode 批
        assert seqs and all(n == 1 for n in ns.values())
