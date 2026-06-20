"""
swap 抢占单元测试（CPU 纯逻辑）：

  - KVCacheManager：swap_out/swap_in 的槽位分配、块表与 num_cached_tokens 处理
  - Scheduler：swap_enabled 时抢占走 swap（而非 recompute），换回恢复 decode
  - 兜底：swap 槽位耗尽时回退到 recompute

GPU 张量 D2H/H2D 往返不在此（需显卡），见 docs/arch_swap/testing.md。
"""
import pytest

from nanovllm.engine.kv_cache.kv_cache_manager import KVCacheManager
from nanovllm.engine.sched import Scheduler
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.sampling_params import SamplingParams


def _seq(num_tokens: int, block_size: int = 4, max_tokens: int = 10) -> Sequence:
    Sequence.block_size = block_size
    return Sequence(list(range(num_tokens)), SamplingParams(max_tokens=max_tokens))


# ─── KVCacheManager swap 原语 ──────────────────────────────────────────────────


class TestKVCacheManagerSwap:

    @pytest.mark.unit
    def test_swap_out_allocates_slots_and_preserves_cached_tokens(self):
        mgr = KVCacheManager(num_blocks=10, block_size=4, num_swap_blocks=8)
        seq = _seq(8)                       # 2 块
        mgr.allocate(seq)
        seq.num_cached_tokens = 8
        gpu_blocks = list(seq.block_table)

        mapping = mgr.swap_out(seq)

        assert [g for g, _ in mapping] == gpu_blocks   # 按逻辑块序
        assert len(mgr.free_swap_slots) == 8 - 2       # 占用 2 个 swap 槽
        assert seq.seq_id in mgr.swapped_slots
        assert seq.block_table == []                   # GPU 块已释放
        assert seq.num_cached_tokens == 8              # 保留，恢复后续 decode

    @pytest.mark.unit
    def test_swap_in_restores_block_table_and_returns_slots(self):
        mgr = KVCacheManager(num_blocks=10, block_size=4, num_swap_blocks=8)
        seq = _seq(8)
        mgr.allocate(seq)
        seq.num_cached_tokens = 8
        slots = [s for _, s in mgr.swap_out(seq)]

        mapping = mgr.swap_in(seq)

        assert [s for _, s in mapping] == slots        # 同一批 swap 槽
        assert len(mgr.free_swap_slots) == 8           # 槽位全部归还
        assert seq.seq_id not in mgr.swapped_slots
        assert len(seq.block_table) == 2               # GPU 块重新分配

    @pytest.mark.unit
    def test_can_swap_out_in_thresholds(self):
        mgr = KVCacheManager(num_blocks=2, block_size=4, num_swap_blocks=1)
        seq = _seq(8)                       # 需 2 块/槽
        mgr.allocate(seq)
        assert mgr.can_swap_out(seq) is False          # swap 槽只有 1，不足 2
        small = _seq(4)
        mgr2 = KVCacheManager(num_blocks=2, block_size=4, num_swap_blocks=4)
        mgr2.allocate(small)
        assert mgr2.can_swap_out(small) is True


# ─── Scheduler swap 抢占决策 ───────────────────────────────────────────────────


class TestSchedulerSwap:

    @pytest.mark.unit
    def test_preempt_routes_to_swap_when_enabled(self):
        sched = Scheduler(num_kvcache_blocks=10, block_size=4, max_num_seqs=8,
                          max_num_batched_tokens=16, eos=-1, num_swap_blocks=8)
        seq = _seq(4)
        sched.add_request(seq)
        out = sched.schedule()
        sched.update_from_output(out, [1])             # decode 一步 → len 5
        assert seq.status == SequenceStatus.RUNNING

        swap_out_list = []
        sched.preempt(seq, swap_out_list)

        assert seq.status == SequenceStatus.WAITING
        assert seq in sched.swapped                     # 进 swapped 而非 waiting
        assert len(sched.waiting) == 0
        assert swap_out_list                            # 收集了 D2H 搬运映射
        assert seq.num_cached_tokens > 0                # 保留缓存（恢复 decode）

    @pytest.mark.unit
    def test_swap_in_resume_on_next_schedule(self):
        sched = Scheduler(num_kvcache_blocks=10, block_size=4, max_num_seqs=8,
                          max_num_batched_tokens=16, eos=-1, num_swap_blocks=8)
        seq = _seq(4)
        sched.add_request(seq)
        out = sched.schedule()
        sched.update_from_output(out, [1])
        sched.preempt(seq, [])                          # 手动换出
        assert seq in sched.swapped

        out2 = sched.schedule()                         # phase 0 应换回

        assert seq not in sched.swapped
        assert seq.status == SequenceStatus.RUNNING
        assert out2.blocks_to_swap_in                   # 收集了 H2D 搬运映射
        assert seq.block_table                          # GPU 块已恢复
        assert seq in out2.scheduled_seqs               # 换回后立即 decode

    @pytest.mark.unit
    def test_falls_back_to_recompute_when_swap_full(self):
        # swap 区满 → 抢占回退到 recompute（推回 waiting）
        sched = Scheduler(num_kvcache_blocks=10, block_size=4, max_num_seqs=8,
                          max_num_batched_tokens=16, eos=-1, num_swap_blocks=1)
        seq = _seq(8)                                   # 需 2 槽，swap 只 1
        sched.add_request(seq)
        out = sched.schedule()
        sched.update_from_output(out, [1])

        swap_out_list = []
        sched.preempt(seq, swap_out_list)

        assert seq not in sched.swapped
        assert seq.status == SequenceStatus.WAITING
        assert sched.waiting.peek_request() is seq      # recompute：回 waiting 队首
        assert seq.block_table == []
        assert not swap_out_list

    @pytest.mark.unit
    def test_unfinished_count_includes_swapped(self):
        sched = Scheduler(num_kvcache_blocks=10, block_size=4, max_num_seqs=8,
                          max_num_batched_tokens=16, eos=-1, num_swap_blocks=8)
        seq = _seq(4)
        sched.add_request(seq)
        out = sched.schedule()
        sched.update_from_output(out, [1])
        sched.running.remove(seq)                       # schedule() 先从 running 弹出受害者
        sched.preempt(seq, [])
        assert sched.get_num_unfinished_requests() == 1
        assert not sched.is_finished()

    @pytest.mark.unit
    def test_abort_swapped_returns_slots(self):
        sched = Scheduler(num_kvcache_blocks=10, block_size=4, max_num_seqs=8,
                          max_num_batched_tokens=16, eos=-1, num_swap_blocks=8)
        seq = _seq(8)
        sched.add_request(seq)
        out = sched.schedule()
        sched.update_from_output(out, [1])
        sched.preempt(seq, [])
        used = 8 - len(sched.block_manager.free_swap_slots)
        assert used > 0

        sched.abort(seq)

        assert seq not in sched.swapped
        assert len(sched.block_manager.free_swap_slots) == 8   # 槽位归还
        assert seq.seq_id in sched.finished_req_ids
