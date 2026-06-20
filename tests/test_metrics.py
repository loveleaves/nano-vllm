"""metrics 单元测试（CPU）：SchedulerStats / StatLogger / Scheduler.make_stats。"""
import pytest

from nanovllm.engine.metrics import SchedulerStats, StatLogger
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams


class TestSchedulerStats:

    @pytest.mark.unit
    def test_kv_cache_usage(self):
        s = SchedulerStats(num_gpu_blocks=10, num_gpu_blocks_used=3)
        assert s.kv_cache_usage == pytest.approx(0.3)

    @pytest.mark.unit
    def test_kv_cache_usage_zero_blocks(self):
        assert SchedulerStats(num_gpu_blocks=0).kv_cache_usage == 0.0


class TestStatLogger:

    @pytest.mark.unit
    def test_record_accumulates_by_step_kind(self):
        lg = StatLogger(log_interval_s=1e9)   # 永不触发打印
        lg.record(SchedulerStats(num_scheduled_tokens=100), is_prefill_step=True)
        lg.record(SchedulerStats(num_scheduled_tokens=4), is_prefill_step=False)
        assert lg.total_prompt_tokens == 100
        assert lg.total_generation_tokens == 4
        assert lg.maybe_log(SchedulerStats()) is None   # 未到间隔

    @pytest.mark.unit
    def test_maybe_log_emits_after_interval(self):
        lg = StatLogger(log_interval_s=0.0)   # 立即可打印
        line = lg.maybe_log(SchedulerStats(num_running=2, num_waiting=1,
                                           num_gpu_blocks=10, num_gpu_blocks_used=5))
        assert line is not None and "running=2" in line and "kv_usage=50.0%" in line


class TestSchedulerMakeStats:

    @pytest.mark.unit
    def test_make_stats_reflects_queues_and_blocks(self):
        Sequence.block_size = 4
        sch = Scheduler(num_kvcache_blocks=16, block_size=4, max_num_seqs=8,
                        max_num_batched_tokens=64, eos=999)
        sch.add(Sequence([1, 2, 3], SamplingParams()))
        sch.add(Sequence([4, 5], SamplingParams()))
        sch.schedule()                         # 两个 prompt 进入调度、分配块
        st = sch.make_stats(num_scheduled_tokens=5)
        assert st.num_scheduled_tokens == 5
        assert st.num_gpu_blocks == 16
        assert st.num_gpu_blocks_used >= 1     # 已分配若干块
        assert 0.0 < st.kv_cache_usage <= 1.0
