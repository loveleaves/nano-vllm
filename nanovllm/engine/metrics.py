"""
轻量可观测性（对齐 vLLM V1 `v1/metrics/` 的最小子集）。

EngineCore 每步聚合一份 SchedulerStats（运行/等待数、本步调度 token 数、KV 利用率），
供上层取用或周期性打印。nano 不引入 Prometheus，仅提供进程内统计 + 可选日志。
"""
import time
from dataclasses import dataclass


@dataclass
class SchedulerStats:
    """单步调度快照。"""
    num_running: int = 0           # 正在 decode 的序列数
    num_waiting: int = 0           # 等待 prefill 的序列数
    num_scheduled_tokens: int = 0  # 本步调度的 token 数
    num_gpu_blocks: int = 0        # KV cache 物理块总数
    num_gpu_blocks_used: int = 0   # 已占用块数

    @property
    def kv_cache_usage(self) -> float:
        """KV cache 占用率（0~1）。"""
        return self.num_gpu_blocks_used / self.num_gpu_blocks if self.num_gpu_blocks else 0.0


class StatLogger:
    """累计吞吐统计 + 按间隔打印（可选）。"""

    def __init__(self, log_interval_s: float = 5.0):
        self.log_interval_s = log_interval_s
        self.start = time.monotonic()
        self.last_log = self.start
        self.total_prompt_tokens = 0       # 累计 prefill token
        self.total_generation_tokens = 0   # 累计 decode（生成）token

    def record(self, stats: SchedulerStats, is_prefill_step: bool) -> None:
        if is_prefill_step:
            self.total_prompt_tokens += stats.num_scheduled_tokens
        else:
            self.total_generation_tokens += stats.num_scheduled_tokens

    def maybe_log(self, stats: SchedulerStats) -> str | None:
        """到间隔则返回一行统计字符串（调用方决定是否打印），否则 None。"""
        now = time.monotonic()
        if now - self.last_log < self.log_interval_s:
            return None
        self.last_log = now
        elapsed = max(now - self.start, 1e-6)
        line = (f"[metrics] running={stats.num_running} waiting={stats.num_waiting} "
                f"kv_usage={stats.kv_cache_usage:.1%} "
                f"prompt_tps={self.total_prompt_tokens / elapsed:.1f} "
                f"gen_tps={self.total_generation_tokens / elapsed:.1f}")
        return line
