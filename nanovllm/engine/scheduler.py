"""向后兼容垫片：Scheduler 已迁入 `nanovllm.engine.sched` 包（对齐 V1 sched/ 结构）。

旧 import 路径 `from nanovllm.engine.scheduler import Scheduler` 仍可用。
"""
from nanovllm.engine.sched import Scheduler, SchedulerOutput, SchedulingPolicy

__all__ = ["Scheduler", "SchedulerOutput", "SchedulingPolicy"]
