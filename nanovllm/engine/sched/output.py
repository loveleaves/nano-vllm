"""
调度决策的结构化输出（对齐 vLLM V1 `v1/core/sched/output.py::SchedulerOutput`）。

V1 用 SchedulerOutput 把"本步调度了什么"固化成结构体（含 new/cached 请求拆分、
spec/encoder/kv-connector 元数据等）。nano 取其最小可用子集：因执行器直接吃 Sequence
对象、且无 spec/多模态/连接器，故不拆 NewRequestData/CachedRequestData，只保留调度
结果的核心字段。
"""
from dataclasses import dataclass, field

from nanovllm.engine.sequence import Sequence


@dataclass
class SchedulerOutput:
    """一次 `Scheduler.schedule()` 的结构化结果。

    scheduled_seqs            — 本步被调度的序列（prefill chunk 与 decode 混排）
    num_scheduled_tokens      — seq_id → 本步调度的 token 数（decode 为 1，prefill 为 chunk 长）
    total_num_scheduled_tokens— 本步 token 总数（= sum(num_scheduled_tokens)）
    preempted_seq_ids         — 本步因显存不足被抢占回 waiting 的 seq_id
    finished_seq_ids          — 需从执行器持久批回收行槽位的 seq_id（上步结束/中止
                                + 本步被抢占）；随本步下发给各 rank 的 ModelRunner.InputBatch
    """
    scheduled_seqs: list[Sequence] = field(default_factory=list)
    num_scheduled_tokens: dict[int, int] = field(default_factory=dict)
    total_num_scheduled_tokens: int = 0
    preempted_seq_ids: set[int] = field(default_factory=set)
    finished_seq_ids: set[int] = field(default_factory=set)

    @property
    def is_empty(self) -> bool:
        return not self.scheduled_seqs
