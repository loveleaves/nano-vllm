"""
调度器接口抽象（对齐 vLLM V1 `v1/core/sched/interface.py::SchedulerInterface`）。

固化调度器对外契约，使 EngineCore 只依赖接口而非具体实现（未来可替换异步调度器等）。
nano 取最小子集：去掉 spec decode / grammar bitmask / kv connector / stats 等 V1 专属钩子。
"""
from abc import ABC, abstractmethod

from nanovllm.engine.sched.output import SchedulerOutput
from nanovllm.engine.sequence import Sequence


class SchedulerInterface(ABC):

    @abstractmethod
    def add_request(self, seq: Sequence) -> None:
        """登记一个新请求（进入 waiting 队列）。"""
        ...

    @abstractmethod
    def schedule(self) -> SchedulerOutput:
        """决定本步调度哪些序列、各调度多少 token。"""
        ...

    @abstractmethod
    def update_from_output(self, output: SchedulerOutput, token_ids: list[int]) -> None:
        """用执行器产出的 token 更新序列状态（追加 token / 终止 / 释放 KV）。"""
        ...

    @abstractmethod
    def abort(self, seq: Sequence) -> None:
        """显式中止一个请求并释放其资源。"""
        ...

    @abstractmethod
    def get_num_unfinished_requests(self) -> int:
        ...

    def has_unfinished_requests(self) -> bool:
        return self.get_num_unfinished_requests() > 0

    def is_finished(self) -> bool:
        return not self.has_unfinished_requests()
