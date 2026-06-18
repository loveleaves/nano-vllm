"""
可插拔 waiting 队列策略（对齐 vLLM V1 `v1/core/sched/request_queue.py`）。

提供 FCFS（先到先服务）与 PRIORITY（优先级，值越小越先；同级按到达顺序 seq_id）
两种策略，经 `create_request_queue(policy)` 工厂构造。Scheduler 的 waiting 队列由它
承载，从而解耦"调度算法"与"排队策略"。
"""
import heapq
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Iterator
from enum import Enum

from nanovllm.engine.sequence import Sequence


class SchedulingPolicy(str, Enum):
    FCFS = "fcfs"
    PRIORITY = "priority"


class RequestQueue(ABC):
    """waiting 队列抽象。pop/peek 返回"下一个该调度"的序列。

    除下列抽象方法外，实现还须支持容器协议：`__contains__` / `__len__` / `__iter__`
    及基于长度的真值判断（`bool(q)`）。这些不声明为 abstractmethod——否则其抽象占位会
    在 MRO 中遮蔽 deque 基于 `__len__` 的隐式真值（导致 `__bool__` 返回 None）。
    """

    @abstractmethod
    def add_request(self, seq: Sequence) -> None: ...
    @abstractmethod
    def pop_request(self) -> Sequence: ...
    @abstractmethod
    def peek_request(self) -> Sequence: ...
    @abstractmethod
    def prepend_request(self, seq: Sequence) -> None: ...
    @abstractmethod
    def remove_request(self, seq: Sequence) -> None: ...


class FCFSRequestQueue(deque, RequestQueue):
    """先到先服务：直接复用 deque（队尾入、队首出，抢占回插队首）。"""

    def add_request(self, seq: Sequence) -> None:
        self.append(seq)

    def pop_request(self) -> Sequence:
        return self.popleft()

    def peek_request(self) -> Sequence:
        return self[0]

    def prepend_request(self, seq: Sequence) -> None:
        self.appendleft(seq)

    def remove_request(self, seq: Sequence) -> None:
        self.remove(seq)

    # __contains__ / __bool__ / __len__ / __iter__ 由 deque 提供


class PriorityRequestQueue(RequestQueue):
    """优先级队列：按 (priority, seq_id) 最小堆出队（priority 越小越先；同级 FCFS）。"""

    def __init__(self) -> None:
        self._heap: list[tuple[int, int, Sequence]] = []

    def add_request(self, seq: Sequence) -> None:
        heapq.heappush(self._heap, (seq.priority, seq.seq_id, seq))

    def pop_request(self) -> Sequence:
        return heapq.heappop(self._heap)[2]

    def peek_request(self) -> Sequence:
        return self._heap[0][2]

    def prepend_request(self, seq: Sequence) -> None:
        # 优先级队列无"队首"概念，重新入堆即按优先级归位（抢占的 seq 优先级不变）
        self.add_request(seq)

    def remove_request(self, seq: Sequence) -> None:
        self._heap = [e for e in self._heap if e[2] is not seq]
        heapq.heapify(self._heap)

    def __contains__(self, seq: Sequence) -> bool:
        return any(e[2] is seq for e in self._heap)

    def __bool__(self) -> bool:
        return bool(self._heap)

    def __len__(self) -> int:
        return len(self._heap)

    def __iter__(self) -> Iterator[Sequence]:
        return (e[2] for e in sorted(self._heap))


def create_request_queue(policy: SchedulingPolicy) -> RequestQueue:
    if policy == SchedulingPolicy.PRIORITY:
        return PriorityRequestQueue()
    if policy == SchedulingPolicy.FCFS:
        return FCFSRequestQueue()
    raise ValueError(f"未知调度策略: {policy}")
