"""RequestQueue 策略单元测试（FCFS / Priority）。"""
import pytest

from nanovllm.engine.sched.request_queue import (
    FCFSRequestQueue,
    PriorityRequestQueue,
    SchedulingPolicy,
    create_request_queue,
)
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams


def _seq(priority: int = 0) -> Sequence:
    return Sequence([1, 2], SamplingParams(), priority=priority)


# ─── 工厂 ──────────────────────────────────────────────────────────────────────

@pytest.mark.unit
def test_factory_creates_expected_types():
    assert isinstance(create_request_queue(SchedulingPolicy.FCFS), FCFSRequestQueue)
    assert isinstance(create_request_queue(SchedulingPolicy.PRIORITY), PriorityRequestQueue)


# ─── FCFS ──────────────────────────────────────────────────────────────────────

@pytest.mark.unit
def test_fcfs_order_and_prepend():
    q = FCFSRequestQueue()
    a, b, c = _seq(), _seq(), _seq()
    q.add_request(a)
    q.add_request(b)
    assert q.peek_request() is a and len(q) == 2 and bool(q)
    q.prepend_request(c)                     # 抢占回插队首
    assert q.pop_request() is c
    assert q.pop_request() is a
    assert q.pop_request() is b
    assert not q


@pytest.mark.unit
def test_fcfs_contains_and_remove():
    q = FCFSRequestQueue()
    a, b = _seq(), _seq()
    q.add_request(a)
    q.add_request(b)
    assert a in q
    q.remove_request(a)
    assert a not in q and len(q) == 1


# ─── Priority ────────────────────────────────────────────────────────────────

@pytest.mark.unit
def test_priority_pops_lowest_value_first():
    q = PriorityRequestQueue()
    p5, p1, p3 = _seq(5), _seq(1), _seq(3)
    for s in (p5, p1, p3):
        q.add_request(s)
    assert q.pop_request() is p1   # 1 < 3 < 5
    assert q.pop_request() is p3
    assert q.pop_request() is p5


@pytest.mark.unit
def test_priority_ties_broken_by_arrival():
    q = PriorityRequestQueue()
    first, second = _seq(2), _seq(2)         # 同优先级，first 先到（seq_id 更小）
    q.add_request(second)
    q.add_request(first)
    assert q.pop_request() is first          # 同级按 seq_id（到达顺序）


@pytest.mark.unit
def test_priority_contains_remove_iter():
    q = PriorityRequestQueue()
    a, b = _seq(1), _seq(2)
    q.add_request(a)
    q.add_request(b)
    assert a in q and list(q) == [a, b]      # __iter__ 按优先级序
    q.remove_request(a)
    assert a not in q and len(q) == 1
