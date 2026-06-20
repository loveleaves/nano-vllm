"""EngineCore 单元测试（CPU）。

EngineCore 真实构造会经 Executor 拉起 Worker/ModelRunner（GPU + 多进程），故这里用
object.__new__ 绕过 __init__，注入真实 Scheduler（纯 Python）+ 假 Executor，只检验
add_request/step/abort 的调度执行逻辑与 EngineCoreOutputs 产出。
"""
import pytest

from nanovllm.engine.core import EngineCore
from nanovllm.engine.core_types import EngineCoreRequest, FinishReason
from nanovllm.engine.sched import Scheduler
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams


class FakeExecutor:
    """execute_model 为每个 seq 返回固定 token（模拟采样结果），不触碰 GPU。"""

    def __init__(self, token: int):
        self.token = token

    def execute_model(self, seqs, finished_seq_ids=None):
        return [self.token for _ in seqs], None


def _make_core(token: int, block_size=4, num_blocks=32,
               max_num_batched_tokens=64, eos=999) -> EngineCore:
    Sequence.block_size = block_size
    ec = object.__new__(EngineCore)
    ec.executor = FakeExecutor(token)
    ec.scheduler = Scheduler(num_blocks, block_size, max_num_seqs=8,
                             max_num_batched_tokens=max_num_batched_tokens, eos=eos)
    ec.requests = {}
    ec.async_scheduling = False
    ec._inflight = None
    return ec


def _add(ec: EngineCore, rid: str, prompt, **sp):
    ec.add_request(EngineCoreRequest(rid, list(prompt), SamplingParams(**sp)))


@pytest.mark.unit
def test_prefill_then_length_finish():
    ec = _make_core(token=7)
    _add(ec, "r0", range(3), max_tokens=2, ignore_eos=True)

    # step1：prefill 覆盖整个 prompt 并产出首个 token
    out1 = ec.step()
    (o1,) = out1.outputs
    assert o1.new_token_ids == [7] and not o1.finished
    assert out1.num_tokens == 3                       # 含 prefill

    # step2：纯 decode，达到 max_tokens → LENGTH
    out2 = ec.step()
    (o2,) = out2.outputs
    assert o2.finished and o2.finish_reason is FinishReason.LENGTH
    assert out2.num_tokens == -1                      # 纯 decode 1 seq
    assert not ec.has_unfinished_requests()
    assert ec.requests == {}


@pytest.mark.unit
def test_eos_gives_stop_reason():
    ec = _make_core(token=999)                        # 返回 eos
    _add(ec, "r0", range(3), max_tokens=10)
    out = ec.step()
    (o,) = out.outputs
    assert o.finished and o.finish_reason is FinishReason.STOP


@pytest.mark.unit
def test_chunked_prefill_emits_no_token_until_complete():
    # prompt 6 token，预算 4：首步只 prefill 半个 prompt，不产 token
    ec = _make_core(token=7, max_num_batched_tokens=4)
    _add(ec, "r0", range(6), max_tokens=5, ignore_eos=True)

    out1 = ec.step()
    assert out1.outputs == [] and out1.num_tokens == 4   # 半个 prefill chunk

    out2 = ec.step()
    (o2,) = out2.outputs                                  # 余下 chunk 完成 → 产 token
    assert o2.new_token_ids == [7] and not o2.finished


@pytest.mark.unit
def test_abort_releases_request():
    ec = _make_core(token=7)
    _add(ec, "r0", range(3), max_tokens=10, ignore_eos=True)
    ec.step()
    assert ec.has_unfinished_requests()

    ec.abort_requests(["r0"])
    assert not ec.has_unfinished_requests()
    assert "r0" not in ec.requests


@pytest.mark.unit
def test_empty_schedule_returns_empty_outputs():
    ec = _make_core(token=7)
    out = ec.step()
    assert out.outputs == [] and out.num_tokens == 0
