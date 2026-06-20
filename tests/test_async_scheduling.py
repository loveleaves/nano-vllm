"""
异步调度单元测试（CPU 纯逻辑）：

  - Sequence 占位 token：append_placeholder / resolve_placeholder / truncate_pending
  - EngineCore 异步 vs 同步**等价**：用假执行器（确定 token）驱动两条流水到完成，
    逐请求比对产出 token 流与结束原因——验证 advance/resolve 记账与 EOS 多调度清理

GPU 端 token 前向与真模型等价见 docs/arch_async/testing.md + scripts/gpu_validate_async.py。
"""
import pytest

from nanovllm.engine.core import EngineCore
from nanovllm.engine.core_types import EngineCoreRequest, FinishReason
from nanovllm.engine.sched import Scheduler
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams


# ─── Sequence 占位 token ───────────────────────────────────────────────────────


class TestSequencePlaceholder:

    @pytest.mark.unit
    def test_append_and_resolve(self):
        Sequence.block_size = 4
        seq = Sequence([1, 2, 3])
        seq.append_placeholder()
        assert seq.num_pending == 1 and seq.num_tokens == 4 and seq.last_token == 0
        seq.resolve_placeholder(9)
        assert seq.num_pending == 0 and seq.token_ids[-1] == 9 and seq.last_token == 9

    @pytest.mark.unit
    def test_truncate_pending(self):
        Sequence.block_size = 4
        seq = Sequence([1, 2, 3])
        seq.append_placeholder()
        seq.truncate_pending()
        assert seq.num_pending == 0 and seq.num_tokens == 3 and seq.last_token == 3


# ─── EngineCore 异步 vs 同步等价 ───────────────────────────────────────────────


class _FakeSync:
    def __init__(self, token): self.token = token
    def execute_model(self, seqs, finished_seq_ids=None):
        return [self.token for _ in seqs], None


class _FakeAsync:
    """两槽：模拟"采样张量留 GPU + 跨步回收"，token 为确定值。"""
    def __init__(self, token):
        self.token = token
        self._ai = self._ap = None
    def execute_model_async(self, seqs, finished_seq_ids=None):
        self._ap = {s.seq_id: self.token for s in seqs}
    def resolve_inflight(self):
        return dict(self._ai), None
    def promote_async(self):
        self._ai, self._ap = self._ap, None


def _core(executor, async_mode, block_size=4, num_blocks=64,
          max_num_batched_tokens=64, eos=999) -> EngineCore:
    Sequence.block_size = block_size
    ec = object.__new__(EngineCore)
    ec.executor = executor
    ec.scheduler = Scheduler(num_blocks, block_size, max_num_seqs=8,
                             max_num_batched_tokens=max_num_batched_tokens, eos=eos)
    ec.requests = {}
    ec.async_scheduling = async_mode
    ec._inflight = None
    return ec


def _drive(ec: EngineCore, requests):
    """把请求灌入并驱动到完成，返回 {rid: (token_list, finish_reason)}。"""
    for rid, prompt, sp in requests:
        ec.add_request(EngineCoreRequest(rid, list(prompt), sp))
    collected: dict[str, list[int]] = {rid: [] for rid, _, _ in requests}
    finish: dict[str, FinishReason] = {}
    steps = 0
    while ec.has_unfinished_requests():
        out = ec.step()
        for o in out.outputs:
            collected[o.request_id].extend(o.new_token_ids)
            if o.finished:
                finish[o.request_id] = o.finish_reason
        steps += 1
        assert steps < 1000, "未在合理步数内完成"
    return {rid: (collected[rid], finish.get(rid)) for rid in collected}


def _assert_equiv(requests, token, **core_kw):
    sync = _drive(_core(_FakeSync(token), False, **core_kw), requests)
    # 异步与同步须用各自独立的 Sequence 实例（重新构造请求）
    asyn = _drive(_core(_FakeAsync(token), True, **core_kw), requests)
    assert sync == asyn, f"\nsync={sync}\nasync={asyn}"
    return sync


def _reqs(specs):
    return [(rid, prompt, SamplingParams(**sp)) for rid, prompt, sp in specs]


class TestAsyncEquivalence:

    @pytest.mark.unit
    def test_length_finish_single(self):
        r = _reqs([("r0", range(3), dict(max_tokens=3, ignore_eos=True))])
        res = _assert_equiv(r, token=7)
        assert res["r0"] == ([7, 7, 7], FinishReason.LENGTH)

    @pytest.mark.unit
    def test_eos_finish_single(self):
        r = _reqs([("r0", range(3), dict(max_tokens=10))])
        res = _assert_equiv(r, token=999)
        assert res["r0"] == ([999], FinishReason.STOP)

    @pytest.mark.unit
    def test_multiple_requests(self):
        r = _reqs([
            ("r0", range(3), dict(max_tokens=4, ignore_eos=True)),
            ("r1", range(5), dict(max_tokens=2, ignore_eos=True)),
            ("r2", range(2), dict(max_tokens=6, ignore_eos=True)),
        ])
        res = _assert_equiv(r, token=5)
        assert res["r0"][0] == [5, 5, 5, 5]
        assert res["r1"][0] == [5, 5]
        assert res["r2"][0] == [5] * 6

    @pytest.mark.unit
    def test_chunked_prefill(self):
        # prompt 长、预算小 → 多个 partial prefill chunk（不产 token），再 decode
        r = _reqs([("r0", range(10), dict(max_tokens=3, ignore_eos=True))])
        res = _assert_equiv(r, token=7, max_num_batched_tokens=4)
        assert res["r0"] == ([7, 7, 7], FinishReason.LENGTH)


class TestAsyncMechanics:

    @pytest.mark.unit
    def test_inflight_drains_after_schedule_empty(self):
        ec = _core(_FakeAsync(7), True)
        ec.add_request(EngineCoreRequest("r0", [1, 2, 3],
                                         SamplingParams(max_tokens=1, ignore_eos=True)))
        # step1：下发 prefill，但无在飞结果可回收 → 本步无产出
        out1 = ec.step()
        assert out1.outputs == [] and ec._inflight is not None
        assert ec.has_unfinished_requests()      # 在飞步未排空
        # 后续步回收在飞结果并最终排空
        tokens = []
        while ec.has_unfinished_requests():
            for o in ec.step().outputs:
                tokens.extend(o.new_token_ids)
        assert tokens == [7] and ec._inflight is None
        assert not ec.has_unfinished_requests()
