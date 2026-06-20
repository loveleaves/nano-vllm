"""
EngineCore 客户端 / 进程化单测（CPU，免 GPU）。

策略：
  - busy_loop / _handle_input 是进程拓扑的真实逻辑，用假核心 + queue.Queue 在进程内直测；
  - MPClient 端到端用 **fork** 上下文 + 假核心子进程跑通（真跨进程队列收发，无 GPU/CUDA）；
    生产默认 spawn + 真 EngineCore，由 GPU 集成覆盖。

运行：pytest tests/test_core_client.py -m unit -v
"""
import multiprocessing as mp
import queue
import threading

import pytest

from nanovllm.engine.core_client import (
    ABORT,
    ADD,
    EXIT,
    OUTPUTS,
    EngineCoreProc,
    InprocClient,
    MPClient,
)
from nanovllm.engine.core_types import (
    EngineCoreOutput,
    EngineCoreOutputs,
    EngineCoreRequest,
    FinishReason,
)
from nanovllm.sampling_params import SamplingParams


# ─── 假核心：每请求每步吐 1 token，满 2 token 即结束（无 GPU） ──────────────────
class FakeProcCore:
    """模拟 EngineCore：add/abort/has_unfinished/step/get_stats/exit。"""

    TOKENS_PER_REQ = 2

    def __init__(self, config=None):
        self.remaining: dict[str, int] = {}
        self.exited = False

    def add_request(self, req: EngineCoreRequest):
        self.remaining[req.request_id] = self.TOKENS_PER_REQ

    def abort_requests(self, ids):
        for i in ids:
            self.remaining.pop(i, None)

    def has_unfinished_requests(self) -> bool:
        return bool(self.remaining)

    def step(self) -> EngineCoreOutputs:
        outputs = []
        for rid in list(self.remaining):
            self.remaining[rid] -= 1
            finished = self.remaining[rid] == 0
            if finished:
                self.remaining.pop(rid)
            outputs.append(EngineCoreOutput(
                request_id=rid, new_token_ids=[42], finished=finished,
                finish_reason=FinishReason.LENGTH if finished else None))
        return EngineCoreOutputs(outputs=outputs, num_tokens=-len(outputs))

    def get_stats(self):
        return None

    def exit(self):
        self.exited = True


def _req(rid: str) -> EngineCoreRequest:
    return EngineCoreRequest(request_id=rid, prompt_token_ids=[1, 2, 3],
                             sampling_params=SamplingParams())


# ─── 协议 / busy_loop（进程内） ───────────────────────────────────────────────
class TestBusyLoop:

    @pytest.mark.unit
    def test_handle_input_add_abort_exit(self):
        core = FakeProcCore()
        assert EngineCoreProc._handle_input(core, (ADD, _req("r0"))) is True
        assert core.has_unfinished_requests()
        assert EngineCoreProc._handle_input(core, (ABORT, ["r0"])) is True
        assert not core.has_unfinished_requests()
        assert EngineCoreProc._handle_input(core, (EXIT, None)) is False

    @pytest.mark.unit
    def test_busy_loop_runs_to_finish(self):
        core = FakeProcCore()
        in_q, out_q = queue.Queue(), queue.Queue()
        t = threading.Thread(target=EngineCoreProc.busy_loop,
                             args=(core, in_q, out_q), daemon=True)
        t.start()
        in_q.put((ADD, _req("r0")))

        tokens, finished = [], False
        while not finished:
            msg_type, payload = out_q.get(timeout=5)
            assert msg_type == OUTPUTS
            outs, _stats = payload
            for o in outs.outputs:
                tokens.extend(o.new_token_ids)
                finished = finished or o.finished
        assert tokens == [42, 42]

        in_q.put((EXIT, None))
        t.join(timeout=5)
        assert not t.is_alive()

    @pytest.mark.unit
    def test_busy_loop_idle_blocks_then_exits(self):
        """空闲时阻塞在 input_queue，不空转；EXIT 能从空闲态唤醒退出。"""
        core = FakeProcCore()
        in_q, out_q = queue.Queue(), queue.Queue()
        t = threading.Thread(target=EngineCoreProc.busy_loop,
                             args=(core, in_q, out_q), daemon=True)
        t.start()
        # 无任何请求：循环应阻塞、不产出
        with pytest.raises(queue.Empty):
            out_q.get(timeout=0.3)
        in_q.put((EXIT, None))
        t.join(timeout=5)
        assert not t.is_alive()


# ─── InprocClient 委派 ───────────────────────────────────────────────────────
class TestInprocClient:

    @pytest.mark.unit
    def test_delegates_to_core(self):
        c = object.__new__(InprocClient)
        c.engine_core = FakeProcCore()
        c.add_request(_req("r0"))
        assert c.has_unfinished_requests()
        out = c.get_output()          # 委派 step()
        assert out.outputs[0].request_id == "r0"
        c.abort_requests(["r0"])
        assert not c.has_unfinished_requests()
        c.exit()
        assert c.engine_core.exited


# ─── MPClient 端到端（fork 子进程，真跨进程队列） ─────────────────────────────
class TestMPClientFork:

    @pytest.mark.unit
    def test_end_to_end(self):
        ctx = mp.get_context("fork")
        client = MPClient(None, ctx=ctx, core_factory=FakeProcCore, ready_timeout=10)
        try:
            client.add_request(_req("r0"))
            client.add_request(_req("r1"))
            assert client.has_unfinished_requests()

            collected: dict[str, list[int]] = {}
            while client.has_unfinished_requests():
                outs = client.get_output()
                for o in outs.outputs:
                    collected.setdefault(o.request_id, []).extend(o.new_token_ids)
            assert set(collected) == {"r0", "r1"}
            assert all(v == [42, 42] for v in collected.values())
        finally:
            client.exit()
        assert client.proc is None

    @pytest.mark.unit
    def test_abort_clears_unfinished(self):
        ctx = mp.get_context("fork")
        client = MPClient(None, ctx=ctx, core_factory=FakeProcCore, ready_timeout=10)
        try:
            client.add_request(_req("r0"))
            assert client.has_unfinished_requests()
            client.abort_requests(["r0"])
            assert not client.has_unfinished_requests()
        finally:
            client.exit()
