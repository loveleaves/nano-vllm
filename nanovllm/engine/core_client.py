"""
EngineCore 客户端抽象（对齐 vLLM V1 `v1/engine/core_client.py`）。

V1 把"前端 ↔ EngineCore"的耦合收敛到一个 EngineCoreClient 抽象，并提供两种实现：
  - InprocClient：EngineCore 与前端同进程，get_output() 直接驱动 step()（offline/调试）。
  - MPClient：EngineCore 跑在独立子进程的 busy-loop 里，前端经队列收发（生产/服务）。

nano 沿用同一抽象，但**用 stdlib multiprocessing.Queue 替代 ZMQ**（与 Worker 层用
SharedMemory 替代 ZMQ 的取舍一致，见 [[arch_worker_isolation]]）。进程拓扑由此对齐 V1：

    [前端进程] Processor/OutputProcessor + 客户端
        │  mp.Queue（pickle）          ← MPClient
        ▼
    [EngineCore 进程] busy-loop：Scheduler + Executor
        │  (UniProc 内联 rank0 / MultiProc：ShmTransport+ResultChannel)
        ▼
    [Worker 进程 × N]

要点：
  - 前端不再驱动 step；EngineCore 子进程自跑 busy-loop，前端只 add_request / 收 outputs。
  - has_unfinished 由客户端本地跟踪（add 时计入、收到 finished 或 abort 时移除），
    无需往返查询核心。
  - tokenize/detokenize 始终在前端；跨进程只传 EngineCoreRequest / EngineCoreOutputs
    （纯 int / dataclass，pickle 友好）。
"""
import asyncio
import multiprocessing as mp
import queue
import traceback
from abc import ABC, abstractmethod

from nanovllm.config import Config
from nanovllm.engine.core import EngineCore
from nanovllm.engine.core_types import EngineCoreOutputs, EngineCoreRequest

# ── 消息类型（前端 → 核心 / 核心 → 前端） ───────────────────────────────────────
ADD = "ADD"        # (ADD, EngineCoreRequest)
ABORT = "ABORT"    # (ABORT, list[str])
EXIT = "EXIT"      # (EXIT, None)
READY = "READY"    # (READY, None)  核心初始化完成
OUTPUTS = "OUTPUTS"  # (OUTPUTS, (EngineCoreOutputs, stats))
ERROR = "ERROR"    # (ERROR, traceback_str)


class EngineCoreClient(ABC):
    """前端持有的 EngineCore 句柄。两实现：InprocClient / MPClient。"""

    @staticmethod
    def make_client(config: Config) -> "EngineCoreClient":
        if config.multiproc_engine_core:
            return MPClient(config)
        return InprocClient(config)

    @abstractmethod
    def get_output(self) -> EngineCoreOutputs: ...

    async def get_output_async(self) -> EngineCoreOutputs:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.get_output)

    @abstractmethod
    def add_request(self, request: EngineCoreRequest) -> None: ...

    @abstractmethod
    def abort_requests(self, request_ids: list[str]) -> None: ...

    @abstractmethod
    def has_unfinished_requests(self) -> bool: ...

    def get_stats(self):
        return None

    def exit(self) -> None: ...


class InprocClient(EngineCoreClient):
    """同进程实现：直接持 EngineCore，get_output 即 step（默认路径，行为零变化）。"""

    def __init__(self, config: Config):
        self.engine_core = EngineCore(config)

    def get_output(self) -> EngineCoreOutputs:
        return self.engine_core.step()

    def add_request(self, request: EngineCoreRequest) -> None:
        self.engine_core.add_request(request)

    def abort_requests(self, request_ids: list[str]) -> None:
        self.engine_core.abort_requests(request_ids)

    def has_unfinished_requests(self) -> bool:
        return self.engine_core.has_unfinished_requests()

    def get_stats(self):
        return self.engine_core.get_stats()

    def exit(self) -> None:
        self.engine_core.exit()


# ── EngineCore 子进程：busy-loop + 进程入口 ───────────────────────────────────
class EngineCoreProc:
    """把 EngineCore 包成"独立进程 busy-loop"（对齐 V1 EngineCoreProc）。

    busy_loop 与传输无关（吃 input/output 两个队列 + 一个 core），可在进程内用
    queue.Queue + 假 core 直接单测；真实运行时 run_engine_core_proc 在子进程里跑它。
    """

    @staticmethod
    def _handle_input(core: EngineCore, msg) -> bool:
        """处理一条前端指令；返回 False 表示收到 EXIT、应退出循环。"""
        msg_type, payload = msg
        if msg_type == ADD:
            core.add_request(payload)
        elif msg_type == ABORT:
            core.abort_requests(payload)
        elif msg_type == EXIT:
            return False
        return True

    @staticmethod
    def busy_loop(core: EngineCore, input_queue, output_queue) -> None:
        """核心循环：空闲时阻塞等输入；有活时每步 drain 输入 → step → 投递非空产出。"""
        while True:
            # 空闲：阻塞直到有新请求（避免空转）。
            if not core.has_unfinished_requests():
                if not EngineCoreProc._handle_input(core, input_queue.get()):
                    return
            # 把已到达的其余指令一次性吸收（abort / 追加请求），不阻塞。
            while True:
                try:
                    msg = input_queue.get_nowait()
                except queue.Empty:
                    break
                if not EngineCoreProc._handle_input(core, msg):
                    return
            # 推进一步；仅在产出非空时回传（空步=chunked prefill 中途/空调度，前端续等）。
            if core.has_unfinished_requests():
                outputs = core.step()
                if outputs.outputs:
                    output_queue.put((OUTPUTS, (outputs, core.get_stats())))


def run_engine_core_proc(config: Config, input_queue, output_queue,
                         core_factory=EngineCore) -> None:
    """EngineCore 子进程入口：建核心 → 报 READY → 跑 busy-loop → 退出时回收。

    core_factory 仅供测试注入假核心（默认真 EngineCore）。任何阶段异常都回传
    ERROR + traceback，避免前端在队列上永久阻塞。
    """
    core = None
    try:
        core = core_factory(config)
        output_queue.put((READY, None))
    except Exception:
        output_queue.put((ERROR, traceback.format_exc()))
        return
    try:
        EngineCoreProc.busy_loop(core, input_queue, output_queue)
    except Exception:
        output_queue.put((ERROR, traceback.format_exc()))
    finally:
        try:
            core.exit()
        except Exception:
            pass


class MPClient(EngineCoreClient):
    """独立进程实现：spawn EngineCore 子进程，经 mp.Queue 收发。

    has_unfinished 本地跟踪：add_request 计入 request_id，收到 finished 输出或
    abort 时移除——故不需要向子进程往返查询。
    """

    def __init__(self, config: Config, ctx=None, core_factory=EngineCore,
                 ready_timeout: float = 600.0):
        self.ctx = ctx or mp.get_context("spawn")
        self.input_queue = self.ctx.Queue()
        self.output_queue = self.ctx.Queue()
        self.proc = self.ctx.Process(
            target=run_engine_core_proc,
            args=(config, self.input_queue, self.output_queue, core_factory),
            daemon=False,
        )
        self.proc.start()

        self._unfinished: set[str] = set()
        self._stats = None

        # 等待核心初始化完成（含 Executor/Worker 建模 + warmup）。
        msg_type, payload = self.output_queue.get(timeout=ready_timeout)
        if msg_type == ERROR:
            self.proc.join(timeout=5)
            raise RuntimeError(f"EngineCore 子进程初始化失败:\n{payload}")
        assert msg_type == READY, f"预期 READY，收到 {msg_type}"

    def add_request(self, request: EngineCoreRequest) -> None:
        self._unfinished.add(request.request_id)
        self.input_queue.put((ADD, request))

    def abort_requests(self, request_ids: list[str]) -> None:
        for rid in request_ids:
            self._unfinished.discard(rid)
        self.input_queue.put((ABORT, list(request_ids)))

    def has_unfinished_requests(self) -> bool:
        return bool(self._unfinished)

    def get_output(self) -> EngineCoreOutputs:
        msg_type, payload = self.output_queue.get()
        if msg_type == ERROR:
            raise RuntimeError(f"EngineCore 子进程异常:\n{payload}")
        assert msg_type == OUTPUTS, f"预期 OUTPUTS，收到 {msg_type}"
        outputs, stats = payload
        self._stats = stats
        for o in outputs.outputs:
            if o.finished:
                self._unfinished.discard(o.request_id)
        return outputs

    def get_stats(self):
        return self._stats

    def exit(self) -> None:
        if getattr(self, "proc", None) is None:
            return
        if self.proc.is_alive():
            try:
                self.input_queue.put((EXIT, None))
            except Exception:
                pass
            self.proc.join(timeout=30)
            if self.proc.is_alive():
                self.proc.terminate()
        self.proc = None
