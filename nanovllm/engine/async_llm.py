"""
AsyncLLM：异步流式入口（对齐 vLLM V1 `v1/engine/async_llm.py`）。

与同步 LLMEngine 共用同一套组件（Processor / EngineCore / OutputProcessor），但对外
暴露 **async generator**：每步 yield 增量 RequestOutput（`delta_text` 为本步新增文本），
适合 streaming 场景。

并发模型（单进程、教学化简）：
  - 一个后台 output_handler 协程驱动循环；
  - 每步把阻塞的 `EngineCore.step()`（GPU + 多进程 TP）丢进默认线程池执行，
    torch 在 kernel 期间释放 GIL，故事件循环不被阻塞；
  - 新请求经 _pending 列表交给 handler，在 **handler 协程内**（非线程内）落入
    Scheduler，从而与正在执行的 step 天然错开，避免对调度结构的并发改写。
"""
import asyncio
from dataclasses import fields

from transformers import AutoTokenizer

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.core import EngineCore
from nanovllm.engine.core_types import EngineCoreRequest, RequestOutput
from nanovllm.engine.processor import Processor
from nanovllm.engine.output_processor import OutputProcessor


class RequestOutputCollector:
    """单请求的异步输出队列（对齐 V1 RequestOutputCollector）。"""

    def __init__(self):
        self.queue: asyncio.Queue = asyncio.Queue()

    def put(self, output: RequestOutput):
        self.queue.put_nowait(output)

    async def get(self) -> RequestOutput:
        return await self.queue.get()


class AsyncLLM:

    def __init__(self, model: str, **kwargs):
        config_fields = {f.name for f in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)

        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id

        self.processor = Processor(self.tokenizer)
        self.engine_core = EngineCore(config)
        self.output_processor = OutputProcessor(self.tokenizer)

        self.collectors: dict[str, RequestOutputCollector] = {}
        self._pending: list[EngineCoreRequest] = []
        self._handler_task: asyncio.Task | None = None

    def exit(self):
        self.engine_core.exit()

    # ── 后台输出循环 ──────────────────────────────────────────────────────────
    def _ensure_handler(self):
        if self._handler_task is None or self._handler_task.done():
            self._handler_task = asyncio.create_task(self._run_output_handler())

    async def _run_output_handler(self):
        loop = asyncio.get_running_loop()
        while True:
            # 在 handler 协程内把新请求落入 Scheduler（与 step 错开，无并发写）
            while self._pending:
                self.engine_core.add_request(self._pending.pop(0))
            if not self.engine_core.has_unfinished_requests():
                break  # 空闲：退出，新请求到来时由 _ensure_handler 重启

            core_outputs = await loop.run_in_executor(None, self.engine_core.step)
            processed = self.output_processor.process_outputs(core_outputs.outputs)
            if processed.reqs_to_abort:
                self.engine_core.abort_requests(processed.reqs_to_abort)
            for ro in processed.request_outputs:
                collector = self.collectors.get(ro.request_id)
                if collector is not None:
                    collector.put(ro)
                    if ro.finished:
                        self.collectors.pop(ro.request_id, None)

    # ── 对外接口 ──────────────────────────────────────────────────────────────
    async def add_request(self, prompt: str | list[int],
                          sampling_params: SamplingParams,
                          request_id: str | None = None) -> RequestOutputCollector:
        req = self.processor.process_inputs(prompt, sampling_params, request_id)
        collector = RequestOutputCollector()
        self.collectors[req.request_id] = collector
        self.output_processor.add_request(req)
        self._pending.append(req)
        self._ensure_handler()
        return collector

    async def generate(self, prompt: str | list[int],
                       sampling_params: SamplingParams,
                       request_id: str | None = None):
        """异步生成器：逐步 yield 增量 RequestOutput，直到 finished。"""
        collector = await self.add_request(prompt, sampling_params, request_id)
        while True:
            ro = await collector.get()
            yield ro
            if ro.finished:
                break
