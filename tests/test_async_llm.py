"""AsyncLLM 流式通路单元测试（CPU，注入假 EngineCore）。

不依赖 pytest-asyncio：用 asyncio.run 驱动协程。验证 generate() 逐步 yield 增量
RequestOutput，且增量累计 == 最终文本，最后一条 finished。
"""
import asyncio

import pytest

from nanovllm.engine.async_llm import AsyncLLM
from nanovllm.engine.core_types import (
    EngineCoreOutput,
    EngineCoreOutputs,
    FinishReason,
)
from nanovllm.engine.output_processor import OutputProcessor
from nanovllm.engine.processor import Processor
from nanovllm.sampling_params import SamplingParams


class FakeTokenizer:
    eos_token_id = 999

    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(i) for i in ids)


class FakeEngineCore:
    """假 EngineCoreClient：实现前端依赖的客户端接口（get_output_async 等）。"""

    def __init__(self, script: list[EngineCoreOutputs]):
        self.script = script
        self.i = 0
        self.added = []
        self.aborted = []

    def add_request(self, req):
        self.added.append(req)

    def has_unfinished_requests(self) -> bool:
        return self.i < len(self.script)

    def get_output(self) -> EngineCoreOutputs:
        out = self.script[self.i]
        self.i += 1
        return out

    async def get_output_async(self) -> EngineCoreOutputs:
        return self.get_output()

    def abort_requests(self, ids):
        self.aborted.extend(ids)


def _make_async_llm(script) -> AsyncLLM:
    tok = FakeTokenizer()
    allm = object.__new__(AsyncLLM)
    allm.tokenizer = tok
    allm.processor = Processor(tok)
    allm.output_processor = OutputProcessor(tok)
    allm.engine_core = FakeEngineCore(script)
    allm.collectors = {}
    allm._pending = []
    allm._handler_task = None
    return allm


@pytest.mark.unit
def test_streaming_yields_incremental_outputs():
    script = [
        EngineCoreOutputs([EngineCoreOutput("r0", [72, 105], finished=False)], 2),
        EngineCoreOutputs([EngineCoreOutput("r0", [33], finished=True,
                                            finish_reason=FinishReason.LENGTH)], -1),
    ]
    allm = _make_async_llm(script)

    async def run():
        deltas, final = [], None
        async for ro in allm.generate([1], SamplingParams(), request_id="r0"):
            deltas.append(ro.delta_text)
            final = ro
        return deltas, final

    deltas, final = asyncio.run(run())
    assert deltas == ["Hi", "!"]
    assert "".join(deltas) == final.text == "Hi!"
    assert final.finished and final.finish_reason is FinishReason.LENGTH
    assert allm.engine_core.added and allm.engine_core.added[0].request_id == "r0"
