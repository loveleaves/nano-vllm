"""
引擎层数据契约（对齐 vLLM V1 `v1/engine/__init__.py` 的 EngineCore* 结构体）。

V1 把"前端 ↔ EngineCore ↔ 输出处理"之间传递的对象固化成显式结构体，使三段
逻辑（输入处理 / 调度执行 / 输出处理）可以独立演进、独立测试。nano 沿用同一组
契约，但因 EngineCore 与前端同进程（见 design.md），用普通 dataclass 即可，无需
msgspec.Struct 的跨进程编码。

数据流：
    Processor      ──► EngineCoreRequest ──► EngineCore.add_request
    EngineCore.step ──► EngineCoreOutputs(outputs=[EngineCoreOutput,...])
    OutputProcessor ──► RequestOutput（面向用户，含增量/累计文本）
"""
from dataclasses import dataclass, field
from enum import Enum

from nanovllm.sampling_params import SamplingParams


class FinishReason(str, Enum):
    """请求终止原因（值即对外字符串，对齐 vLLM 的 "stop"/"length"/"abort"）。"""
    STOP = "stop"        # 命中 EOS 或停止串
    LENGTH = "length"    # 达到 max_tokens
    ABORT = "abort"      # 被显式中止

    def __str__(self) -> str:
        return self.value


@dataclass
class EngineCoreRequest:
    """Processor 产出、喂给 EngineCore 的一次请求（prompt 已 tokenize）。"""
    request_id: str
    prompt_token_ids: list[int]
    sampling_params: SamplingParams
    priority: int = 0   # 调度优先级（值越小越先；仅 priority 策略生效）
    grammar: object = None   # 引导解码 Grammar（Processor 用 tokenizer 构造；仅 UniProc）


@dataclass
class EngineCoreOutput:
    """EngineCore 单步为某个请求产出的增量结果。"""
    request_id: str
    new_token_ids: list[int]
    finished: bool = False
    finish_reason: FinishReason | None = None
    # 本步采样 token 的 logprobs：token_id → logprob（含采样 token + top-k），
    # 仅当请求设置了 SamplingParams.logprobs 时非 None
    logprobs: dict[int, float] | None = None


@dataclass
class EngineCoreOutputs:
    """EngineCore.step() 的整步产出。

    num_tokens 仅用于吞吐展示：>0 表示该步含 prefill（token 总数），
    <0 表示纯 decode（-序列数），0 表示空步。
    """
    outputs: list[EngineCoreOutput] = field(default_factory=list)
    num_tokens: int = 0


@dataclass
class RequestOutput:
    """面向用户的请求输出（对齐 vLLM RequestOutput 的关键字段，扁平化）。

    text       — 累计文本（cumulative）
    delta_text — 本步新增文本（streaming 增量，AsyncLLM 用）
    """
    request_id: str
    prompt_token_ids: list[int]
    token_ids: list[int]
    text: str
    delta_text: str = ""
    finished: bool = False
    finish_reason: FinishReason | None = None
    # 每个输出 token 一个 dict（token_id → logprob，含采样 token + top-k）；未请求时为 None
    logprobs: list[dict[int, float]] | None = None
