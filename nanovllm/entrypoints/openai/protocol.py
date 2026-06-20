"""
OpenAI 兼容协议层（对齐 vLLM `entrypoints/openai/*/protocol.py`）。

用 Pydantic v2 定义 /v1/completions、/v1/chat/completions、/v1/models 的请求/响应模型，
并把 OpenAI 采样参数映射到 nano-vllm 的 SamplingParams。

范围裁剪（教学/单机）：n 仅支持 1；不支持 logit_bias/tools/function_call；
chat 不产 logprobs（completion 产基础 logprobs）。
"""
import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field

from nanovllm.sampling_params import SamplingParams


def random_uuid(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex}"


# ─── 通用 ────────────────────────────────────────────────────────────────────
class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ErrorInfo(BaseModel):
    message: str
    type: str = "invalid_request_error"
    code: int = 400
    param: str | None = None


class ErrorResponse(BaseModel):
    error: ErrorInfo


# ─── /v1/models ──────────────────────────────────────────────────────────────
class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "nano-vllm"


class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelCard] = Field(default_factory=list)


# ─── 采样参数基类（completion / chat 共用映射逻辑） ────────────────────────────
class _SamplingMixin(BaseModel):
    max_tokens: int | None = 64
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    min_p: float = 0.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: float = 1.0
    seed: int | None = None
    stop: str | list[str] | None = None
    ignore_eos: bool = False
    n: int = 1

    def to_sampling_params(self) -> SamplingParams:
        return SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_tokens or 64,
            ignore_eos=self.ignore_eos,
            stop=self.stop,
            top_p=self.top_p,
            top_k=self.top_k,
            min_p=self.min_p,
            presence_penalty=self.presence_penalty,
            frequency_penalty=self.frequency_penalty,
            repetition_penalty=self.repetition_penalty,
            seed=self.seed,
            logprobs=getattr(self, "_logprobs_n", None),
        )


# ─── /v1/completions ─────────────────────────────────────────────────────────
class CompletionRequest(_SamplingMixin):
    model: str
    prompt: str | list[int] | list[str] | list[list[int]]
    stream: bool = False
    logprobs: int | None = None

    def to_sampling_params(self) -> SamplingParams:
        self._logprobs_n = self.logprobs
        return super().to_sampling_params()


class CompletionLogprobs(BaseModel):
    tokens: list[str] = Field(default_factory=list)
    token_logprobs: list[float] = Field(default_factory=list)
    top_logprobs: list[dict[str, float]] = Field(default_factory=list)
    text_offset: list[int] = Field(default_factory=list)


class CompletionResponseChoice(BaseModel):
    index: int
    text: str
    logprobs: CompletionLogprobs | None = None
    finish_reason: str | None = None


class CompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: random_uuid("cmpl"))
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[CompletionResponseChoice]
    usage: UsageInfo


class CompletionResponseStreamChoice(BaseModel):
    index: int
    text: str
    logprobs: CompletionLogprobs | None = None
    finish_reason: str | None = None


class CompletionStreamResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[CompletionResponseStreamChoice]


# ─── /v1/chat/completions ────────────────────────────────────────────────────
class ChatMessage(BaseModel):
    role: str
    content: str | None = None


class ChatCompletionRequest(_SamplingMixin):
    model: str
    messages: list[ChatMessage]
    stream: bool = False
    add_generation_prompt: bool = True


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str | None = None


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: random_uuid("chatcmpl"))
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionResponseChoice]
    usage: UsageInfo


class DeltaMessage(BaseModel):
    role: str | None = None
    content: str | None = None


class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: str | None = None


class ChatCompletionStreamResponse(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionResponseStreamChoice]
