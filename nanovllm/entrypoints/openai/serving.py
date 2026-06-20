"""
OpenAI 服务处理器（对齐 vLLM `entrypoints/openai/*/serving.py`）。

把 HTTP 层的 OpenAI 请求翻译为对 AsyncLLM 的调用，再把流式 RequestOutput 组装成
OpenAI 响应：非流式聚合为完整 JSON，流式则产 SSE 文本块（`data: {json}\n\n` + `[DONE]`）。

依赖 nano 的 AsyncLLM.generate（async generator，逐步 yield 增量 RequestOutput）。
"""
import asyncio
import json
from collections.abc import AsyncGenerator

from nanovllm.engine.async_llm import AsyncLLM
from nanovllm.engine.core_types import RequestOutput
from nanovllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    ChatMessage,
    CompletionLogprobs,
    CompletionRequest,
    CompletionResponse,
    CompletionResponseChoice,
    CompletionResponseStreamChoice,
    CompletionStreamResponse,
    DeltaMessage,
    ErrorInfo,
    ErrorResponse,
    UsageInfo,
    random_uuid,
)


class OpenAIServing:
    """处理器基类：持有引擎与模型名，提供错误响应辅助。"""

    def __init__(self, engine: AsyncLLM, model_name: str):
        self.engine = engine
        self.model_name = model_name
        self.tokenizer = engine.tokenizer

    @staticmethod
    def create_error(message: str, code: int = 400, type: str = "invalid_request_error") -> ErrorResponse:
        return ErrorResponse(error=ErrorInfo(message=message, type=type, code=code))

    def _check_model(self, model: str) -> ErrorResponse | None:
        # 宽松匹配：允许客户端传任意模型名（单模型服务），仅在显式不符时报错可选
        return None


def _build_completion_logprobs(tokenizer, token_ids, logprobs, text_offsets) -> CompletionLogprobs:
    """从 RequestOutput.logprobs（list[dict[int,float]]）构造 OpenAI completion logprobs。"""
    out = CompletionLogprobs()
    for tid, lp_map, offset in zip(token_ids, logprobs, text_offsets):
        tok_str = tokenizer.decode([tid])
        out.tokens.append(tok_str)
        out.token_logprobs.append(lp_map.get(tid, 0.0))
        out.top_logprobs.append({tokenizer.decode([k]): v for k, v in lp_map.items()})
        out.text_offset.append(offset)
    return out


class OpenAIServingCompletion(OpenAIServing):

    async def create_completion(self, request: CompletionRequest):
        if request.n != 1:
            return self.create_error("仅支持 n=1")

        prompts = self._normalize_prompts(request.prompt)
        sampling_params = request.to_sampling_params()
        request_id = random_uuid("cmpl")

        if request.stream:
            if len(prompts) != 1:
                return self.create_error("流式模式仅支持单个 prompt")
            return self._stream_generator(request, prompts[0], sampling_params, request_id)

        # 非流式：并发跑完所有 prompt，按 index 组装 choices
        results = await asyncio.gather(*[
            self._collect(prompt, sampling_params, f"{request_id}-{i}")
            for i, prompt in enumerate(prompts)
        ])
        choices: list[CompletionResponseChoice] = []
        usage = UsageInfo()
        for i, final in enumerate(results):
            logprobs = None
            if request.logprobs is not None and final.logprobs:
                offsets = self._text_offsets(final)
                logprobs = _build_completion_logprobs(
                    self.tokenizer, final.token_ids, final.logprobs, offsets)
            choices.append(CompletionResponseChoice(
                index=i, text=final.text, logprobs=logprobs,
                finish_reason=final.finish_reason.value if final.finish_reason else None))
            usage.prompt_tokens += len(final.prompt_token_ids)
            usage.completion_tokens += len(final.token_ids)
        usage.total_tokens = usage.prompt_tokens + usage.completion_tokens
        return CompletionResponse(model=request.model, choices=choices, usage=usage)

    async def _stream_generator(self, request, prompt, sampling_params, request_id) -> AsyncGenerator[str, None]:
        resp_id = random_uuid("cmpl")
        async for ro in self.engine.generate(prompt, sampling_params, request_id):
            chunk = CompletionStreamResponse(
                id=resp_id, model=request.model,
                choices=[CompletionResponseStreamChoice(
                    index=0, text=ro.delta_text,
                    finish_reason=ro.finish_reason.value if ro.finished and ro.finish_reason else None)])
            yield f"data: {chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"

    async def _collect(self, prompt, sampling_params, request_id) -> RequestOutput:
        final = None
        async for ro in self.engine.generate(prompt, sampling_params, request_id):
            final = ro
        return final

    @staticmethod
    def _normalize_prompts(prompt) -> list:
        # str / list[int] → 单 prompt；list[str] / list[list[int]] → 多 prompt
        if isinstance(prompt, str):
            return [prompt]
        if isinstance(prompt, list) and prompt and isinstance(prompt[0], int):
            return [prompt]
        if isinstance(prompt, list):
            return list(prompt)
        return [prompt]

    def _text_offsets(self, ro: RequestOutput) -> list[int]:
        offsets, acc = [], 0
        for tid in ro.token_ids:
            offsets.append(acc)
            acc += len(self.tokenizer.decode([tid]))
        return offsets


class OpenAIServingChat(OpenAIServing):

    async def create_chat_completion(self, request: ChatCompletionRequest):
        if request.n != 1:
            return self.create_error("仅支持 n=1")

        prompt = self._render_chat_prompt(request)
        sampling_params = request.to_sampling_params()
        request_id = random_uuid("chatcmpl")

        if request.stream:
            return self._stream_generator(request, prompt, sampling_params, request_id)

        final = None
        async for ro in self.engine.generate(prompt, sampling_params, request_id):
            final = ro
        usage = UsageInfo(
            prompt_tokens=len(final.prompt_token_ids),
            completion_tokens=len(final.token_ids),
            total_tokens=len(final.prompt_token_ids) + len(final.token_ids))
        choice = ChatCompletionResponseChoice(
            index=0, message=ChatMessage(role="assistant", content=final.text),
            finish_reason=final.finish_reason.value if final.finish_reason else None)
        return ChatCompletionResponse(model=request.model, choices=[choice], usage=usage)

    async def _stream_generator(self, request, prompt, sampling_params, request_id) -> AsyncGenerator[str, None]:
        resp_id = random_uuid("chatcmpl")
        # 首块：role 声明（对齐 OpenAI 流式语义）
        first = ChatCompletionStreamResponse(
            id=resp_id, model=request.model,
            choices=[ChatCompletionResponseStreamChoice(
                index=0, delta=DeltaMessage(role="assistant", content=""))])
        yield f"data: {first.model_dump_json()}\n\n"
        async for ro in self.engine.generate(prompt, sampling_params, request_id):
            chunk = ChatCompletionStreamResponse(
                id=resp_id, model=request.model,
                choices=[ChatCompletionResponseStreamChoice(
                    index=0, delta=DeltaMessage(content=ro.delta_text),
                    finish_reason=ro.finish_reason.value if ro.finished and ro.finish_reason else None)])
            yield f"data: {chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"

    def _render_chat_prompt(self, request: ChatCompletionRequest) -> str:
        messages = [{"role": m.role, "content": m.content or ""} for m in request.messages]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=request.add_generation_prompt)
