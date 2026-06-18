"""
输出处理组件（对齐 vLLM V1 `v1/engine/output_processor.py`）。

把 EngineCore 产出的"裸 token 增量"转成面向用户的 RequestOutput：
  1. 增量 detokenize（IncrementalDetokenizer）
  2. 停止串检测（命中则截断文本、标记结束、回报需 abort 的 request_id）
  3. 维护每请求状态（RequestState），结束后清理

EngineCore 不感知文本，文本相关逻辑全部收敛在此（与 V1 一致）。
"""
from dataclasses import dataclass, field

from nanovllm.engine.core_types import (
    EngineCoreOutput,
    EngineCoreRequest,
    FinishReason,
    RequestOutput,
)
from nanovllm.engine.detokenizer import IncrementalDetokenizer, check_stop_strings


class RequestState:
    """单个请求的输出侧状态。"""

    def __init__(self, request: EngineCoreRequest, tokenizer):
        self.request_id = request.request_id
        self.prompt_token_ids = request.prompt_token_ids
        self.sampling_params = request.sampling_params
        self.detokenizer = IncrementalDetokenizer(tokenizer)


@dataclass
class OutputProcessorOutput:
    """一步输出处理的产物。

    reqs_to_abort：因停止串等在输出侧判定结束、需通知 EngineCore 释放的 request_id。
    """
    request_outputs: list[RequestOutput] = field(default_factory=list)
    reqs_to_abort: list[str] = field(default_factory=list)


class OutputProcessor:

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.request_states: dict[str, RequestState] = {}

    def add_request(self, request: EngineCoreRequest):
        self.request_states[request.request_id] = RequestState(request, self.tokenizer)

    def has_unfinished_requests(self) -> bool:
        return bool(self.request_states)

    def process_outputs(
        self, engine_outputs: list[EngineCoreOutput]
    ) -> OutputProcessorOutput:
        request_outputs: list[RequestOutput] = []
        reqs_to_abort: list[str] = []

        for out in engine_outputs:
            st = self.request_states.get(out.request_id)
            if st is None:
                continue

            prev_len = len(st.detokenizer.text)
            delta = st.detokenizer.update(out.new_token_ids)
            cum_text = st.detokenizer.text
            token_ids = list(st.detokenizer.output_token_ids)
            finished = out.finished
            finish_reason = out.finish_reason

            # 停止串：在累计文本中检测，命中则截断（不含停止串）并提前结束
            stop_str = check_stop_strings(cum_text, st.sampling_params.stop)
            if stop_str is not None and not finished:
                pos = cum_text.find(stop_str)
                cum_text = cum_text[:pos]
                delta = cum_text[prev_len:] if len(cum_text) > prev_len else ""
                finished = True
                finish_reason = FinishReason.STOP
                reqs_to_abort.append(out.request_id)

            request_outputs.append(RequestOutput(
                request_id=out.request_id,
                prompt_token_ids=st.prompt_token_ids,
                token_ids=token_ids,
                text=cum_text,
                delta_text=delta,
                finished=finished,
                finish_reason=finish_reason,
            ))
            if finished:
                self.request_states.pop(out.request_id, None)

        return OutputProcessorOutput(request_outputs, reqs_to_abort)
