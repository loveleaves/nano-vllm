"""
输入处理组件（对齐 vLLM V1 `v1/engine/processor.py` + `input_processor`）。

职责：把用户层输入（str / list[int] + SamplingParams）规范化成 EngineCore 可直接
消费的 EngineCoreRequest。在 V1 中此步还包含多模态预处理 / LoRA 解析等，nano 仅保留
最核心的 tokenize + request_id 分配。

与 OutputProcessor 共享同一个 tokenizer 实例（前者 encode，后者 decode）。
"""
from itertools import count

from nanovllm.engine.core_types import EngineCoreRequest
from nanovllm.sampling_params import SamplingParams


class Processor:

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self._counter = count()

    def process_inputs(
        self,
        prompt: str | list[int],
        sampling_params: SamplingParams,
        request_id: str | None = None,
    ) -> EngineCoreRequest:
        """规范化单条输入为 EngineCoreRequest。

        prompt 为 str 时用 tokenizer 编码；为 list[int] 时视作已 tokenize 的 id。
        request_id 缺省时自增分配（保证唯一）。
        """
        if request_id is None:
            request_id = str(next(self._counter))
        if isinstance(prompt, str):
            prompt_token_ids = self.tokenizer.encode(prompt)
        else:
            prompt_token_ids = list(prompt)
        assert prompt_token_ids, "prompt 不能为空"
        return EngineCoreRequest(
            request_id=request_id,
            prompt_token_ids=prompt_token_ids,
            sampling_params=sampling_params,
        )
