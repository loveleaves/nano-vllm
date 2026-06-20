from dataclasses import fields
from time import perf_counter

from tqdm.auto import tqdm
from transformers import AutoTokenizer

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.core_client import EngineCoreClient
from nanovllm.engine.core_types import RequestOutput
from nanovllm.engine.processor import Processor
from nanovllm.engine.output_processor import OutputProcessor


class LLMEngine:
    """
    同步推理引擎入口（V1 风格组件装配，见 docs/arch_engine/design.md）。

    分层（对齐 vLLM V1 LLMEngine）：
      Processor         — 输入处理：tokenize → EngineCoreRequest
      EngineCoreClient  — 调度 + 执行核心的句柄（InprocClient 同进程 / MPClient 独立进程，
                          由 config.multiproc_engine_core 选择；前者驱动 step，后者收 busy-loop 产出）
      OutputProcessor   — 增量 detokenize / 停止串 / finish reason → RequestOutput

    本类只做装配与编排，不含调度、kernel 或文本处理细节。流式/异步入口见 AsyncLLM。
    """

    def __init__(self, model: str, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)

        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id

        self.processor = Processor(self.tokenizer)
        self.engine_core = EngineCoreClient.make_client(config)
        self.output_processor = OutputProcessor(self.tokenizer)

    def exit(self):
        self.engine_core.exit()

    def add_request(self, prompt: str | list[int],
                    sampling_params: SamplingParams, priority: int = 0) -> str:
        """登记一条请求，返回其 request_id。"""
        req = self.processor.process_inputs(prompt, sampling_params, priority=priority)
        self.engine_core.add_request(req)
        self.output_processor.add_request(req)
        return req.request_id

    def step(self) -> tuple[list[RequestOutput], int]:
        """推进一步：取核心产出 → OutputProcessor → 处理输出侧 abort。

        get_output：InprocClient 即同步 step()；MPClient 阻塞读子进程 busy-loop 的下一批产出。"""
        core_outputs = self.engine_core.get_output()
        processed = self.output_processor.process_outputs(core_outputs.outputs)
        if processed.reqs_to_abort:
            self.engine_core.abort_requests(processed.reqs_to_abort)
        return processed.request_outputs, core_outputs.num_tokens

    def is_finished(self) -> bool:
        return not self.engine_core.has_unfinished_requests()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[dict]:
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)

        request_ids = [self.add_request(p, sp)
                       for p, sp in zip(prompts, sampling_params)]

        pbar = tqdm(total=len(prompts), desc="Generating",
                    dynamic_ncols=True, disable=not use_tqdm)
        results: dict[str, RequestOutput] = {}
        prefill_throughput = decode_throughput = 0.0

        while not self.is_finished():
            t = perf_counter()
            request_outputs, num_tokens = self.step()
            if num_tokens > 0:
                prefill_throughput = num_tokens / (perf_counter() - t)
            elif num_tokens < 0:
                decode_throughput = -num_tokens / (perf_counter() - t)
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            for ro in request_outputs:
                if ro.finished:
                    results[ro.request_id] = ro
                    pbar.update(1)

        pbar.close()
        ordered = [results[rid] for rid in request_ids]
        return [{"text": ro.text, "token_ids": ro.token_ids} for ro in ordered]
