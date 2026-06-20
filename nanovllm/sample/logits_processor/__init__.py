"""
Logits Processor 框架（对齐 vLLM V1 `v1/sample/logits_processor/`）。

把"采样前对 logits 的逐项修改"从 Sampler 里硬编码的调用，收敛为一组可插拔的
LogitsProcessor。Sampler 持一个有序列表，逐个 apply(logits, metadata)；每个处理器
从 SamplingMetadata 读自己的配置、对相关行生效、无配置时快速 no-op。

内置项：惩罚（包装 ops.penalties）、bad_words（包装 ops.bad_words）、logit_bias、
min_tokens（抑制 EOS 直到达到最小生成长度）。引导解码（guided/）作为同一框架下的
一个处理器接入。
"""
from nanovllm.sample.logits_processor.interface import (
    LogitsProcessor,
    build_logits_processors,
)
from nanovllm.sample.logits_processor.builtin import (
    BadWordsLogitsProcessor,
    LogitBiasLogitsProcessor,
    MinTokensLogitsProcessor,
    PenaltiesLogitsProcessor,
)

__all__ = [
    "LogitsProcessor",
    "build_logits_processors",
    "PenaltiesLogitsProcessor",
    "BadWordsLogitsProcessor",
    "LogitBiasLogitsProcessor",
    "MinTokensLogitsProcessor",
]
