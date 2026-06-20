"""向后兼容垫片：Sampler 已升级为结构化采样层 `nanovllm/layers/sample/`。

旧的 `forward(logits, temperatures)` 签名被 `forward(logits, SamplingMetadata)` 取代
（对齐 V1 v1/sample/sampler.py）。新代码请直接从 `nanovllm.layers.sample` 导入。
"""
from nanovllm.layers.sample import (  # noqa: F401
    LogprobsTensors,
    Sampler,
    SamplerOutput,
    SamplingMetadata,
)
