"""
采样输出契约（对齐 vLLM V1 `v1/outputs.py::SamplerOutput` / `LogprobsTensors`）。
"""
from dataclasses import dataclass

import torch


@dataclass
class LogprobsTensors:
    """采样 logprobs 的三元张量（GPU）。

    logprob_token_ids — [n, 1+k] 每行：采样 token + top-k token 的 id
    logprobs          — [n, 1+k] 对应 logprob
    selected_token_ranks — [n] 采样 token 在词表 logprob 降序中的排名（1-based）
    """
    logprob_token_ids: torch.Tensor
    logprobs: torch.Tensor
    selected_token_ranks: torch.Tensor


@dataclass
class SamplerOutput:
    """Sampler 一次前向的结果。

    sampled_token_ids — [n] 采样得到的 token id（int64）
    logprobs_tensors  — 若请求 logprobs 则非 None
    """
    sampled_token_ids: torch.Tensor
    logprobs_tensors: LogprobsTensors | None = None
