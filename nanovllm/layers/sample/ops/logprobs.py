"""
logprobs 计算与收集（对齐 vLLM V1 `v1/sample/ops/logprobs.py` + Sampler.gather_logprobs）。
"""
import torch

from nanovllm.layers.sample.outputs import LogprobsTensors


def compute_logprobs(logits: torch.Tensor) -> torch.Tensor:
    """对（惩罚/温度前的原始）logits 做 log_softmax，得到全词表 logprob。"""
    return logits.log_softmax(dim=-1, dtype=torch.float32)


def gather_logprobs(logprobs: torch.Tensor, num_logprobs: int,
                    token_ids: torch.Tensor) -> LogprobsTensors:
    """收集 top-`num_logprobs` 与采样 token 的 logprob 及其排名。

    返回 [n, 1+num_logprobs]：第 0 列为采样 token，其后为 top-k。
    """
    topk_logprobs, topk_indices = torch.topk(logprobs, num_logprobs, dim=-1)
    token_logprobs = logprobs.gather(-1, token_ids.unsqueeze(-1))
    # 采样 token 在词表中的排名（严格大于它的 logprob 个数 + 1）
    ranks = (logprobs > token_logprobs).sum(dim=-1) + 1

    indices = torch.cat([token_ids.unsqueeze(-1), topk_indices], dim=-1)
    values = torch.cat([token_logprobs, topk_logprobs], dim=-1)
    return LogprobsTensors(indices.to(torch.int32), values, ranks.to(torch.int32))
