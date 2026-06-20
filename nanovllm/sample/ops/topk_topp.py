"""
top-k / top-p 过滤 + 加权随机采样（对齐 vLLM V1 `v1/sample/ops/topk_topp_sampler.py`
的 forward_native 路径）。

nano 用 Gumbel-max（`probs / Exponential(1)` 后 argmax）做无 CPU-GPU 同步的批量采样，
与 V1 `random_sample` 一致（不引入 flashinfer / 逐请求 generator）。
"""
import torch
from torch import nn


def apply_top_k_only(logits: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """仅 top-k：把每行第 k 大以下的 logit 置 -inf（不排序整词表）。"""
    no_top_k_mask = k == logits.shape[1]
    k = k.masked_fill(no_top_k_mask, 1)
    max_top_k = int(k.max())
    k_index = k.sub(1).unsqueeze(1)                        # 0-based 第 k 大
    top_k_mask = logits.topk(max_top_k, dim=1).values.gather(1, k_index.long())
    top_k_mask.masked_fill_(no_top_k_mask.unsqueeze(1), -float("inf"))
    logits.masked_fill_(logits < top_k_mask, -float("inf"))
    return logits


def apply_top_k_top_p(logits: torch.Tensor, k: torch.Tensor | None,
                      p: torch.Tensor | None) -> torch.Tensor:
    """对 logits 施加 top-k 与/或 top-p 掩码（可 in-place）。"""
    if p is None:
        if k is None:
            return logits
        return apply_top_k_only(logits, k)

    logits_sort, logits_idx = logits.sort(dim=-1, descending=False)
    if k is not None:
        top_k_mask = logits_sort.size(1) - k.to(torch.long)   # 升序下第 k 大的位置
        top_k_mask = logits_sort.gather(1, top_k_mask.unsqueeze(dim=1))
        logits_sort.masked_fill_(logits_sort < top_k_mask, -float("inf"))
    if p is not None:
        probs_sort = logits_sort.softmax(dim=-1)
        probs_sum = torch.cumsum(probs_sort, dim=-1, out=probs_sort)
        top_p_mask = probs_sum <= 1 - p.unsqueeze(dim=1)
        top_p_mask[:, -1] = False                              # 至少保留一个
        logits_sort.masked_fill_(top_p_mask, -float("inf"))
    return logits_sort.scatter(dim=-1, index=logits_idx, src=logits_sort)


def apply_min_p(logits: torch.Tensor, min_p: torch.Tensor) -> torch.Tensor:
    """min-p 过滤：屏蔽 prob < min_p * max_prob 的 token（argmax 不变）。"""
    probs = logits.softmax(dim=-1)
    top_probs = probs.max(dim=-1, keepdim=True).values
    threshold = min_p.unsqueeze(1) * top_probs
    return logits.masked_fill(probs < threshold, -float("inf"))


def random_sample(probs: torch.Tensor,
                  generators: dict[int, torch.Generator] | None = None) -> torch.Tensor:
    """Gumbel-max：probs / Exponential(1) 后取 argmax（向量化，无 CPU-GPU 同步）。

    generators 非空时，对应行用各自的 torch.Generator 采样（可复现）；其余行用默认 RNG。
    """
    q = torch.empty_like(probs)
    if not generators or len(generators) != probs.shape[0]:
        q.exponential_(1)                     # 无种子的行先批量填充
    if generators:
        for i, g in generators.items():
            q[i].exponential_(1, generator=g)  # 有种子的行覆盖为可复现采样
    q.clamp_min_(1e-10)
    return probs.div_(q).argmax(dim=-1).view(-1)


class TopKTopPSampler(nn.Module):
    """top-k/top-p 过滤后做加权随机采样，返回采样 token id。"""

    def forward(self, logits: torch.Tensor, k: torch.Tensor | None,
                p: torch.Tensor | None,
                generators: dict[int, torch.Generator] | None = None) -> torch.Tensor:
        logits = apply_top_k_top_p(logits, k, p)
        probs = logits.softmax(dim=-1)
        return random_sample(probs, generators)
