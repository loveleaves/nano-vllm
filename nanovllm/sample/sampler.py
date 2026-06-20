"""
采样器（对齐 vLLM V1 `v1/sample/sampler.py::Sampler`）。

按 SamplingMetadata 对一批 logits 依次：可选 logprobs 留存 → float32 → 惩罚 → 采样
（greedy / 温度 + top-k/top-p 随机，逐行按温度决定）→ 收集 logprobs → SamplerOutput。

nano 子集：不含 bad_words / allowed_token_ids / min_p / logits_processors / spec。
"""
import torch
from torch import nn

from nanovllm.sample.metadata import SamplingMetadata
from nanovllm.sample.outputs import SamplerOutput
from nanovllm.sample.ops.bad_words import apply_bad_words
from nanovllm.sample.ops.logprobs import compute_logprobs, gather_logprobs
from nanovllm.sample.ops.penalties import apply_all_penalties
from nanovllm.sample.ops.topk_topp import TopKTopPSampler, apply_min_p

_SAMPLING_EPS = 1e-5


class Sampler(nn.Module):

    def __init__(self):
        super().__init__()
        self.topk_topp_sampler = TopKTopPSampler()

    def forward(self, logits: torch.Tensor,
                sampling_metadata: SamplingMetadata) -> SamplerOutput:
        num_logprobs = sampling_metadata.max_num_logprobs
        # 原始（惩罚/温度前）logits 的 logprobs，对齐 V1（与 V0 不同）
        raw_logprobs = None
        if num_logprobs is not None:
            raw_logprobs = compute_logprobs(logits)

        logits = logits.float()
        if not sampling_metadata.no_penalties:
            logits = apply_all_penalties(
                logits,
                sampling_metadata.prompt_token_ids,
                sampling_metadata.output_token_ids,
                sampling_metadata.presence_penalties,
                sampling_metadata.frequency_penalties,
                sampling_metadata.repetition_penalties,
            )
        if sampling_metadata.bad_words_token_ids:
            logits = apply_bad_words(
                logits, sampling_metadata.bad_words_token_ids,
                sampling_metadata.output_token_ids)

        sampled = self.sample(logits, sampling_metadata).long()

        logprobs_tensors = None
        if num_logprobs is not None:
            logprobs_tensors = gather_logprobs(raw_logprobs, num_logprobs, sampled)
        return SamplerOutput(sampled_token_ids=sampled,
                             logprobs_tensors=logprobs_tensors)

    def sample(self, logits: torch.Tensor,
               sampling_metadata: SamplingMetadata) -> torch.Tensor:
        """按行选择 greedy / 随机：temperature < eps 的行取 argmax，其余走采样。"""
        assert not (sampling_metadata.all_greedy and sampling_metadata.all_random)

        if sampling_metadata.all_random:
            greedy_sampled = None
        else:
            greedy_sampled = logits.argmax(dim=-1).view(-1)
            if sampling_metadata.all_greedy:
                return greedy_sampled

        # 温度缩放（greedy 行用 1.0 占位避免除零）
        temp = sampling_metadata.temperature
        if not sampling_metadata.all_random:
            temp = torch.where(temp < _SAMPLING_EPS, 1.0, temp)
        logits = logits.div_(temp.unsqueeze(dim=1))

        # min-p 过滤（argmax 不变，置于温度后、top-k/p 前）
        if sampling_metadata.min_p is not None:
            logits = apply_min_p(logits, sampling_metadata.min_p)

        random_sampled = self.topk_topp_sampler(
            logits, sampling_metadata.top_k, sampling_metadata.top_p,
            sampling_metadata.generators)

        if greedy_sampled is None:
            return random_sampled
        return torch.where(
            sampling_metadata.temperature < _SAMPLING_EPS,
            greedy_sampled, random_sampled)
