"""内置 LogitsProcessor：惩罚 / bad_words / logit_bias / min_tokens。"""
import torch

from nanovllm.sample.logits_processor.interface import LogitsProcessor
from nanovllm.sample.ops.bad_words import apply_bad_words
from nanovllm.sample.ops.penalties import apply_all_penalties


class PenaltiesLogitsProcessor(LogitsProcessor):
    """presence / frequency / repetition 惩罚（包装 ops.penalties）。"""

    def apply(self, logits, metadata):
        if metadata.no_penalties:
            return logits
        return apply_all_penalties(
            logits,
            metadata.prompt_token_ids,
            metadata.output_token_ids,
            metadata.presence_penalties,
            metadata.frequency_penalties,
            metadata.repetition_penalties,
        )


class BadWordsLogitsProcessor(LogitsProcessor):
    """禁止词：某禁止序列前缀匹配输出尾部时屏蔽其末 token（包装 ops.bad_words）。"""

    def apply(self, logits, metadata):
        if not metadata.bad_words_token_ids:
            return logits
        return apply_bad_words(
            logits, metadata.bad_words_token_ids, metadata.output_token_ids)


class LogitBiasLogitsProcessor(LogitsProcessor):
    """OpenAI logit_bias：行 → {token_id: bias}，在对应 logit 上加偏置。"""

    def apply(self, logits, metadata):
        logit_bias = metadata.logit_bias
        if not logit_bias:
            return logits
        for row, bias_map in logit_bias.items():
            for token_id, bias in bias_map.items():
                logits[row, token_id] += bias
        return logits


class MinTokensLogitsProcessor(LogitsProcessor):
    """min_tokens：已生成数 < 阈值的行屏蔽 EOS，强制继续生成。"""

    def apply(self, logits, metadata):
        min_tokens = metadata.min_tokens
        if not min_tokens or metadata.eos_token_id is None:
            return logits
        out = metadata.output_token_ids
        for row, threshold in min_tokens.items():
            produced = len(out[row]) if out is not None else 0
            if produced < threshold:
                logits[row, metadata.eos_token_id] = -float("inf")
        return logits
