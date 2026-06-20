"""GuidedDecodingLogitsProcessor：按每行 Grammar 掩码非法 token。"""
import torch

from nanovllm.sample.logits_processor.interface import LogitsProcessor


class GuidedDecodingLogitsProcessor(LogitsProcessor):
    """对每个受约束行，把 grammar 当前不允许的 token 的 logit 置 -inf。

    metadata.grammars: dict[row, Grammar]；None/空 时整批 no-op。状态推进（accept）由
    Sampler 在采样后调用，不在此处。
    """

    def apply(self, logits: torch.Tensor, metadata) -> torch.Tensor:
        grammars = metadata.grammars
        if not grammars:
            return logits
        vocab_size = logits.size(-1)
        neg_inf = float("-inf")
        for row, grammar in grammars.items():
            allowed = grammar.allowed_token_ids()
            if allowed is None:
                continue
            # 构造允许掩码：仅 allowed 中的 token 保留，其余置 -inf
            mask = torch.ones(vocab_size, dtype=torch.bool, device=logits.device)
            idx = torch.tensor(sorted(allowed), dtype=torch.long, device=logits.device)
            mask[idx] = False
            logits[row].masked_fill_(mask, neg_inf)
        return logits
