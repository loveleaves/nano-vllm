"""
bad_words 屏蔽（对齐 vLLM V1 `v1/sample/ops/bad_words.py`）。

每个禁止 token 序列：若其"前缀"（除末 token 外）恰好等于该请求已生成 token 的尾部，
则把末 token 的 logit 置 -inf（禁止生成）。单 token 序列（前缀空）则始终屏蔽该 token。

依赖已生成 token 历史（output_token_ids）；进程隔离的 decode 路径仅传 last_token，
故隔离模式下多 token bad_words 不可用（见 docs/arch_worker_isolation/design.md 边界）。
"""
import torch


def _apply_one(logits: torch.Tensor, row: int, bad_words: list[list[int]],
               past_tokens: list[int]) -> None:
    for seq in bad_words:
        if not seq:
            continue
        last = seq[-1]
        prefix = seq[:-1]
        if not prefix:
            logits[row, last] = -float("inf")          # 单 token：始终屏蔽
        elif len(past_tokens) >= len(prefix) and past_tokens[-len(prefix):] == prefix:
            logits[row, last] = -float("inf")          # 前缀匹配已生成尾部 → 屏蔽末 token


def apply_bad_words(logits: torch.Tensor,
                    bad_words_token_ids: dict[int, list[list[int]]],
                    output_token_ids: list[list[int]]) -> torch.Tensor:
    """对各行施加 bad_words 屏蔽（in-place）。output_token_ids 按行对齐。"""
    for row, bad_words in bad_words_token_ids.items():
        _apply_one(logits, row, bad_words, output_token_ids[row])
    return logits
