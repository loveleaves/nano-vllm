"""
重复 / 频率 / 存在惩罚（对齐 vLLM V1 `v1/sample/ops/penalties.py`
+ `model_executor/layers/utils.py::apply_penalties`）。

nano 用纯 torch 实现 repetition penalty（V1 走 _custom_ops 的 CUDA kernel），语义一致：
  - 对历史出现过（prompt ∪ output）的 token：logit>0 则 /=rep，否则 *=rep；
  - 频率惩罚：logit -= freq * 出现次数（output）；
  - 存在惩罚：logit -= pres * 出现掩码（output）。
"""
import numpy as np
import torch


def _to_padded(token_lists: list[list[int]], vocab_size: int,
               device: torch.device) -> torch.Tensor:
    """把变长 token 列表 pad 成 [n, maxlen]，pad 值用 vocab_size（不对应任何有效 token）。"""
    maxlen = max((len(t) for t in token_lists), default=0)
    maxlen = max(maxlen, 1)
    arr = np.full((len(token_lists), maxlen), vocab_size, dtype=np.int64)
    for i, toks in enumerate(token_lists):
        if toks:
            arr[i, :len(toks)] = toks
    return torch.from_numpy(arr).to(device, non_blocking=True)


def get_token_bin_counts_and_mask(tokens: torch.Tensor, vocab_size: int,
                                  num_seqs: int) -> tuple[torch.Tensor, torch.Tensor]:
    """统计每行各 token 的出现次数与出现掩码（pad 列 vocab_size 被切掉）。"""
    bin_counts = torch.zeros((num_seqs, vocab_size + 1),
                             dtype=torch.long, device=tokens.device)
    bin_counts.scatter_add_(1, tokens, torch.ones_like(tokens))
    bin_counts = bin_counts[:, :vocab_size]
    mask = bin_counts > 0
    return bin_counts, mask


def apply_repetition_penalties_torch(logits: torch.Tensor, prompt_mask: torch.Tensor,
                                     output_mask: torch.Tensor,
                                     repetition_penalties: torch.Tensor) -> torch.Tensor:
    """对 prompt∪output 中出现过的 token 施加重复惩罚（in-place）。"""
    repeated = prompt_mask | output_mask
    rep = repetition_penalties.unsqueeze(dim=1)            # [n, 1]
    penalized = torch.where(logits > 0, logits / rep, logits * rep)
    return torch.where(repeated, penalized, logits)


def apply_all_penalties(logits: torch.Tensor,
                        prompt_token_ids: list[list[int]],
                        output_token_ids: list[list[int]],
                        presence_penalties: torch.Tensor,
                        frequency_penalties: torch.Tensor,
                        repetition_penalties: torch.Tensor) -> torch.Tensor:
    """对 logits 施加存在 / 频率 / 重复三类惩罚（对齐 OpenAI 定义）。"""
    num_seqs, vocab_size = logits.shape
    device = logits.device
    prompt_t = _to_padded(prompt_token_ids, vocab_size, device)
    output_t = _to_padded(output_token_ids, vocab_size, device)

    _, prompt_mask = get_token_bin_counts_and_mask(prompt_t, vocab_size, num_seqs)
    output_bin_counts, output_mask = get_token_bin_counts_and_mask(
        output_t, vocab_size, num_seqs)

    logits = apply_repetition_penalties_torch(
        logits, prompt_mask, output_mask, repetition_penalties)
    logits -= frequency_penalties.unsqueeze(dim=1) * output_bin_counts
    logits -= presence_penalties.unsqueeze(dim=1) * output_mask
    return logits
