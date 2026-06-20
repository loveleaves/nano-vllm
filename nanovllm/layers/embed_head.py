import torch
from torch import nn
import torch.nn.functional as F

try:
    import torch.distributed as dist
    _DIST_AVAILABLE = True
except ImportError:
    _DIST_AVAILABLE = False

from nanovllm.utils.context import AttentionMetadata
from nanovllm.layers.linear import _get_tp_info, divide


class VocabParallelEmbedding(nn.Module):
    """
    词表并行 Embedding 层：将 vocab 均分到各 GPU。

    rank i → vocab[vocab/N * i : vocab/N * (i+1)]
    forward：mask 超出范围的 token，embedding 后 all_reduce 汇总。
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.tp_rank, self.tp_size = _get_tp_info()
        assert num_embeddings % self.tp_size == 0
        self.num_embeddings = num_embeddings
        self.num_embeddings_per_partition = num_embeddings // self.tp_size
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        shard_size = param.data.size(0)
        start_idx = self.tp_rank * shard_size
        param.data.copy_(loaded_weight.narrow(0, start_idx, shard_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.tp_size > 1:
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            x_local = mask * (x - self.vocab_start_idx)
        else:
            x_local = x
        y = F.embedding(x_local, self.weight)
        if self.tp_size > 1:
            y = mask.unsqueeze(1) * y
            if _DIST_AVAILABLE and dist.is_initialized():
                dist.all_reduce(y)
        return y


class ParallelLMHead(VocabParallelEmbedding):
    """
    并行 LM Head：与 VocabParallelEmbedding 共享权重结构（转置矩阵乘）。

    统一连续批：每个 seq 只取其 query 段最后一个 token 计算 logits。
    选取规则 cu_seqlens_q[1:]-1 对 prefill / decode / 混合批一致——decode 段
    长度为 1 时退化为恒等选取。
    TP logits 汇聚：rank 0 用 dist.gather 收集所有 rank 的 partial logits。
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, bias: bool = False):
        assert not bias
        super().__init__(num_embeddings, embedding_dim)

    def forward(self, x: torch.Tensor, attn_md: AttentionMetadata | None = None) -> torch.Tensor:
        if attn_md is not None and attn_md.query_start_loc is not None:
            last_indices = attn_md.query_start_loc[1:] - 1
            x = x[last_indices].contiguous()
        logits = F.linear(x, self.weight)
        if self.tp_size > 1 and _DIST_AVAILABLE and dist.is_initialized():
            all_logits = (
                [torch.empty_like(logits) for _ in range(self.tp_size)]
                if self.tp_rank == 0 else None
            )
            dist.gather(logits, all_logits, 0)
            logits = torch.cat(all_logits, dim=-1) if self.tp_rank == 0 else None
        return logits


# 为了向后兼容 Phase 2/3 的 VocabEmbedding / LMHead 接口
VocabEmbedding = VocabParallelEmbedding
LMHead = ParallelLMHead
