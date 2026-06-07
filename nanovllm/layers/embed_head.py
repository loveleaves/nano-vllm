import torch
from torch import nn
import torch.nn.functional as F

from nanovllm.utils.context import get_context


class VocabEmbedding(nn.Module):
    """
    词表 Embedding（Phase 2 非并行版本）。
    标准 nn.Embedding，weight_loader 注册供 loader.py 使用。
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.embedding(x, self.weight)


class LMHead(nn.Module):
    """
    LM Head（Phase 2 非并行版本）。

    prefill 优化：只对每个序列的最后一个 token 计算 logits。
    逻辑：cu_seqlens_q[1:] - 1 取每个序列的最后位置索引。
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        context = get_context()
        if context.is_prefill and context.cu_seqlens_q is not None:
            last_indices = context.cu_seqlens_q[1:] - 1
            x = x[last_indices].contiguous()
        return F.linear(x, self.weight)
