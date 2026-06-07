import torch
from torch import nn


class Sampler(nn.Module):
    """
    Token 采样器：Gumbel-max trick（完全向量化，等价于 categorical 采样）。

    步骤：
      1. logits / temperature：温度缩放
      2. softmax：转换为概率分布
      3. probs / Exponential(1)：等价于 Gumbel 噪声的 argmax
      4. argmax：取最大值对应的 token

    @torch.compile：将 div + softmax + exponential + argmax 融合，减少 HBM 读写。
    clamp_min_(1e-10) 防止 Exponential 极小值导致 inf。
    """

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor) -> torch.Tensor:
        logits = logits.float().div_(temperatures.unsqueeze(1))
        probs = torch.softmax(logits, dim=-1)
        return probs.div_(
            torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)
        ).argmax(dim=-1)
