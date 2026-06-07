import torch
from torch import nn


class Sampler(nn.Module):
    """
    Token 采样器。

    Phase 2 实现：使用 Gumbel-max trick（完全向量化，等价于 categorical 采样）。

    步骤：
      1. logits / temperature：温度缩放
      2. softmax：转换为概率分布
      3. probs / Exponential(1)：等价于加 Gumbel 噪声的 argmax
      4. argmax：取最大值对应的 token

    等价于：argmax(log(probs) + Gumbel(0,1))
    """

    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor) -> torch.Tensor:
        """
        输入：
          logits:       [batch, vocab_size]
          temperatures: [batch]
        返回：
          sample_tokens: [batch]（int64）
        """
        logits = logits.float().div_(temperatures.unsqueeze(1))
        probs = torch.softmax(logits, dim=-1)
        noise = torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)
        return probs.div_(noise).argmax(dim=-1)
