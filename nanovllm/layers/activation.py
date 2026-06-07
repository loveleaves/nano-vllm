import torch
from torch import nn
import torch.nn.functional as F


class SiluAndMul(nn.Module):
    """
    SwiGLU 激活函数：将输入沿最后维度对半分，gate 分支过 SiLU，然后与 up 分支相乘。

    对应 Qwen3 MLP：
      gate_up = gate_up_proj(x)       # [N, 2 * intermediate_size]
      x, y = gate_up.chunk(2, -1)     # x=gate, y=up
      output = silu(x) * y            # SwiGLU

    数学公式：SwiGLU(x, y) = SiLU(x) * y = (x * sigmoid(x)) * y
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, y = x.chunk(2, dim=-1)
        return F.silu(x) * y
