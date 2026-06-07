import torch
from torch import nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization（RMSNorm）。

    两种 forward 路径：
    1. rms_forward(x): 标准 RMSNorm，x = x / rms * weight
    2. add_rms_forward(x, residual): Fused Add-RMSNorm：
         residual = x + residual    → 更新残差流
         x = rms_norm(residual)     → 归一化
       避免两次单独内存读写，节省约 50% HBM 带宽。

    @torch.compile：将多个 elementwise 算子融合为单个 kernel。
    均在 float32 下计算以保证精度，结果转回原 dtype。
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    @torch.compile
    def rms_forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return (x * self.weight.float()).to(orig_dtype)

    @torch.compile
    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        orig_dtype = x.dtype
        x = x.float() + residual.float()
        residual = x.to(orig_dtype)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return (x * self.weight.float()).to(orig_dtype), residual

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ):
        if residual is None:
            return self.rms_forward(x)
        return self.add_rms_forward(x, residual)
