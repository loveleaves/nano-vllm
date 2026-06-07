import torch
from torch import nn
import torch.nn.functional as F


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    """默认权重加载：直接 copy。"""
    param.data.copy_(loaded_weight)


class LinearBase(nn.Module):
    """
    线性层基类，提供统一的权重初始化和 weight_loader 注册接口。

    weight_loader 机制：
      每个参数上注册 weight_loader 函数属性。
      loader.py 调用 param.weight_loader(param, loaded_weight, [shard_id])，
      实现参数自己决定如何接受 HF 权重（支持 TP 切分）。

    Phase 2（非并行版本）：tp_size=1，不做任何切分。
    """

    def __init__(self, input_size: int, output_size: int, bias: bool = False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, *args):
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ReplicatedLinear(LinearBase):
    """非 TP 线性层：标准 F.linear。"""

    def __init__(self, input_size: int, output_size: int, bias: bool = False):
        super().__init__(input_size, output_size, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class ColumnParallelLinear(LinearBase):
    """
    列并行线性层（Phase 2 单 GPU 版：无实际切分，等价于 ReplicatedLinear）。
    接口与 TP 版本兼容，供模型组装使用。
    """

    def __init__(self, input_size: int, output_size: int, bias: bool = False):
        super().__init__(input_size, output_size, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class MergedColumnParallelLinear(ColumnParallelLinear):
    """
    合并列并行线性层（gate + up → gate_up_proj）。

    weight_loader 接受 shard_id（int），按各子矩阵的偏移写入合并参数。
    """

    def __init__(self, input_size: int, output_sizes: list[int], bias: bool = False):
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, shard_id: int):
        offset = sum(self.output_sizes[:shard_id])
        size = self.output_sizes[shard_id]
        param.data[offset: offset + size].copy_(loaded_weight)


class QKVParallelLinear(ColumnParallelLinear):
    """
    QKV 合并线性层（Q + K + V → qkv_proj）。

    weight_loader 接受 shard_id（"q"/"k"/"v"），按 Q/K/V 在合并参数中的偏移写入。
    支持 GQA（Q head 数 > KV head 数）。
    """

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
    ):
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        self.head_size = head_size
        self.num_heads = total_num_heads
        self.num_kv_heads = total_num_kv_heads
        output_size = (total_num_heads + 2 * total_num_kv_heads) * head_size
        super().__init__(hidden_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, shard_id: str):
        q_size = self.num_heads * self.head_size
        kv_size = self.num_kv_heads * self.head_size
        if shard_id == "q":
            param.data[0: q_size].copy_(loaded_weight)
        elif shard_id == "k":
            param.data[q_size: q_size + kv_size].copy_(loaded_weight)
        elif shard_id == "v":
            param.data[q_size + kv_size: q_size + 2 * kv_size].copy_(loaded_weight)
        else:
            raise ValueError(f"unknown shard_id: {shard_id}")


class RowParallelLinear(LinearBase):
    """
    行并行线性层（Phase 2 单 GPU 版：等价于普通线性层）。
    接口与 TP 版本兼容。
    """

    def __init__(self, input_size: int, output_size: int, bias: bool = False):
        super().__init__(input_size, output_size, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)
