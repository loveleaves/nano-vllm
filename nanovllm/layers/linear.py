import torch
from torch import nn
import torch.nn.functional as F

try:
    import torch.distributed as dist
    _DIST_AVAILABLE = True
except ImportError:
    _DIST_AVAILABLE = False


def _get_tp_info():
    """获取张量并行 rank 和 world_size（进程组未初始化时返回 (0, 1)）。"""
    if _DIST_AVAILABLE and dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def divide(numerator: int, denominator: int) -> int:
    assert numerator % denominator == 0, f"{numerator} not divisible by {denominator}"
    return numerator // denominator


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    """默认权重加载：直接 copy。"""
    param.data.copy_(loaded_weight)


class LinearBase(nn.Module):
    """
    线性层基类，提供统一的权重初始化和 weight_loader 注册接口。

    每个参数上注册 weight_loader 函数属性，供 loader.py 调用。
    TP 参数的 weight_loader 负责自动切片；非 TP 参数用 default_weight_loader。
    """

    def __init__(self, input_size: int, output_size: int,
                 bias: bool = False, tp_dim: int | None = None):
        super().__init__()
        self.tp_dim = tp_dim
        self.tp_rank, self.tp_size = _get_tp_info()
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, *args):
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ReplicatedLinear(LinearBase):
    """非 TP 线性层：所有 GPU 保持完整权重副本。"""

    def __init__(self, input_size: int, output_size: int, bias: bool = False):
        super().__init__(input_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, *args):
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class ColumnParallelLinear(LinearBase):
    """
    列并行线性层：按输出维度切分权重。

    每个 GPU 持有 weight[output/tp_size * rank : output/tp_size * (rank+1), :]
    forward：各 GPU 独立计算部分输出（不需要通信）。
    weight_loader：从完整权重中取 rank 对应的切片。
    """

    def __init__(self, input_size: int, output_size: int, bias: bool = False):
        tp_rank, tp_size = _get_tp_info()
        super().__init__(input_size, divide(output_size, tp_size), bias, tp_dim=0)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, *args):
        shard_size = param.data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class MergedColumnParallelLinear(ColumnParallelLinear):
    """
    合并列并行线性层（gate_proj + up_proj → gate_up_proj）。

    weight_loader 接受 shard_id（int），按各子矩阵在合并参数中的偏移写入对应区域。
    """

    def __init__(self, input_size: int, output_sizes: list[int], bias: bool = False):
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, shard_id: int):
        shard_offset = sum(self.output_sizes[:shard_id]) // self.tp_size
        shard_size = self.output_sizes[shard_id] // self.tp_size
        param_slice = param.data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_shard = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_slice.copy_(loaded_shard)


class QKVParallelLinear(ColumnParallelLinear):
    """
    QKV 合并列并行线性层（Q + K + V → qkv_proj）。

    合并参数形状：[(num_heads + 2*num_kv_heads) * head_dim / tp_size, hidden]
    内存布局：[Q 分片 | K 分片 | V 分片]

    weight_loader 接受 shard_id（"q"/"k"/"v"），按 GQA head 分配方式写入正确位置。
    """

    def __init__(self, hidden_size: int, head_size: int,
                 total_num_heads: int, total_num_kv_heads: int | None = None,
                 bias: bool = False):
        _, tp_size = _get_tp_info()
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        self.head_size = head_size
        self.num_heads = divide(total_num_heads, tp_size)
        self.num_kv_heads = divide(total_num_kv_heads, tp_size)
        output_size = (total_num_heads + 2 * total_num_kv_heads) * head_size
        super().__init__(hidden_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, shard_id: str):
        q_size = self.num_heads * self.head_size
        kv_size = self.num_kv_heads * self.head_size
        if shard_id == "q":
            shard_offset, shard_size = 0, q_size
        elif shard_id == "k":
            shard_offset, shard_size = q_size, kv_size
        elif shard_id == "v":
            shard_offset, shard_size = q_size + kv_size, kv_size
        else:
            raise ValueError(f"Unknown shard_id: {shard_id}")
        param_slice = param.data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_shard = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_slice.copy_(loaded_shard)


class RowParallelLinear(LinearBase):
    """
    行并行线性层：按输入维度切分权重。

    每个 GPU 持有 weight[:, input/tp_size * rank : input/tp_size * (rank+1)]
    forward：各 GPU 计算部分 matmul → all_reduce 求和得完整输出。
    bias 只在 rank 0 加（bias 是全局值，不切分，其余 rank 加避免重复累加）。
    """

    def __init__(self, input_size: int, output_size: int, bias: bool = False):
        _, tp_size = _get_tp_info()
        super().__init__(divide(input_size, tp_size), output_size, bias, tp_dim=1)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, *args):
        if param.data.ndim == 1:
            # bias：不切分，所有 rank 保持完整副本
            param.data.copy_(loaded_weight)
            return
        shard_size = param.data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        if self.tp_size > 1 and _DIST_AVAILABLE and dist.is_initialized():
            dist.all_reduce(y)
        return y
