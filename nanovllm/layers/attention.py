import torch
from torch import nn
import torch.nn.functional as F

from nanovllm.utils.context import get_context


class Attention(nn.Module):
    """
    Naive 注意力层（Phase 2：不接 KV cache，不用 FlashAttention）。

    使用 torch.scaled_dot_product_attention 计算注意力，每次重新计算（无 KV cache 读写）。
    Phase 3 将替换为 PagedAttention + FlashAttention。

    prefill: causal mask，q/k/v 形状 [N, num_heads, head_dim]
    decode:  每个 seq 只有 1 个 query token，但 k/v 来自完整历史（Phase 2 无 cache，暂不支持 decode）
    """

    def __init__(self, num_heads: int, head_dim: int, scale: float, num_kv_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        # GQA 扩展比（Phase 2 直接 repeat_interleave 实现 GQA）
        self.num_kv_groups = num_heads // num_kv_heads

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        输入：
          q: [N, num_heads, head_dim]
          k: [N, num_kv_heads, head_dim]
          v: [N, num_kv_heads, head_dim]
        返回：
          o: [N, num_heads, head_dim]
        """
        context = get_context()

        if context.is_prefill:
            # prefill：单序列或 batch 统一处理
            # 为了使用 SDPA，需要 [batch, heads, seq, head_dim] 格式
            # Phase 2 简化为每次处理所有 token（不分 seq，不用 cu_seqlens）
            N = q.size(0)
            # GQA：重复 KV heads 以匹配 Q heads 数量
            if self.num_kv_groups > 1:
                k = k.repeat_interleave(self.num_kv_groups, dim=1)
                v = v.repeat_interleave(self.num_kv_groups, dim=1)
            # [1, num_heads, N, head_dim]
            q = q.transpose(0, 1).unsqueeze(0)
            k = k.transpose(0, 1).unsqueeze(0)
            v = v.transpose(0, 1).unsqueeze(0)
            o = F.scaled_dot_product_attention(q, k, v, scale=self.scale, is_causal=True)
            # [N, num_heads, head_dim]
            return o.squeeze(0).transpose(0, 1)
        else:
            # decode（Phase 2 不支持 KV cache，仅占位）
            raise NotImplementedError("Decode with KV cache requires Phase 3+")
