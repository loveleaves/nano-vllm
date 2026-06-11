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
            # 按序列边界逐条计算注意力，防止多序列 batch 时跨序列 attend 污染。
            cu_q = context.cu_seqlens_q
            if cu_q is None:
                # 单序列 fallback（无 context 信息时）
                if self.num_kv_groups > 1:
                    k = k.repeat_interleave(self.num_kv_groups, dim=1)
                    v = v.repeat_interleave(self.num_kv_groups, dim=1)
                q = q.transpose(0, 1).unsqueeze(0)
                k = k.transpose(0, 1).unsqueeze(0)
                v = v.transpose(0, 1).unsqueeze(0)
                o = F.scaled_dot_product_attention(q, k, v, scale=self.scale, is_causal=True)
                return o.squeeze(0).transpose(0, 1)
            out_parts = []
            for s in range(cu_q.shape[0] - 1):
                s0, s1 = cu_q[s].item(), cu_q[s + 1].item()
                q_s = q[s0:s1]
                k_s = k[s0:s1]
                v_s = v[s0:s1]
                if self.num_kv_groups > 1:
                    k_s = k_s.repeat_interleave(self.num_kv_groups, dim=1)
                    v_s = v_s.repeat_interleave(self.num_kv_groups, dim=1)
                q_t = q_s.transpose(0, 1).unsqueeze(0)
                k_t = k_s.transpose(0, 1).unsqueeze(0)
                v_t = v_s.transpose(0, 1).unsqueeze(0)
                o_s = F.scaled_dot_product_attention(q_t, k_t, v_t, scale=self.scale, is_causal=True)
                out_parts.append(o_s.squeeze(0).transpose(0, 1))
            return torch.cat(out_parts, dim=0)
        else:
            # decode（Phase 2 不支持 KV cache，仅占位）
            raise NotImplementedError("Decode with KV cache requires Phase 3+")
