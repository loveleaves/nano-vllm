from functools import lru_cache
import torch
from torch import nn


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """
    应用旋转位置编码（RoPE）。

    实现（2D 旋转）：
      将 x 拆为前后各 head_dim/2 维：x1, x2
      y1 = x1 * cos - x2 * sin
      y2 = x2 * cos + x1 * sin

    在 float32 下计算，避免 bf16/fp16 精度损失。
    """
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):
    """
    旋转位置编码（RoPE）层，预计算并缓存所有位置的 cos/sin 值。

    预计算（__init__）：
      inv_freq = 1 / (base^(2i/d))
      freqs[pos, i] = pos * inv_freq[i]
      cos_sin_cache[pos] = cat(freqs.cos(), freqs.sin()) → [max_position, 1, head_dim]

    @lru_cache（get_rope）：所有 Attention 层共享同一实例，避免重复分配 cos_sin_cache。
    """

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
    ):
        super().__init__()
        self.head_size = head_size
        self.rotary_dim = rotary_dim

        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    def forward(
        self,
        positions: torch.Tensor,   # [N]，每个 token 的绝对位置
        query: torch.Tensor,       # [N, num_heads, head_dim]
        key: torch.Tensor,         # [N, num_kv_heads, head_dim]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos_sin = self.cos_sin_cache[positions]          # [N, 1, rotary_dim]
        cos, sin = cos_sin.chunk(2, dim=-1)              # 各 [N, 1, rotary_dim/2]
        if self.rotary_dim < self.head_size:
            q_rot, q_pass = query[..., :self.rotary_dim], query[..., self.rotary_dim:]
            k_rot, k_pass = key[..., :self.rotary_dim], key[..., self.rotary_dim:]
            query = torch.cat([apply_rotary_emb(q_rot, cos, sin), q_pass], dim=-1)
            key   = torch.cat([apply_rotary_emb(k_rot, cos, sin), k_pass], dim=-1)
        else:
            query = apply_rotary_emb(query, cos, sin)
            key   = apply_rotary_emb(key, cos, sin)
        return query, key


@lru_cache(maxsize=16)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
) -> RotaryEmbedding:
    """单例工厂：相同参数只创建一个 RotaryEmbedding 实例，所有层共享。"""
    return RotaryEmbedding(head_size, rotary_dim, max_position, base)
