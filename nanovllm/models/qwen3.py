import torch
from torch import nn

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import (
    QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear,
)
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead


class Qwen3Attention(nn.Module):
    """
    Qwen3 注意力模块，支持 GQA 和 QK-Norm。

    架构：
      - GQA：Q head 数 > KV head 数
      - QK-Norm：qkv_bias=False 时对 Q/K 各做一次 RMSNorm
      - RoPE：@lru_cache 单例，所有层共享
      - TP：QKV 列并行，O 行并行
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-6,
        qkv_bias: bool = False,
        rope_theta: float = 10000,
        rope_scaling: dict | None = None,
    ):
        super().__init__()
        self.total_num_heads = num_heads
        self.total_num_kv_heads = num_kv_heads
        self.head_dim = head_dim or hidden_size // num_heads
        self.qkv_bias = qkv_bias

        self.qkv_proj = QKVParallelLinear(
            hidden_size, self.head_dim, num_heads, num_kv_heads, bias=qkv_bias,
        )
        # 运行时 num_heads/num_kv_heads 会被 TP 切分
        from nanovllm.layers.linear import _get_tp_info, divide
        _, tp_size = _get_tp_info()
        self.num_heads = divide(num_heads, tp_size)
        self.num_kv_heads = divide(num_kv_heads, tp_size)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5

        self.o_proj = RowParallelLinear(num_heads * self.head_dim, hidden_size, bias=False)

        if isinstance(rope_scaling, dict):
            rope_theta = rope_scaling.get("rope_theta", rope_theta)
        self.rotary_emb = get_rope(
            self.head_dim, rotary_dim=self.head_dim,
            max_position=max_position, base=rope_theta,
        )
        self.attn = Attention(self.num_heads, self.head_dim, self.scaling, self.num_kv_heads)

        if not self.qkv_bias:
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor,
                attn_md) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        if not self.qkv_bias:
            q = self.q_norm(q)
            k = self.k_norm(k)
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v, attn_md)
        return self.o_proj(o.flatten(1, -1))


class Qwen3MLP(nn.Module):
    """Qwen3 FFN：SwiGLU (gate_up_proj → SiluAndMul → down_proj)。"""

    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(hidden_size, [intermediate_size] * 2, bias=False)
        self.down_proj = RowParallelLinear(intermediate_size, hidden_size, bias=False)
        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_up_proj(x)))


class Qwen3DecoderLayer(nn.Module):
    """
    Qwen3 Transformer 解码层（Pre-LN + Fused Add-RMSNorm）。

    残差流在层间传递：
      首层:  hidden, residual = norm(hidden), hidden
      后续:  hidden, residual = norm(hidden + residual)（fused）
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, "attention_bias", False),
            head_dim=getattr(config, "head_dim", None),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        self.mlp = Qwen3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        attn_md,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states, attn_md)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen3Model(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor,
                attn_md) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual, attn_md)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3ForCausalLM(nn.Module):
    """
    Qwen3 因果语言模型（完整版，含 TP + CUDA graph 兼容接口）。

    packed_modules_mapping：HF 权重名 → nano-vllm 参数名的映射。
    forward 与 compute_logits 分离：CUDA graph 只录 forward（不含 lm_head）。
    """

    packed_modules_mapping = {
        "q_proj":    ("qkv_proj", "q"),
        "k_proj":    ("qkv_proj", "k"),
        "v_proj":    ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj":   ("gate_up_proj", 1),
    }

    def __init__(self, config) -> None:
        super().__init__()
        self.model = Qwen3Model(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if getattr(config, "tie_word_embeddings", False):
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor,
                attn_md) -> torch.Tensor:
        return self.model(input_ids, positions, attn_md)

    def compute_logits(self, hidden_states: torch.Tensor,
                       attn_md=None) -> torch.Tensor:
        return self.lm_head(hidden_states, attn_md)
