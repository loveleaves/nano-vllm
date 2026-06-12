import torch
import torch.nn.functional as F
from torch import nn

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.linear import MergedColumnParallelLinear, RowParallelLinear
from nanovllm.layers.embed_head import VocabEmbedding, LMHead
from nanovllm.models.qwen35 import (
    Qwen35RMSNorm,
    GatedDeltaNet,
    Qwen35Attention,
)


# ── 专家权重存储 ───────────────────────────────────────────────────────────────

class ExpertWeights(nn.Module):
    """存放所有路由专家的打包权重。
    参数路径与 safetensors key 严格对齐：mlp.experts.gate_up_proj / down_proj。
    gate_up_proj: [num_experts, 2*moe_inter, hidden]  — gate+up 已合并
    down_proj:    [num_experts, hidden, moe_inter]
    """

    def __init__(self, num_experts: int, moe_inter: int, hidden: int):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.empty(num_experts, 2 * moe_inter, hidden))
        self.down_proj    = nn.Parameter(torch.empty(num_experts, hidden, moe_inter))


# ── 共享专家 MLP ───────────────────────────────────────────────────────────────

class SharedExpertMLP(nn.Module):
    """始终激活的共享专家，SwiGLU FFN。
    权重名 shared_expert.gate_proj / up_proj / down_proj，
    由 packed_modules_mapping 将前两者合并到 gate_up_proj。
    """

    def __init__(self, hidden: int, moe_inter: int):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden, [moe_inter, moe_inter], bias=False
        )
        self.down_proj = RowParallelLinear(moe_inter, hidden, bias=False)
        self.act_fn    = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_up_proj(x)))


# ── MoE FFN ────────────────────────────────────────────────────────────────────

class Qwen35MoEFFN(nn.Module):
    """
    Qwen3.5-35B-A3B MoE FFN：256 路由专家（top-8 激活）+ 1 共享专家。

    路由逻辑：
      1. router logits = gate(hidden)，softmax 归一化
      2. topk(8) 选专家，对所选分数再次归一化（renormalize）
      3. 按 expert 索引分组 dispatch，batch matmul，scatter_add 累加
      4. 共享专家始终计算，乘 sigmoid(shared_expert_gate) 后加入输出
    """

    def __init__(self, config):
        super().__init__()
        hidden    = config.hidden_size
        num_exp   = config.num_experts
        top_k     = config.num_experts_per_tok
        moe_inter = config.moe_intermediate_size
        shared_inter = config.shared_expert_intermediate_size

        self.num_experts = num_exp
        self.top_k       = top_k

        # 路由器：weight 路径 mlp.gate.weight
        self.gate = nn.Linear(hidden, num_exp, bias=False)

        # 路由专家权重（packed）：路径 mlp.experts.{gate_up_proj,down_proj}
        self.experts = ExpertWeights(num_exp, moe_inter, hidden)

        # 共享专家：路径 mlp.shared_expert.*
        self.shared_expert = SharedExpertMLP(hidden, shared_inter)

        # 共享专家标量门：路径 mlp.shared_expert_gate.weight
        self.shared_expert_gate = nn.Linear(hidden, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T, hidden = x.shape

        # ── 路由 ──────────────────────────────────────────────────────────────
        router_logits = self.gate(x)                              # [T, E]
        scores = F.softmax(router_logits.float(), dim=-1)
        top_scores, top_indices = torch.topk(scores, self.top_k, dim=-1)  # [T, k]
        top_scores = top_scores / top_scores.sum(dim=-1, keepdim=True)    # renormalize

        # ── Routed experts dispatch ───────────────────────────────────────────
        # 展平为 (token, expert) 对，按 expert 排序以减少不连续访问
        flat_exp   = top_indices.reshape(-1)          # [T*k]
        flat_score = top_scores.reshape(-1).to(x.dtype)  # [T*k]
        tok_idx    = torch.arange(T, device=x.device).repeat_interleave(self.top_k)  # [T*k]

        sort_order = flat_exp.argsort()
        flat_exp   = flat_exp[sort_order]
        flat_score = flat_score[sort_order]
        tok_idx    = tok_idx[sort_order]

        out = torch.zeros_like(x)
        gup = self.experts.gate_up_proj   # [E, 2*moe_inter, hidden]
        dwn = self.experts.down_proj      # [E, hidden, moe_inter]

        for e in range(self.num_experts):
            mask = (flat_exp == e)
            if not mask.any():
                continue
            ti  = tok_idx[mask]                            # token 索引 [Te]
            x_e = x[ti]                                    # [Te, hidden]
            # gate+up: F.linear(x_e, W) = x_e @ W.T
            gu  = F.linear(x_e, gup[e])                   # [Te, 2*moe_inter]
            g_, u_ = gu.chunk(2, dim=-1)                   # gate, up: [Te, moe_inter]
            act = F.silu(g_) * u_                          # SwiGLU
            o_e = F.linear(act, dwn[e])                    # [Te, hidden]
            weighted = flat_score[mask].unsqueeze(-1) * o_e
            out.scatter_add_(0, ti.unsqueeze(-1).expand_as(o_e), weighted)

        # ── Shared expert ─────────────────────────────────────────────────────
        gate_val   = torch.sigmoid(self.shared_expert_gate(x))   # [T, 1]
        shared_out = self.shared_expert(x)                        # [T, hidden]
        out = out + gate_val * shared_out

        return out


# ── Decoder Layer ──────────────────────────────────────────────────────────────

class Qwen35MoEDecoderLayer(nn.Module):
    """
    Qwen3.5-35B MoE 解码层。子模块命名与 vllm 保持一致：
      linear_attention 层 → self.linear_attn (GatedDeltaNet，复用 qwen35.py)
      full_attention 层   → self.self_attn   (Qwen35Attention，复用 qwen35.py)
    MoE FFN 存于 self.mlp。
    """

    def __init__(self, config, layer_type: str):
        super().__init__()
        self.layer_type = layer_type
        if layer_type == "linear_attention":
            self.linear_attn = GatedDeltaNet(config)
        else:
            self.self_attn = Qwen35Attention(config)
        self.mlp = Qwen35MoEFFN(config)
        self.input_layernorm          = Qwen35RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen35RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states = hidden_states + residual
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)

        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(hidden_states)
        else:
            hidden_states = self.self_attn(positions, hidden_states)

        hidden_states = hidden_states + residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


# ── 顶层模型 ───────────────────────────────────────────────────────────────────

class Qwen35MoEModel(nn.Module):
    """Qwen3.5-35B MoE Transformer 骨干（不含 LM Head）。"""

    def __init__(self, config):
        super().__init__()
        self.embed_tokens = VocabEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            Qwen35MoEDecoderLayer(config, lt) for lt in config.layer_types
        ])
        self.norm = Qwen35RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states = self.norm(hidden_states + residual)
        return hidden_states


class Qwen35MoEForCausalLM(nn.Module):
    """
    Qwen3.5-35B-A3B 因果语言模型。

    权重加载路径映射：
      文件 key:  model.language_model.layers.0.linear_attn.in_proj_qkv.weight
        → 剥离 "model." → language_model.layers.0.linear_attn.in_proj_qkv.weight
        → 匹配   self.language_model.layers[0].linear_attn.in_proj_qkv.weight ✓

      shared_expert.gate_proj / up_proj → packed_modules_mapping → gate_up_proj ✓
      mlp.experts.gate_up_proj（已打包）→ 直接 copy ✓
    """

    weight_prefix_to_strip = "model."
    weight_skip_prefixes   = ("model.visual.", "mtp.")

    packed_modules_mapping = {
        "gate_proj": ("gate_up_proj", 0),
        "up_proj":   ("gate_up_proj", 1),
    }

    def __init__(self, config) -> None:
        super().__init__()
        self.language_model = Qwen35MoEModel(config)
        self.lm_head = LMHead(config.vocab_size, config.hidden_size)
        if getattr(config, 'tie_word_embeddings', False):
            self.lm_head.weight = self.language_model.embed_tokens.weight

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return self.language_model(input_ids, positions)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)
