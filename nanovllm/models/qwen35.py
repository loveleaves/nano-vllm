import torch
import torch.nn.functional as F
from torch import nn

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import (
    ColumnParallelLinear, MergedColumnParallelLinear, RowParallelLinear,
)
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabEmbedding, LMHead
from nanovllm.utils.context import get_context


# ── 归一化层 ──────────────────────────────────────────────────────────────────

class Qwen35RMSNorm(nn.Module):
    """零初始化乘性权重的 RMSNorm：(1 + w) * rms_norm(x)。
    与 vllm Qwen3NextRMSNorm 行为一致（zeros-init）。
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))   # zeros init

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return ((1.0 + self.weight.float()) * xf).to(x.dtype)


# ── GatedDeltaNet（线性注意力）─────────────────────────────────────────────────

def _recurrent_step(
    state: torch.Tensor,   # [nv, dk, dv] float32
    q: torch.Tensor,       # [nk, dk]
    k: torch.Tensor,       # [nk, dk]  L2-normalized
    v: torch.Tensor,       # [nv, dv]
    g: torch.Tensor,       # [nv]  负值，exp(g) 为衰减因子（已是 nv-sized）
    beta: torch.Tensor,    # [nv]  已 sigmoid，更新率（已是 nv-sized）
) -> tuple[torch.Tensor, torch.Tensor]:
    """单步 GatedDeltaNet 递推，返回 (out [nv, dv], new_state [nv, dk, dv])。
    g/beta 已是 nv-sized，只扩展 k（和 q）以匹配 nv。
    """
    nv, dk, dv = state.shape
    nk = k.shape[0]

    # GQA：当 nv > nk 时只扩展 k 和 q，g/beta 已经是 nv-sized
    if nv > nk:
        ratio = nv // nk
        k   = k.repeat_interleave(ratio, dim=0)
        q_f = q.float().repeat_interleave(ratio, dim=0)
    else:
        q_f = q.float()

    k_f = k.float()
    v_f = v.float()

    # 状态衰减：state[v] *= exp(g[v])
    state = state * g.exp()[:, None, None]

    # 检索：kv_mem[v, d] = Σ_k state[v, k, d] * k[v, k]
    kv_mem = torch.einsum("vk,vkd->vd", k_f, state)          # [nv, dv]

    # 差分更新
    delta = (v_f - kv_mem) * beta[:, None]                    # [nv, dv]
    state = state + torch.einsum("vk,vd->vkd", k_f, delta)   # [nv, dk, dv]

    # 输出查询
    out = torch.einsum("vk,vkd->vd", q_f, state)              # [nv, dv]
    return out.to(v.dtype), state


class GatedDeltaNet(nn.Module):
    """
    GatedDeltaNet 线性注意力层，权重命名与 vllm Qwen3Next 实现保持一致：
      in_proj_qkv  — q/k/v 合并投影（不含 z）
      in_proj_z    — 输出门投影（独立）
      in_proj_b    — beta 投影（per-head）
      in_proj_a    — dt a 投影（per-head）
      conv1d       — 深度 causal conv，bias=False
      norm         — Qwen35RMSNorm(dv)，per-head 归一化
      out_proj     — 输出线性层
    """

    def __init__(self, config):
        super().__init__()
        hidden  = config.hidden_size
        nk      = config.linear_num_key_heads       # 16
        nv      = config.linear_num_value_heads     # 16
        dk      = config.linear_key_head_dim        # 128
        dv      = config.linear_value_head_dim      # 128
        kernel  = config.linear_conv_kernel_dim     # 4
        conv_dim = nk * dk + nk * dk + nv * dv      # 6144

        self.nk, self.nv = nk, nv
        self.dk, self.dv = dk, dv
        self.kernel   = kernel
        self.conv_dim = conv_dim
        self.q_dim = nk * dk
        self.k_dim = nk * dk
        self.v_dim = nv * dv

        # 投影层（与 vllm qwen3_next.py 命名一致）
        self.in_proj_qkv = ColumnParallelLinear(hidden, conv_dim, bias=False)
        self.in_proj_z   = ColumnParallelLinear(hidden, nv * dv, bias=False)
        self.in_proj_b   = ColumnParallelLinear(hidden, nv, bias=False)   # beta，维度为 nv
        self.in_proj_a   = ColumnParallelLinear(hidden, nv, bias=False)   # dt a，维度为 nv

        # causal conv，bias=False（与实际权重文件一致）
        self.conv1d = nn.Conv1d(conv_dim, conv_dim, kernel,
                                groups=conv_dim, bias=False, padding=kernel - 1)

        # per-head 输出归一化（weight shape: [dv]，ones-init 标准 RMSNorm）
        self.norm     = RMSNorm(dv, eps=config.rms_norm_eps)
        self.out_proj = RowParallelLinear(nv * dv, hidden, bias=False)

        # SSM 可学习参数（float32 精度），维度为 nv（35B: nv=32≠nk=16；2B: nk=nv=16 无差别）
        self.A_log   = nn.Parameter(torch.empty(nv))
        self.dt_bias = nn.Parameter(torch.empty(nv))

        # 状态张量（由 ModelRunner 注入）
        self.conv_state: torch.Tensor = torch.empty(0)
        self.recurrent_state: torch.Tensor = torch.empty(0)

    def allocate_states(self, max_seqs: int):
        """由 ModelRunner 在 warmup 前调用，分配 conv/recurrent 状态池。"""
        device = self.A_log.device
        conv_dtype = self.in_proj_qkv.weight.dtype   # bf16
        self.conv_state = torch.zeros(
            max_seqs, self.conv_dim, self.kernel, device=device, dtype=conv_dtype
        )
        self.recurrent_state = torch.zeros(
            max_seqs, self.nv, self.dk, self.dv, device=device, dtype=torch.float32
        )

    def _split_projections(self, h: torch.Tensor):
        """投影并分离各分量，返回 (q_raw, k_raw, v_raw, z, beta, g)。"""
        qkv  = self.in_proj_qkv(h)                                   # [T, conv_dim]
        z    = self.in_proj_z(h)                                      # [T, nv*dv]
        b    = self.in_proj_b(h)                                      # [T, nk]
        a    = self.in_proj_a(h)                                      # [T, nk]

        q_raw = qkv[:, :self.q_dim]
        k_raw = qkv[:, self.q_dim: self.q_dim + self.k_dim]
        v_raw = qkv[:, self.q_dim + self.k_dim:]

        beta = torch.sigmoid(b)                                       # [T, nk]
        g    = -self.A_log.exp() * F.softplus(a + self.dt_bias)      # [T, nk]

        return q_raw, k_raw, v_raw, z, beta, g

    def _apply_conv_prefill(self, q_raw: torch.Tensor, k_raw: torch.Tensor,
                             v_raw: torch.Tensor, slot: int):
        """整段 causal conv1d（prefill），保存尾部状态。"""
        T = q_raw.shape[0]
        qkv = torch.cat([q_raw, k_raw, v_raw], dim=-1).mT.unsqueeze(0)  # [1, conv_dim, T]
        # padding=kernel-1 → 输出长 T+kernel-1，取前 T 实现因果性
        out = F.silu(self.conv1d(qkv)[:, :, :T])                         # [1, conv_dim, T]
        tail_len = min(T, self.kernel)
        self.conv_state[slot, :, -tail_len:].copy_(qkv[0, :, -tail_len:].detach())
        q_c = out[0, :self.q_dim, :].mT                                   # [T, nk*dk]
        k_c = out[0, self.q_dim:self.q_dim + self.k_dim, :].mT
        v_c = out[0, self.q_dim + self.k_dim:, :].mT
        return q_c, k_c, v_c

    def _apply_conv_decode(self, q_raw: torch.Tensor, k_raw: torch.Tensor,
                            v_raw: torch.Tensor, slot: int):
        """单步 conv1d（decode），滚动更新 conv_state。"""
        token = torch.cat([q_raw, k_raw, v_raw]).unsqueeze(0).unsqueeze(-1)  # [1, conv_dim, 1]
        combined = torch.cat([self.conv_state[slot].unsqueeze(0), token], dim=-1)  # [1, cv, k+1]
        self.conv_state[slot].copy_(combined[0, :, -self.kernel:].detach())
        # F.conv1d 无 padding，利用已滚动的状态作左侧上下文
        out = F.silu(
            F.conv1d(combined, self.conv1d.weight, None,
                     groups=self.conv_dim)[:, :, -1:]
        )                                                                  # [1, conv_dim, 1]
        q_c = out[0, :self.q_dim, 0]
        k_c = out[0, self.q_dim:self.q_dim + self.k_dim, 0]
        v_c = out[0, self.q_dim + self.k_dim:, 0]
        return q_c, k_c, v_c

    def _prefill(self, hidden: torch.Tensor, slots: list) -> torch.Tensor:
        context = get_context()
        cu_q    = context.cu_seqlens_q
        out_parts = []

        for i, slot in enumerate(slots):
            h = hidden[cu_q[i]:cu_q[i + 1]]    # [T, hidden]
            T = h.shape[0]
            q_raw, k_raw, v_raw, z, beta, g = self._split_projections(h)

            q_c, k_c, v_c = self._apply_conv_prefill(q_raw, k_raw, v_raw, slot)

            scale = self.dk ** -0.5
            q_c = F.normalize(q_c.view(T, self.nk, self.dk), dim=-1) * scale  # [T, nk, dk]
            k_c = F.normalize(k_c.view(T, self.nk, self.dk), dim=-1)
            v_c = v_c.view(T, self.nv, self.dv)

            state = self.recurrent_state[slot].clone()  # [nv, dk, dv] float32
            seq_out = []
            for t in range(T):
                o_t, state = _recurrent_step(
                    state, q_c[t], k_c[t], v_c[t], g[t], beta[t]
                )
                seq_out.append(o_t)
            self.recurrent_state[slot].copy_(state.detach())

            # core_out: [T, nv, dv]
            core_out = torch.stack(seq_out, dim=0).to(h.dtype)         # [T, nv, dv]
            normed   = self.norm(core_out)                               # [T, nv, dv]
            # 输出门：z → [T, nv, dv]，与 normed 逐元素相乘
            z_gate   = z.view(T, self.nv, self.dv)
            gated    = (normed * F.silu(z_gate)).reshape(T, -1)         # [T, nv*dv]
            out_parts.append(self.out_proj(gated))

        return torch.cat(out_parts, dim=0)

    def _decode(self, hidden: torch.Tensor, slots: list) -> torch.Tensor:
        out_parts = []
        for i, slot in enumerate(slots):
            h = hidden[i:i + 1]                  # [1, hidden]
            q_raw, k_raw, v_raw, z, beta, g = self._split_projections(h)

            q_c, k_c, v_c = self._apply_conv_decode(
                q_raw[0], k_raw[0], v_raw[0], slot
            )

            scale = self.dk ** -0.5
            q_c = F.normalize(q_c.view(self.nk, self.dk), dim=-1) * scale
            k_c = F.normalize(k_c.view(self.nk, self.dk), dim=-1)
            v_c = v_c.view(self.nv, self.dv)

            state = self.recurrent_state[slot].clone()  # [nv, dk, dv] float32
            o, state = _recurrent_step(state, q_c, k_c, v_c, g[0], beta[0])
            self.recurrent_state[slot].copy_(state.detach())

            # o: [nv, dv]
            normed = self.norm(o)                                        # [nv, dv]
            z_gate = z[0].view(self.nv, self.dv)
            gated  = (normed * F.silu(z_gate)).reshape(1, -1).to(h.dtype)  # [1, nv*dv]
            out_parts.append(self.out_proj(gated))

        return torch.cat(out_parts, dim=0)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        context = get_context()
        slots   = context.lin_attn_seq_slots
        if context.is_prefill:
            return self._prefill(hidden, slots)
        return self._decode(hidden, slots)


# ── Qwen35Attention（全注意力 + 输出门）───────────────────────────────────────

class Qwen35Attention(nn.Module):
    """全注意力层：GQA + QK-Norm(Qwen35RMSNorm) + Partial RoPE + 输出门。
    k_proj / v_proj 独立（与实际权重文件保持一致）。
    """

    def __init__(self, config):
        super().__init__()
        hidden   = config.hidden_size
        num_h    = config.num_attention_heads
        num_kv_h = config.num_key_value_heads
        head_dim = getattr(config, 'head_dim', hidden // num_h)
        partial  = getattr(config, 'partial_rotary_factor', 1.0)
        rotary_dim = int(head_dim * partial)

        # q_proj 输出 2×num_h×head_dim：前半 query，后半 output gate
        self.q_proj = ColumnParallelLinear(hidden, 2 * num_h * head_dim, bias=False)
        # k/v 独立（vllm 实现中无合并）
        self.k_proj = ColumnParallelLinear(hidden, num_kv_h * head_dim, bias=False)
        self.v_proj = ColumnParallelLinear(hidden, num_kv_h * head_dim, bias=False)
        self.o_proj = RowParallelLinear(num_h * head_dim, hidden, bias=False)

        self.q_norm = Qwen35RMSNorm(head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen35RMSNorm(head_dim, eps=config.rms_norm_eps)

        rope_theta = getattr(config, 'rope_theta', 1_000_000)
        self.rotary_emb = get_rope(
            head_dim, rotary_dim=rotary_dim,
            max_position=config.max_position_embeddings,
            base=rope_theta,
        )
        self.attn = Attention(num_h, head_dim, head_dim ** -0.5, num_kv_h)

        self.num_heads    = num_h
        self.num_kv_heads = num_kv_h
        self.head_dim     = head_dim

    def forward(self, positions: torch.Tensor,
                hidden_states: torch.Tensor) -> torch.Tensor:
        T = hidden_states.shape[0]

        # q_proj → split query / gate per head (each head: first half=query, second half=gate)
        # matches HF: view(T, num_heads, head_dim*2).chunk(2, dim=-1)
        qg   = self.q_proj(hidden_states).view(T, self.num_heads, self.head_dim * 2)
        q, gate = qg.chunk(2, dim=-1)               # each [T, num_heads, head_dim]
        gate = gate.reshape(T, self.num_heads * self.head_dim)

        k = self.k_proj(hidden_states).view(T, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(T, self.num_kv_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = self.rotary_emb(positions, q, k)

        o = self.attn(q, k, v)                      # [T, num_heads, head_dim]
        o = o.flatten(1) * torch.sigmoid(gate)
        return self.o_proj(o)


# ── Decoder Layer ──────────────────────────────────────────────────────────────

class Qwen35MLP(nn.Module):
    """Qwen3.5 FFN：SwiGLU。"""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size, [intermediate_size, intermediate_size], bias=False
        )
        self.down_proj = RowParallelLinear(intermediate_size, hidden_size, bias=False)
        self.act_fn    = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_up_proj(x)))


class Qwen35DecoderLayer(nn.Module):
    """
    Qwen3.5 解码层。子模块命名与 vllm 保持一致：
      linear_attention 层 → self.linear_attn (GatedDeltaNet)
      full_attention 层   → self.self_attn   (Qwen35Attention)
    层归一化均使用 Qwen35RMSNorm（zeros-init）。
    """

    def __init__(self, config, layer_type: str):
        super().__init__()
        self.layer_type = layer_type
        if layer_type == "linear_attention":
            self.linear_attn = GatedDeltaNet(config)
        else:
            self.self_attn = Qwen35Attention(config)
        self.mlp = Qwen35MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
        )
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

class Qwen35Model(nn.Module):
    """Qwen3.5 混合 Transformer 骨干（不含 LM Head）。"""

    def __init__(self, config):
        super().__init__()
        self.embed_tokens = VocabEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            Qwen35DecoderLayer(config, lt) for lt in config.layer_types
        ])
        self.norm = Qwen35RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor,
                positions: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        # 最后一层之后 hidden_states 是 mlp 输出，residual 是其输入；需合并再归一化
        hidden_states = self.norm(hidden_states + residual)
        return hidden_states


class Qwen35ForCausalLM(nn.Module):
    """
    Qwen3.5-2B 因果语言模型。

    权重加载路径映射（loader 先跳过 skip_prefixes，再剥离 weight_prefix_to_strip）：
      文件 key:  model.language_model.layers.0.linear_attn.in_proj_qkv.weight
        → 剥离 "model." → language_model.layers.0.linear_attn.in_proj_qkv.weight
        → 匹配   self.language_model.layers[0].linear_attn.in_proj_qkv.weight  ✓
    """

    weight_prefix_to_strip = "model."
    weight_skip_prefixes   = ("model.visual.", "mtp.")

    packed_modules_mapping = {
        "gate_proj": ("gate_up_proj", 0),
        "up_proj":   ("gate_up_proj", 1),
    }

    def __init__(self, config) -> None:
        super().__init__()
        self.language_model = Qwen35Model(config)
        self.lm_head = LMHead(config.vocab_size, config.hidden_size)
        if getattr(config, 'tie_word_embeddings', False):
            self.lm_head.weight = self.language_model.embed_tokens.weight

    def forward(self, input_ids: torch.Tensor,
                positions: torch.Tensor) -> torch.Tensor:
        return self.language_model(input_ids, positions)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)
