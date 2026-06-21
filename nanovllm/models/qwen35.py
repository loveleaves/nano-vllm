"""
Qwen3.5 dense（混合线性注意力）文本模型 —— 对齐 vLLM 0.15.1 的 Qwen3-Next 系列适配。

架构要点（取自实际权重 Qwen3.5-2B 的 config.text_config / safetensors 布局）：
  - 24 层，按 ``layer_types`` 分发：每 4 层一个 ``full_attention``，其余 ``linear_attention``
    （``full_attention_interval=4``）。
  - 线性注意力层 = GatedDeltaNet（门控 Δ-rule 线性注意力 + 因果短卷积），权重命名
    ``linear_attn.{in_proj_qkv,in_proj_z,in_proj_b,in_proj_a,conv1d,norm,out_proj,A_log,dt_bias}``。
    注意：这是**分离投影**布局（区别于 vLLM main 中 Qwen3-Next-80B 的合并
    ``in_proj_qkvz``/``in_proj_ba``）。
  - 全注意力层 = GQA + QK-Norm + 部分 RoPE（``partial_rotary_factor=0.25``）+ 输出门
    （``attn_output_gate=true``：``q_proj`` 输出 2×heads，后半为 sigmoid 门）。
  - 归一化：``input_layernorm`` / ``post_attention_layernorm`` / 最终 ``norm`` / ``q_norm`` /
    ``k_norm`` 为 **零初始化** 的 ``(1+w)·rmsnorm``（HF GemmaRMSNorm，见 Qwen35RMSNorm）；
    GatedDeltaNet 内部 ``norm`` 为标准 ``w·rmsnorm``（ones-init，RMSNormGated）。
  - ``tie_word_embeddings=true``。

与引擎的契约（线性注意力的递归状态）：
  全注意力层走分页 KV cache（由 ModelRunner 绑定 k_cache/v_cache）。线性注意力层不进 KV
  cache——它维护**每序列的递归状态**（conv_state + recurrent_state），由 ModelRunner 按
  ``max_num_seqs`` 分配状态池，并在每步通过 ``attn_md.state_slots`` 下发该批各序列的槽位。
  GatedDeltaNet 用统一的「以 conv_state 为左context的分段卷积 + 递归扫描」处理 prefill /
  chunked-prefill / decode（query 长度 1 即 decode 退化情形）。
"""
import torch
import torch.nn.functional as F
from torch import nn

from nanovllm.attention import Attention
from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import (
    ColumnParallelLinear, MergedColumnParallelLinear, RowParallelLinear, _get_tp_info,
)
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead
from nanovllm.utils.context import AttentionMetadata


# ── 归一化：零初始化乘性权重 (1+w)·rmsnorm（HF GemmaRMSNorm 语义）──────────────────

class Qwen35RMSNorm(nn.Module):
    """``(1 + weight) · rms_norm(x)``，weight 零初始化。

    与全注意力的 q_norm/k_norm、各层 input/post_attention layernorm、最终 norm 一致。
    （区别于 ``layers.layernorm.RMSNorm`` 的 ones-init ``w·rmsnorm``。）
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return ((1.0 + self.weight.float()) * xf).to(x.dtype)


# ── GatedDeltaNet（线性注意力）────────────────────────────────────────────────

def _recurrent_step(state, q, k, v, g, beta):
    """单步门控 Δ-rule 递推。

    state[nv,dk,dv] float32；q/k[nk,dk]（k 已 L2 归一化、q 已归一化并乘 1/√dk）；
    v[nv,dv]；g[nv]（负值，exp(g) 为衰减）；beta[nv]（sigmoid，更新率）。
    nv≥nk 时按 nv//nk 扩展 k/q（GQA）。返回 (out[nv,dv], new_state[nv,dk,dv])。
    """
    nv, dk, dv = state.shape
    nk = k.shape[0]
    if nv > nk:
        ratio = nv // nk
        k = k.repeat_interleave(ratio, dim=0)
        q = q.repeat_interleave(ratio, dim=0)
    k_f = k.float(); v_f = v.float(); q_f = q.float()

    state = state * g.exp()[:, None, None]                  # 衰减
    kv_mem = torch.einsum("vk,vkd->vd", k_f, state)         # 检索 [nv,dv]
    delta = (v_f - kv_mem) * beta[:, None]                  # 差分更新
    state = state + torch.einsum("vk,vd->vkd", k_f, delta)
    out = torch.einsum("vk,vkd->vd", q_f, state)            # 查询 [nv,dv]
    return out.to(v.dtype), state


class GatedDeltaNet(nn.Module):
    """门控 Δ-rule 线性注意力 + 因果深度卷积。状态池由 ModelRunner 注入。"""

    # 供 ModelRunner 鸭子类型探测「此模块需要递归状态池」（避免引擎硬依赖具体模型）
    needs_state_pool = True

    def __init__(self, config):
        super().__init__()
        _, tp_size = _get_tp_info()
        assert tp_size == 1, "GatedDeltaNet（线性注意力）当前仅支持 TP=1"

        hidden = config.hidden_size
        self.nk = config.linear_num_key_heads
        self.nv = config.linear_num_value_heads
        self.dk = config.linear_key_head_dim
        self.dv = config.linear_value_head_dim
        self.kernel = config.linear_conv_kernel_dim
        self.q_dim = self.nk * self.dk
        self.k_dim = self.nk * self.dk
        self.v_dim = self.nv * self.dv
        self.conv_dim = self.q_dim + self.k_dim + self.v_dim

        # 分离投影（与实际权重命名一致）
        self.in_proj_qkv = ColumnParallelLinear(hidden, self.conv_dim, bias=False)
        self.in_proj_z = ColumnParallelLinear(hidden, self.v_dim, bias=False)
        self.in_proj_b = ColumnParallelLinear(hidden, self.nv, bias=False)
        self.in_proj_a = ColumnParallelLinear(hidden, self.nv, bias=False)
        # 深度因果卷积，weight 形状 [conv_dim,1,kernel]，bias=False
        self.conv1d = nn.Conv1d(self.conv_dim, self.conv_dim, self.kernel,
                                groups=self.conv_dim, bias=False)
        # GatedDeltaNet 内部 norm：标准 ones-init RMSNorm（per-head，dv 维）
        self.norm = RMSNorm(self.dv, eps=config.rms_norm_eps)
        self.out_proj = RowParallelLinear(self.v_dim, hidden, bias=False)
        # SSM 可学习参数（float32 精度）；零初始化使未加载权重的模块也良定义（A=exp(0)=1）
        self.A_log = nn.Parameter(torch.zeros(self.nv))
        self.dt_bias = nn.Parameter(torch.zeros(self.nv))

        # 递归状态（由 ModelRunner.allocate / 绑定）：
        #   conv_state[num_slots, conv_dim, kernel-1] —— 卷积左侧上下文（原始 qkv 历史）
        #   recurrent_state[num_slots, nv, dk, dv]    —— Δ-rule 线性注意力记忆，float32
        self.conv_state: torch.Tensor | None = None
        self.recurrent_state: torch.Tensor | None = None

    def allocate_state(self, num_slots: int) -> None:
        """ModelRunner 在 warmup 前调用，按 max_num_seqs 分配状态池。"""
        dev = self.A_log.device
        conv_dtype = self.in_proj_qkv.weight.dtype
        self.conv_state = torch.zeros(num_slots, self.conv_dim, self.kernel - 1,
                                      device=dev, dtype=conv_dtype)
        self.recurrent_state = torch.zeros(num_slots, self.nv, self.dk, self.dv,
                                           device=dev, dtype=torch.float32)

    def _split(self, h: torch.Tensor):
        qkv = self.in_proj_qkv(h)
        z = self.in_proj_z(h)
        beta = torch.sigmoid(self.in_proj_b(h))                       # [T, nv]
        a = self.in_proj_a(h)                                         # [T, nv]
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias.float())
        q_raw = qkv[:, :self.q_dim]
        k_raw = qkv[:, self.q_dim:self.q_dim + self.k_dim]
        v_raw = qkv[:, self.q_dim + self.k_dim:]
        return q_raw, k_raw, v_raw, z, beta, g

    def _conv(self, q_raw, k_raw, v_raw, slot, T):
        """以 conv_state 为左 context 的分段因果卷积（统一 prefill/decode），并滚动更新状态。"""
        qkv = torch.cat([q_raw, k_raw, v_raw], dim=-1).mT                  # [conv_dim, T]
        prev = self.conv_state[slot]                                      # [conv_dim, kernel-1]
        inp = torch.cat([prev, qkv], dim=-1)                             # [conv_dim, kernel-1+T]
        out = F.silu(F.conv1d(inp.unsqueeze(0), self.conv1d.weight, None,
                              groups=self.conv_dim))[0]                    # [conv_dim, T]
        # 更新左 context：取末尾 kernel-1 列（即下一段卷积所需历史）
        self.conv_state[slot].copy_(inp[:, -(self.kernel - 1):].detach())
        out = out.mT                                                      # [T, conv_dim]
        return (out[:, :self.q_dim], out[:, self.q_dim:self.q_dim + self.k_dim],
                out[:, self.q_dim + self.k_dim:])

    def _run_seq(self, h: torch.Tensor, slot: int) -> torch.Tensor:
        T = h.shape[0]
        q_raw, k_raw, v_raw, z, beta, g = self._split(h)
        q_c, k_c, v_c = self._conv(q_raw, k_raw, v_raw, slot, T)

        scale = self.dk ** -0.5
        q_c = F.normalize(q_c.view(T, self.nk, self.dk), dim=-1) * scale
        k_c = F.normalize(k_c.view(T, self.nk, self.dk), dim=-1)
        v_c = v_c.view(T, self.nv, self.dv)

        state = self.recurrent_state[slot].clone()                       # [nv,dk,dv] f32
        seq_out = []
        for t in range(T):
            o_t, state = _recurrent_step(state, q_c[t], k_c[t], v_c[t], g[t], beta[t])
            seq_out.append(o_t)
        self.recurrent_state[slot].copy_(state.detach())

        core = torch.stack(seq_out, dim=0).to(h.dtype)                   # [T,nv,dv]
        normed = self.norm(core)                                         # 标准 RMSNorm（per-head）
        gated = (normed * F.silu(z.view(T, self.nv, self.dv))).reshape(T, -1)
        return self.out_proj(gated)

    def forward(self, hidden: torch.Tensor, attn_md: AttentionMetadata) -> torch.Tensor:
        # query_start_loc 给出批内各序列在扁平 hidden 中的分段；state_slots 给出各序列状态槽
        cu = attn_md.query_start_loc.tolist()
        slots = attn_md.state_slots
        assert slots is not None, "线性注意力需要 attn_md.state_slots（由 ModelRunner 下发）"
        parts = [self._run_seq(hidden[cu[i]:cu[i + 1]], slot) for i, slot in enumerate(slots)]
        return torch.cat(parts, dim=0)


# ── 全注意力（GQA + QK-Norm + 部分 RoPE + 输出门）──────────────────────────────

class Qwen35Attention(nn.Module):

    def __init__(self, config):
        super().__init__()
        hidden = config.hidden_size
        num_h = config.num_attention_heads
        num_kv = config.num_key_value_heads
        head_dim = getattr(config, "head_dim", None) or hidden // num_h
        self.num_heads = num_h
        self.num_kv_heads = num_kv
        self.head_dim = head_dim

        rp = getattr(config, "rope_parameters", None) or {}
        partial = rp.get("partial_rotary_factor",
                         getattr(config, "partial_rotary_factor", 1.0))
        rope_theta = rp.get("rope_theta", getattr(config, "rope_theta", 1_000_000))
        rotary_dim = int(head_dim * partial)

        # q_proj 输出 2×num_h×head_dim：每个 head 前半 query、后半输出门（attn_output_gate）
        self.q_proj = ColumnParallelLinear(hidden, 2 * num_h * head_dim, bias=False)
        self.k_proj = ColumnParallelLinear(hidden, num_kv * head_dim, bias=False)
        self.v_proj = ColumnParallelLinear(hidden, num_kv * head_dim, bias=False)
        self.o_proj = RowParallelLinear(num_h * head_dim, hidden, bias=False)

        self.q_norm = Qwen35RMSNorm(head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen35RMSNorm(head_dim, eps=config.rms_norm_eps)
        self.rotary_emb = get_rope(head_dim, rotary_dim,
                                   config.max_position_embeddings, rope_theta)
        self.attn = Attention(num_h, head_dim, head_dim ** -0.5, num_kv)

    def forward(self, positions, hidden_states, attn_md):
        T = hidden_states.shape[0]
        qg = self.q_proj(hidden_states).view(T, self.num_heads, 2 * self.head_dim)
        q, gate = qg.chunk(2, dim=-1)                                    # 各 [T, num_heads, head_dim]
        gate = gate.reshape(T, self.num_heads * self.head_dim)

        k = self.k_proj(hidden_states).view(T, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(T, self.num_kv_heads, self.head_dim)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = self.rotary_emb(positions, q, k)

        o = self.attn(q, k, v, attn_md).flatten(1)                      # [T, num_heads*head_dim]
        o = o * torch.sigmoid(gate)                                      # 输出门
        return self.o_proj(o)


# ── MLP / 解码层 / 骨干 ────────────────────────────────────────────────────────

class Qwen35MLP(nn.Module):

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size, [intermediate_size, intermediate_size], bias=False)
        self.down_proj = RowParallelLinear(intermediate_size, hidden_size, bias=False)
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_up_proj(x)))


class Qwen35DecoderLayer(nn.Module):
    """按 layer_type 分发 linear_attn / self_attn；Pre-LN 残差。"""

    def __init__(self, config, layer_type: str):
        super().__init__()
        self.layer_type = layer_type
        if layer_type == "linear_attention":
            self.linear_attn = GatedDeltaNet(config)
        else:
            self.self_attn = Qwen35Attention(config)
        self.mlp = Qwen35MLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = Qwen35RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen35RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, positions, hidden_states, residual, attn_md):
        # Qwen35RMSNorm 不做 fused add，故残差在层内显式累加
        if residual is None:
            residual = hidden_states
        else:
            hidden_states = hidden_states + residual
            residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(hidden_states, attn_md)
        else:
            hidden_states = self.self_attn(positions, hidden_states, attn_md)

        hidden_states = hidden_states + residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen35Model(nn.Module):
    """Qwen3.5 混合 Transformer 骨干（embed + layers + final norm）。"""

    def __init__(self, config):
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [Qwen35DecoderLayer(config, lt) for lt in config.layer_types])
        self.norm = Qwen35RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids, positions, attn_md):
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual, attn_md)
        return self.norm(hidden_states + residual)


class _LanguageModelShell(nn.Module):
    """命名外壳：使参数路径对齐权重键 ``model.language_model.*``（VLM 包装的文本主干）。"""

    def __init__(self, config):
        super().__init__()
        self.language_model = Qwen35Model(config)


class Qwen35ForCausalLM(nn.Module):
    """
    Qwen3.5 dense 文本因果语言模型（注册名 ``Qwen3_5ForConditionalGeneration``）。

    HF 权重为 VLM 布局（``model.language_model.*`` + ``model.visual.*``）；nano 仅取文本
    主干，视觉权重在加载时按「参数不存在」自动跳过。dense = MLP 为标准 SwiGLU（无 MoE）。
    """

    # MLP 的 gate/up 合并；其余（含分离的 q/k/v、linear_attn.in_proj_*）按权重名直接加载
    packed_modules_mapping = {
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config) -> None:
        super().__init__()
        # 顶层为 VLM config，取其 text_config 作为文本主干配置
        tc = getattr(config, "text_config", config)
        self.config = tc
        self.model = _LanguageModelShell(tc)
        self.lm_head = ParallelLMHead(tc.vocab_size, tc.hidden_size)
        if getattr(tc, "tie_word_embeddings", False):
            self.lm_head.weight.data = self.model.language_model.embed_tokens.weight.data

    def forward(self, input_ids, positions, attn_md):
        return self.model.language_model(input_ids, positions, attn_md)

    def compute_logits(self, hidden_states, attn_md=None):
        return self.lm_head(hidden_states, attn_md)
