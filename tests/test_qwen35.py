"""Qwen3.5 dense（混合线性注意力）模型单元测试。

不依赖真实权重/GPU：用微型随机配置在 CPU fp32 上验证
  - Qwen35RMSNorm 的 (1+w) 语义
  - 部分 RoPE（rotary_dim < head_dim）
  - GatedDeltaNet 单步递推数值性质
  - **前缀一致性**（核心不变量）：整段处理 == 分步处理（conv/recurrent 状态续算正确）
    既在 GatedDeltaNet 模块层、也在全线性模型层验证（覆盖 state_slots 下发链路）。
"""
import torch
import torch.nn.functional as F

from nanovllm.models.qwen35 import (
    Qwen35RMSNorm, GatedDeltaNet, Qwen35ForCausalLM, _recurrent_step,
)
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.utils.context import AttentionMetadata

torch.set_default_dtype(torch.float32)


class FakeTextConfig:
    """微型文本主干配置（nv*dv == hidden_size，rotary_dim 偶数）。"""
    hidden_size = 64
    intermediate_size = 128
    vocab_size = 100
    num_attention_heads = 4
    num_key_value_heads = 2
    head_dim = 16
    rms_norm_eps = 1e-6
    max_position_embeddings = 512
    tie_word_embeddings = True
    linear_num_key_heads = 2
    linear_num_value_heads = 2
    linear_key_head_dim = 32
    linear_value_head_dim = 32   # nv*dv = 2*32 = 64 = hidden_size
    linear_conv_kernel_dim = 4
    rope_parameters = {"partial_rotary_factor": 0.25, "rope_theta": 1_000_000}

    def __init__(self, layer_types):
        self.layer_types = layer_types


def _randomize(module):
    """微型随机权重：未加载真实权重时给所有参数赋小值，避免 torch.empty 垃圾值溢出成 NaN。"""
    for p in module.parameters():
        torch.nn.init.normal_(p, std=0.02)
    for m in module.modules():
        if getattr(m, "needs_state_pool", False):
            torch.nn.init.zeros_(m.A_log)
            torch.nn.init.zeros_(m.dt_bias)
    return module


def _md(seg_lens, state_slots):
    """构造仅供线性注意力使用的 attn_md（query_start_loc + state_slots）。"""
    cu = [0]
    for q in seg_lens:
        cu.append(cu[-1] + q)
    return AttentionMetadata(
        query_start_loc=torch.tensor(cu, dtype=torch.int32),
        max_query_len=max(seg_lens), state_slots=state_slots,
    )


# ── Qwen35RMSNorm ─────────────────────────────────────────────────────────────

def test_rmsnorm_zero_weight_is_plain_normalize():
    norm = Qwen35RMSNorm(16)
    assert torch.allclose(norm.weight, torch.zeros(16))
    x = torch.randn(4, 16)
    expected = F.normalize(x, dim=-1) * (16 ** 0.5)   # rms_norm == normalize*sqrt(d)
    assert torch.allclose(norm(x), expected, atol=1e-5)


def test_rmsnorm_one_plus_weight():
    norm = Qwen35RMSNorm(8)
    with torch.no_grad():
        norm.weight.copy_(torch.full((8,), 0.5))
    x = torch.randn(3, 8)
    base = Qwen35RMSNorm(8)(x)        # weight=0 → 纯归一化
    assert torch.allclose(norm(x), base * 1.5, atol=1e-5)


# ── 部分 RoPE ─────────────────────────────────────────────────────────────────

def test_partial_rope_passes_through_tail():
    head_dim, rotary_dim = 16, 4
    rope = get_rope(head_dim, rotary_dim, 128, 1_000_000)
    pos = torch.arange(5)
    q = torch.randn(5, 2, head_dim)
    k = torch.randn(5, 2, head_dim)
    q2, k2 = rope(pos, q.clone(), k.clone())
    # 尾部（rotary_dim 之后）维度原样直通
    assert torch.allclose(q2[..., rotary_dim:], q[..., rotary_dim:])
    # 前 rotary_dim 维被旋转（位置 0 不变，位置>0 改变）
    assert torch.allclose(q2[0, :, :rotary_dim], q[0, :, :rotary_dim], atol=1e-5)
    assert not torch.allclose(q2[1:, :, :rotary_dim], q[1:, :, :rotary_dim])


# ── 单步递推 ──────────────────────────────────────────────────────────────────

def test_recurrent_step_shapes_and_decay():
    nv, dk, dv, nk = 2, 32, 32, 2
    state = torch.randn(nv, dk, dv)
    q = torch.randn(nk, dk); k = F.normalize(torch.randn(nk, dk), dim=-1)
    v = torch.randn(nv, dv)
    g = torch.full((nv,), -100.0)       # exp(g)≈0 → 旧状态被完全衰减
    beta = torch.zeros(nv)              # 不更新 → 新状态≈0 → 输出≈0
    out, new_state = _recurrent_step(state, q, k, v, g, beta)
    assert out.shape == (nv, dv) and new_state.shape == (nv, dk, dv)
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-4)


# ── GatedDeltaNet 前缀一致性（模块层）─────────────────────────────────────────

def test_gdn_prefix_consistency():
    torch.manual_seed(0)
    cfg = FakeTextConfig(["linear_attention"])
    gdn = _randomize(GatedDeltaNet(cfg))
    gdn.allocate_state(2)
    T = 6
    x = torch.randn(T, cfg.hidden_size)

    # (a) 整段一次性处理（slot 0）
    full = gdn(x, _md([T], [0]))

    # (b) 分步：先 0..T-1（slot 1），再续算最后一个 token
    gdn(x[:T - 1], _md([T - 1], [1]))
    last = gdn(x[T - 1:T], _md([1], [1]))

    assert torch.allclose(full[T - 1:T], last, atol=1e-4), \
        "分步续算的末位输出须等于整段处理（conv/recurrent 状态续算一致）"


def test_gdn_slot_isolation():
    """不同槽位的序列状态互不串扰。"""
    torch.manual_seed(1)
    cfg = FakeTextConfig(["linear_attention"])
    gdn = _randomize(GatedDeltaNet(cfg))
    gdn.allocate_state(3)
    x = torch.randn(4, cfg.hidden_size)
    solo = gdn(x, _md([4], [0]))
    # 同时把另一序列放 slot 2，不应改变 slot 0 的结果
    gdn.recurrent_state[0].zero_(); gdn.conv_state[0].zero_()
    batched = gdn(torch.cat([x, torch.randn(4, cfg.hidden_size)]), _md([4, 4], [0, 2]))
    assert torch.allclose(solo, batched[:4], atol=1e-4)


# ── 全线性模型多步前缀一致性（模型层，覆盖 state_slots 链路）───────────────────

def _alloc_model_states(model, num_slots):
    for m in model.modules():
        if getattr(m, "needs_state_pool", False):
            m.allocate_state(num_slots)


def test_all_linear_model_prefix_consistency():
    torch.manual_seed(2)
    cfg = FakeTextConfig(["linear_attention"] * 3)   # 纯线性 → 无需 KV cache
    model = _randomize(Qwen35ForCausalLM(cfg))
    _alloc_model_states(model, 2)
    ids = torch.randint(0, cfg.vocab_size, (5,))
    pos = torch.arange(5)

    # 整段 prefill（slot 0）
    h_full = model(ids, pos, _md([5], [0]))
    logits_full = model.compute_logits(h_full)

    # 分步：prefill 前 4（slot 1）+ decode 第 5（slot 1）
    model(ids[:4], pos[:4], _md([4], [1]))
    h_last = model(ids[4:5], pos[4:5], _md([1], [1]))
    logits_last = model.compute_logits(h_last)

    assert torch.allclose(logits_full[4:5], logits_last, atol=1e-3), \
        "模型级：分步解码末位 logits 须等于整段 prefill"


def test_forward_output_shapes():
    cfg = FakeTextConfig(["linear_attention"] * 2)
    model = _randomize(Qwen35ForCausalLM(cfg))
    _alloc_model_states(model, 1)
    ids = torch.randint(0, cfg.vocab_size, (3,))
    h = model(ids, torch.arange(3), _md([3], [0]))
    assert h.shape == (3, cfg.hidden_size)
    logits = model.compute_logits(h)
    assert logits.shape == (3, cfg.vocab_size)


def test_tie_word_embeddings():
    cfg = FakeTextConfig(["linear_attention"])
    model = _randomize(Qwen35ForCausalLM(cfg))
    assert model.lm_head.weight.data_ptr() == \
        model.model.language_model.embed_tokens.weight.data_ptr()
