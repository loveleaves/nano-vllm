"""
Qwen3.5-2B 适配单元 + 集成测试

覆盖层次：
  Unit       — 单个类/函数，无 GPU，无真实权重
  Integration— 多模块协作，CPU 小模型
  GPU        — 需 NANO_VLLM_QWEN35_MODEL 环境变量指向真实权重目录
"""
import os
import tempfile

import pytest
import torch
import torch.nn as nn

from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.utils.context import set_context, reset_context


# ─── Fake config for small CPU model ─────────────────────────────────────────

class FakeQwen35Config:
    """Qwen3.5-2B 缩小版配置，用于 CPU 单元/集成测试。"""
    model_type             = "qwen3_5_text"
    hidden_size            = 64
    intermediate_size      = 128
    num_hidden_layers      = 4          # 3 linear + 1 full，重复 1 次
    num_attention_heads    = 4
    num_key_value_heads    = 2
    head_dim               = 16
    vocab_size             = 200
    max_position_embeddings = 256
    rms_norm_eps           = 1e-6
    rope_theta             = 10000.0
    partial_rotary_factor  = 0.25      # rotary_dim = 16*0.25 = 4
    tie_word_embeddings    = False
    # 线性注意力参数
    linear_num_key_heads   = 2
    linear_num_value_heads = 2
    linear_key_head_dim    = 8
    linear_value_head_dim  = 32     # nv*dv = 2*32 = 64 = hidden_size ✓
    linear_conv_kernel_dim = 4
    # 4 层：linear, linear, linear, full
    layer_types            = ["linear_attention", "linear_attention",
                              "linear_attention", "full_attention"]


# ─── Unit: Qwen35RMSNorm ──────────────────────────────────────────────────────

class TestQwen35RMSNorm:

    @pytest.mark.unit
    def test_weight_initialized_to_zeros(self):
        from nanovllm.models.qwen35 import Qwen35RMSNorm
        norm = Qwen35RMSNorm(32)
        assert torch.all(norm.weight == 0), "weight 应初始化为 0"

    @pytest.mark.unit
    def test_zero_weight_equals_plain_rmsnorm(self):
        """weight=0 时 (1+0)*norm(x) = rms_norm(x)。"""
        from nanovllm.models.qwen35 import Qwen35RMSNorm
        norm = Qwen35RMSNorm(16, eps=1e-6)
        x = torch.randn(4, 16)
        y = norm(x)
        xf = x.float()
        expected = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6)
        assert torch.allclose(y.float(), expected, atol=1e-5)

    @pytest.mark.unit
    def test_nonzero_weight_scales_output(self):
        """weight=1 时输出为 2*rms_norm(x)。"""
        from nanovllm.models.qwen35 import Qwen35RMSNorm
        norm = Qwen35RMSNorm(8)
        nn.init.ones_(norm.weight)
        x = torch.randn(3, 8)
        y_ones = norm(x)
        norm2 = Qwen35RMSNorm(8)   # weight=0
        y_zero = norm2(x)
        assert torch.allclose(y_ones.float(), (2 * y_zero).float(), atol=1e-5)

    @pytest.mark.unit
    def test_output_shape_preserved(self):
        from nanovllm.models.qwen35 import Qwen35RMSNorm
        norm = Qwen35RMSNorm(24)
        x = torch.randn(5, 24)
        assert norm(x).shape == x.shape

    @pytest.mark.unit
    def test_dtype_preserved(self):
        from nanovllm.models.qwen35 import Qwen35RMSNorm
        norm = Qwen35RMSNorm(8)
        x = torch.randn(2, 8).half()
        assert norm(x).dtype == torch.float16


# ─── Unit: _recurrent_step ───────────────────────────────────────────────────

class TestRecurrentStep:

    def _make_inputs(self, nk=2, nv=2, dk=4, dv=4):
        state = torch.zeros(nv, dk, dv)
        q = torch.randn(nk, dk)
        k = torch.randn(nk, dk)
        v = torch.randn(nv, dv)
        g = torch.full((nk,), -0.5)
        beta = torch.full((nk,), 0.3)
        return state, q, k, v, g, beta

    @pytest.mark.unit
    def test_output_shape(self):
        from nanovllm.models.qwen35 import _recurrent_step
        state, q, k, v, g, beta = self._make_inputs(nk=2, nv=2, dk=4, dv=4)
        out, new_state = _recurrent_step(state, q, k, v, g, beta)
        assert out.shape == (2, 4)
        assert new_state.shape == (2, 4, 4)

    @pytest.mark.unit
    def test_state_decay(self):
        """g<0 时衰减因子 exp(g)<1，state 范数应缩小。"""
        from nanovllm.models.qwen35 import _recurrent_step
        state = torch.ones(2, 4, 4)
        q = torch.zeros(2, 4)
        k = torch.zeros(2, 4)
        v = torch.zeros(2, 4)
        g = torch.full((2,), -1.0)
        beta = torch.zeros(2)
        _, new_state = _recurrent_step(state, q, k, v, g, beta)
        # state 全 1，g=-1 → 衰减为 exp(-1) ≈ 0.368，无写入（beta=0）
        expected = torch.full_like(state, torch.exp(torch.tensor(-1.0)).item())
        assert torch.allclose(new_state.float(), expected.float(), atol=1e-5)

    @pytest.mark.unit
    def test_zero_state_with_unit_input(self):
        """从零状态出发，单步后 state[v,k,d] = k[v,k] * delta[v,d]。"""
        from nanovllm.models.qwen35 import _recurrent_step
        nk = nv = 1
        dk = dv = 2
        state = torch.zeros(nv, dk, dv)
        k = torch.tensor([[1.0, 0.0]])   # k[0,0]=1, k[0,1]=0
        q = torch.tensor([[1.0, 0.0]])
        v = torch.tensor([[0.0, 1.0]])   # delta = v (初始 state=0, beta=1)
        g = torch.zeros(nk)              # exp(0)=1，无衰减
        beta = torch.ones(nk)
        out, new_state = _recurrent_step(state, q, k, v, g, beta)
        # delta = v = [[0, 1]]
        # state[0,k_idx,d] = k[0,k_idx] * delta[0,d]
        #   state[0,0,0] = 1*0 = 0, state[0,0,1] = 1*1 = 1
        #   state[0,1,0] = 0*0 = 0, state[0,1,1] = 0*1 = 0
        expected_state = torch.tensor([[[0.0, 1.0], [0.0, 0.0]]])
        assert torch.allclose(new_state.float(), expected_state, atol=1e-5)

    @pytest.mark.unit
    def test_gqa_expand(self):
        """nv > nk 时应自动扩展 k/g/beta。"""
        from nanovllm.models.qwen35 import _recurrent_step
        nk, nv, dk, dv = 1, 2, 4, 4
        state = torch.zeros(nv, dk, dv)
        q = torch.randn(nk, dk)
        k = torch.randn(nk, dk)
        v = torch.randn(nv, dv)
        g = torch.full((nk,), -0.3)
        beta = torch.full((nk,), 0.5)
        out, new_state = _recurrent_step(state, q, k, v, g, beta)
        assert out.shape == (nv, dv)
        assert new_state.shape == (nv, dk, dv)


# ─── Unit: GatedDeltaNet ─────────────────────────────────────────────────────

class TestGatedDeltaNet:

    def _make_gdn(self):
        from nanovllm.models.qwen35 import GatedDeltaNet
        gdn = GatedDeltaNet(FakeQwen35Config())
        # GDN 参数用 torch.empty 创建（生产中由权重文件填充），测试必须显式
        # 初始化，否则未初始化内存中的极端值会导致衰减为 0 / beta≈0 等假象
        with torch.no_grad():
            for p in gdn.parameters():
                p.normal_(0, 0.02)
        return gdn

    @pytest.mark.unit
    def test_allocate_states_shapes(self):
        gdn = self._make_gdn()
        gdn.allocate_states(max_seqs=8)
        cfg = FakeQwen35Config()
        conv_dim = cfg.linear_num_key_heads * cfg.linear_key_head_dim * 2 \
                 + cfg.linear_num_value_heads * cfg.linear_value_head_dim
        assert gdn.conv_state.shape == (8, conv_dim, cfg.linear_conv_kernel_dim)
        assert gdn.recurrent_state.shape == (
            8, cfg.linear_num_value_heads,
            cfg.linear_key_head_dim, cfg.linear_value_head_dim
        )

    @pytest.mark.unit
    def test_allocate_states_dtype(self):
        """conv_state 使用模型 dtype，recurrent_state 强制 float32。"""
        from nanovllm.models.qwen35 import GatedDeltaNet
        gdn = GatedDeltaNet(FakeQwen35Config())
        gdn.allocate_states(max_seqs=4)
        assert gdn.recurrent_state.dtype == torch.float32

    @pytest.mark.unit
    def test_slot_isolation(self):
        """更新 slot 0 后，slot 1 的 recurrent_state 不受影响。"""
        from nanovllm.models.qwen35 import GatedDeltaNet, _recurrent_step
        gdn = self._make_gdn()
        gdn.allocate_states(max_seqs=4)
        cfg = FakeQwen35Config()
        nk, nv, dk, dv = (cfg.linear_num_key_heads, cfg.linear_num_value_heads,
                           cfg.linear_key_head_dim, cfg.linear_value_head_dim)

        state0 = gdn.recurrent_state[0].clone()
        q = torch.randn(nk, dk)
        k = torch.randn(nk, dk)
        v = torch.randn(nv, dv)
        g = torch.full((nk,), -0.3)
        beta = torch.full((nk,), 0.5)
        _, new_state = _recurrent_step(state0, q, k, v, g, beta)
        gdn.recurrent_state[0] = new_state

        assert torch.all(gdn.recurrent_state[1] == 0), "slot 1 不应被污染"

    @pytest.mark.unit
    def test_prefill_output_shape(self):
        """prefill 路径：输出 shape 应为 [total_tokens, hidden]。"""
        from nanovllm.models.qwen35 import GatedDeltaNet
        gdn = self._make_gdn()
        gdn.allocate_states(max_seqs=4)
        cfg = FakeQwen35Config()
        T, H = 5, cfg.hidden_size

        hidden = torch.randn(T, H)
        cu_q = torch.tensor([0, 3, 5], dtype=torch.int32)  # 2 个序列
        set_context(is_prefill=True, cu_seqlens_q=cu_q, lin_attn_seq_slots=[0, 1])
        out = gdn(hidden)
        reset_context()
        assert out.shape == (T, H)

    @pytest.mark.unit
    def test_decode_output_shape(self):
        """decode 路径：每个序列 1 token，输出 [num_seqs, hidden]。"""
        from nanovllm.models.qwen35 import GatedDeltaNet
        gdn = self._make_gdn()
        gdn.allocate_states(max_seqs=4)
        cfg = FakeQwen35Config()
        BS, H = 3, cfg.hidden_size

        hidden = torch.randn(BS, H)
        set_context(is_prefill=False, lin_attn_seq_slots=[0, 1, 2])
        out = gdn(hidden)
        reset_context()
        assert out.shape == (BS, H)

    @pytest.mark.unit
    def test_prefill_updates_recurrent_state(self):
        """prefill 后 recurrent_state[slot] 不再全为零。"""
        from nanovllm.models.qwen35 import GatedDeltaNet
        gdn = self._make_gdn()
        gdn.allocate_states(max_seqs=4)
        cfg = FakeQwen35Config()

        hidden = torch.randn(4, cfg.hidden_size)
        cu_q = torch.tensor([0, 4], dtype=torch.int32)
        set_context(is_prefill=True, cu_seqlens_q=cu_q, lin_attn_seq_slots=[0])
        gdn(hidden)
        reset_context()
        assert not torch.all(gdn.recurrent_state[0] == 0)

    @pytest.mark.unit
    def test_decode_updates_conv_state(self):
        """decode 后 conv_state[slot] 应已被滚动更新（不再全零）。"""
        from nanovllm.models.qwen35 import GatedDeltaNet
        gdn = self._make_gdn()
        gdn.allocate_states(max_seqs=4)
        cfg = FakeQwen35Config()

        # 先做 prefill 初始化 recurrent_state
        hidden_p = torch.randn(3, cfg.hidden_size)
        cu_q = torch.tensor([0, 3], dtype=torch.int32)
        set_context(is_prefill=True, cu_seqlens_q=cu_q, lin_attn_seq_slots=[0])
        gdn(hidden_p)
        reset_context()

        # 再做 decode
        hidden_d = torch.randn(1, cfg.hidden_size)
        set_context(is_prefill=False, lin_attn_seq_slots=[0])
        gdn(hidden_d)
        reset_context()
        assert not torch.all(gdn.conv_state[0] == 0)

    @pytest.mark.unit
    def test_reset_to_zero_clears_state(self):
        """allocate 后手动 zero_() 能清空状态，模拟 _reset_lin_attn_states。"""
        from nanovllm.models.qwen35 import GatedDeltaNet
        gdn = self._make_gdn()
        gdn.allocate_states(max_seqs=2)
        gdn.recurrent_state.fill_(1.0)
        gdn.recurrent_state.zero_()
        assert torch.all(gdn.recurrent_state == 0)


# ─── Unit: Qwen35Attention ───────────────────────────────────────────────────

class TestQwen35Attention:

    def _make_attn(self):
        import nanovllm.layers.attention as _attn_module
        from nanovllm.engine.model_runner import AttentionWithKVCache
        _attn_module.Attention = AttentionWithKVCache
        from nanovllm.models.qwen35 import Qwen35Attention
        return Qwen35Attention(FakeQwen35Config())

    @pytest.mark.unit
    def test_q_proj_output_width(self):
        """q_proj 输出 2×num_heads×head_dim（含 gate）。"""
        attn = self._make_attn()
        cfg = FakeQwen35Config()
        expected_out = 2 * cfg.num_attention_heads * cfg.head_dim
        assert attn.q_proj.weight.shape[0] == expected_out

    @pytest.mark.unit
    def test_q_k_norm_zeros_init(self):
        """q_norm / k_norm 的 weight 应为零初始化。"""
        attn = self._make_attn()
        assert torch.all(attn.q_norm.weight == 0)
        assert torch.all(attn.k_norm.weight == 0)

    @pytest.mark.unit
    def test_partial_rope_rotary_dim(self):
        """rotary_emb.rotary_dim 应为 head_dim × partial_rotary_factor。"""
        attn = self._make_attn()
        cfg = FakeQwen35Config()
        expected = int(cfg.head_dim * cfg.partial_rotary_factor)
        assert attn.rotary_emb.rotary_dim == expected

    @pytest.mark.unit
    def test_forward_output_shape(self):
        """全注意力层前向：输出 shape = [T, hidden_size]。"""
        attn = self._make_attn()
        cfg = FakeQwen35Config()
        T = 6
        positions = torch.arange(T)
        hidden = torch.randn(T, cfg.hidden_size)
        cu_q = torch.tensor([0, T], dtype=torch.int32)
        set_context(is_prefill=True, cu_seqlens_q=cu_q, lin_attn_seq_slots=[0])
        out = attn(positions, hidden)
        reset_context()
        assert out.shape == (T, cfg.hidden_size)


# ─── Integration: Qwen35ForCausalLM 结构 ─────────────────────────────────────

def _fresh_qwen35():
    get_rope.cache_clear()
    import nanovllm.layers.attention as _attn_module
    from nanovllm.engine.model_runner import AttentionWithKVCache
    _attn_module.Attention = AttentionWithKVCache
    from nanovllm.models.qwen35 import Qwen35ForCausalLM
    return Qwen35ForCausalLM(FakeQwen35Config())


class TestQwen35Structure:

    @pytest.mark.unit
    def test_layer_type_counts(self):
        """3 个 GatedDeltaNet + 1 个 AttentionWithKVCache。"""
        from nanovllm.models.qwen35 import GatedDeltaNet
        from nanovllm.engine.model_runner import AttentionWithKVCache
        model = _fresh_qwen35()
        gdn_count = sum(1 for m in model.modules() if isinstance(m, GatedDeltaNet))
        attn_count = sum(1 for m in model.modules() if isinstance(m, AttentionWithKVCache))
        assert gdn_count == 3
        assert attn_count == 1

    @pytest.mark.unit
    def test_packed_modules_mapping_keys(self):
        """gate/up 投影合并；k/v 独立不应出现在映射表中。"""
        model = _fresh_qwen35()
        mapping = model.packed_modules_mapping
        for key in ("gate_proj", "up_proj"):
            assert key in mapping, f"{key} 缺失于 packed_modules_mapping"
        assert "k_proj" not in mapping, "k_proj 应独立加载，不应合并"
        assert "v_proj" not in mapping, "v_proj 应独立加载，不应合并"

    @pytest.mark.unit
    def test_weight_prefix_strip_attribute(self):
        from nanovllm.models.qwen35 import Qwen35ForCausalLM
        assert Qwen35ForCausalLM.weight_prefix_to_strip == "model."

    @pytest.mark.unit
    def test_weight_skip_prefixes_attribute(self):
        from nanovllm.models.qwen35 import Qwen35ForCausalLM
        assert "model.visual." in Qwen35ForCausalLM.weight_skip_prefixes
        assert "mtp." in Qwen35ForCausalLM.weight_skip_prefixes

    @pytest.mark.unit
    def test_forward_output_shape(self):
        """完整模型前向：hidden shape = [T, hidden_size]。"""
        model = _fresh_qwen35()
        cfg = FakeQwen35Config()
        T = 7
        input_ids = torch.randint(0, cfg.vocab_size, (T,))
        positions = torch.arange(T)
        cu_q = torch.tensor([0, T], dtype=torch.int32)

        # 分配状态
        from nanovllm.models.qwen35 import GatedDeltaNet
        for m in model.modules():
            if isinstance(m, GatedDeltaNet):
                m.allocate_states(max_seqs=4)

        set_context(is_prefill=True, cu_seqlens_q=cu_q, lin_attn_seq_slots=[0])
        hidden = model(input_ids, positions)
        reset_context()
        assert hidden.shape == (T, cfg.hidden_size)

    @pytest.mark.unit
    def test_compute_logits_shape(self):
        """logits shape = [num_seqs, vocab_size]（最后一个 token 每序列）。"""
        model = _fresh_qwen35()
        cfg = FakeQwen35Config()
        # 两个序列，长度 3 和 4
        T = 7
        input_ids = torch.randint(0, cfg.vocab_size, (T,))
        positions = torch.arange(T)
        cu_q = torch.tensor([0, 3, 7], dtype=torch.int32)

        from nanovllm.models.qwen35 import GatedDeltaNet
        for m in model.modules():
            if isinstance(m, GatedDeltaNet):
                m.allocate_states(max_seqs=4)

        set_context(is_prefill=True, cu_seqlens_q=cu_q, lin_attn_seq_slots=[0, 1])
        hidden = model(input_ids, positions)
        logits = model.compute_logits(hidden)
        reset_context()
        assert logits.shape == (2, cfg.vocab_size)

    @pytest.mark.unit
    def test_num_decoder_layers(self):
        model = _fresh_qwen35()
        assert len(model.language_model.layers) == 4


# ─── Integration: loader 前缀剥离 + skip_prefixes ─────────────────────────────

class TestLoaderPrefixAndSkip:

    def _save_safetensors(self, tmpdir, tensors):
        from safetensors.torch import save_file
        path = os.path.join(tmpdir, "model.safetensors")
        save_file(tensors, path)
        return path

    @pytest.mark.unit
    def test_prefix_stripped_correctly(self):
        """weight_prefix_to_strip 剥离后能正确加载参数。"""
        from nanovllm.utils.loader import load_model, default_weight_loader

        class PrefixModel(nn.Module):
            weight_prefix_to_strip = "model.language_model."
            weight_skip_prefixes   = ()
            packed_modules_mapping = {}

            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.zeros(4, 4))
                self.weight.weight_loader = default_weight_loader

        model = PrefixModel()
        w = torch.randn(4, 4)
        with tempfile.TemporaryDirectory() as tmpdir:
            self._save_safetensors(tmpdir, {"model.language_model.weight": w})
            load_model(model, tmpdir)
        assert torch.allclose(model.weight.data, w)

    @pytest.mark.unit
    def test_skip_prefixes_ignored(self):
        """weight_skip_prefixes 匹配的权重应被完全跳过，不影响其他参数。"""
        from nanovllm.utils.loader import load_model, default_weight_loader

        class SkipModel(nn.Module):
            weight_prefix_to_strip = ""
            weight_skip_prefixes   = ("model.visual.", "mtp.")
            packed_modules_mapping = {}

            def __init__(self):
                super().__init__()
                self.lm = nn.Parameter(torch.zeros(2, 2))
                self.lm.weight_loader = default_weight_loader

        model = SkipModel()
        lm_w = torch.ones(2, 2)
        vis_w = torch.full((3, 3), 99.0)
        mtp_w = torch.full((4, 4), 88.0)
        with tempfile.TemporaryDirectory() as tmpdir:
            self._save_safetensors(tmpdir, {
                "lm": lm_w,
                "model.visual.encoder": vis_w,
                "mtp.head": mtp_w,
            })
            load_model(model, tmpdir)
        assert torch.allclose(model.lm.data, lm_w), "lm 参数应被加载"

    @pytest.mark.unit
    def test_prefix_strip_and_skip_combined(self):
        """前缀剥离与跳过同时生效：跳过先于剥离判断。"""
        from nanovllm.utils.loader import load_model, default_weight_loader

        class CombinedModel(nn.Module):
            weight_prefix_to_strip = "model.language_model."
            weight_skip_prefixes   = ("model.visual.",)
            packed_modules_mapping = {}

            def __init__(self):
                super().__init__()
                self.fc = nn.Parameter(torch.zeros(2, 2))
                self.fc.weight_loader = default_weight_loader

        model = CombinedModel()
        fc_w = torch.randn(2, 2)
        with tempfile.TemporaryDirectory() as tmpdir:
            self._save_safetensors(tmpdir, {
                "model.language_model.fc": fc_w,
                "model.visual.patch_embed": torch.randn(5, 5),
            })
            load_model(model, tmpdir)
        assert torch.allclose(model.fc.data, fc_w)


# ─── Integration: Scheduler slot 池（混合模型） ───────────────────────────────

class TestSchedulerLinAttnSlots:

    def _make_hybrid_sched(self, num_slots=4):
        from nanovllm.engine.scheduler import Scheduler
        from nanovllm.engine.sequence import Sequence
        Sequence.block_size = 4
        return Scheduler(num_kvcache_blocks=50, block_size=4,
                         max_num_seqs=num_slots, num_lin_attn_slots=num_slots, eos=999)

    @pytest.mark.unit
    def test_slot_assigned_on_prefill(self):
        """prefill 调度后序列应获得有效 slot（>= 0）。"""
        from nanovllm.engine.sequence import Sequence
        from nanovllm.sampling_params import SamplingParams
        sched = self._make_hybrid_sched(num_slots=4)
        seq = Sequence([1, 2, 3], SamplingParams(max_tokens=5))
        sched.add(seq)
        seqs, is_prefill = sched.schedule()
        assert is_prefill
        assert seq in seqs
        assert seq.lin_attn_slot >= 0

    @pytest.mark.unit
    def test_slot_returned_on_finish(self):
        """序列完成后 slot 归还到空闲池。"""
        from nanovllm.engine.sequence import Sequence
        from nanovllm.sampling_params import SamplingParams
        sched = self._make_hybrid_sched(num_slots=4)
        seq = Sequence([1, 2, 3], SamplingParams(max_tokens=1))
        sched.add(seq)
        sched.schedule()
        slot = seq.lin_attn_slot
        assert slot not in sched.free_lin_attn_slots
        sched.postprocess([seq], [999], is_prefill=False)  # eos=999
        assert seq.lin_attn_slot == -1
        assert slot in sched.free_lin_attn_slots

    @pytest.mark.unit
    def test_slots_exhaustion_stops_prefill(self):
        """slot 池耗尽时不再调度新的 prefill（等待已有序列完成后归还）。"""
        from nanovllm.engine.sequence import Sequence
        from nanovllm.sampling_params import SamplingParams
        sched = self._make_hybrid_sched(num_slots=2)
        for _ in range(4):
            sched.add(Sequence([1, 2], SamplingParams(max_tokens=5)))
        # 第 1 次 schedule：调度 2 个（slot 池满）
        seqs1, _ = sched.schedule()
        assert len(seqs1) == 2
        assert len(sched.free_lin_attn_slots) == 0
        # 第 2 次 schedule（waiting 还有 2 个）：slot 池空 → 转 decode
        seqs2, is_prefill2 = sched.schedule()
        assert not is_prefill2           # 应进入 decode 而非 prefill

    @pytest.mark.unit
    def test_dense_model_no_slot_interference(self):
        """纯 Dense 模型（num_lin_attn_slots=0）schedule 不受 slot 逻辑影响。"""
        from nanovllm.engine.scheduler import Scheduler
        from nanovllm.engine.sequence import Sequence
        from nanovllm.sampling_params import SamplingParams
        Sequence.block_size = 4
        sched = Scheduler(num_kvcache_blocks=50, block_size=4,
                          max_num_seqs=4, num_lin_attn_slots=0, eos=999)
        seq = Sequence([1, 2, 3], SamplingParams(max_tokens=5))
        sched.add(seq)
        seqs, is_prefill = sched.schedule()
        assert is_prefill
        assert seq in seqs
        assert seq.lin_attn_slot == -1   # Dense 模型不分配 slot


# ─── GPU E2E 测试（需真实 Qwen3.5-2B 权重） ───────────────────────────────────

MODEL_PATH = os.environ.get("NANO_VLLM_QWEN35_MODEL", "")


@pytest.mark.gpu
@pytest.mark.skipif(
    not MODEL_PATH or not os.path.isdir(MODEL_PATH),
    reason="需设置 NANO_VLLM_QWEN35_MODEL 环境变量指向 Qwen3.5-2B 权重目录",
)
class TestQwen35E2E:

    @pytest.fixture(scope="class")
    def engine(self):
        from nanovllm.engine.llm_engine import LLMEngine
        return LLMEngine(MODEL_PATH, enforce_eager=True,
                         max_model_len=512, max_num_seqs=8)

    def test_kv_cache_allocated(self, engine):
        assert engine.model_runner.config.num_kvcache_blocks > 0

    def test_kv_cache_only_6_layers(self, engine):
        """KV cache 应只为 6 层全注意力层分配。"""
        kv = engine.model_runner.kv_cache
        assert kv.shape[1] == 6, f"期望 6 层 KV cache，实际 {kv.shape[1]}"

    def test_lin_attn_states_allocated(self, engine):
        """GDN 状态应已分配。"""
        from nanovllm.models.qwen35 import GatedDeltaNet
        for m in engine.model_runner.model.modules():
            if isinstance(m, GatedDeltaNet):
                assert m.conv_state.numel() > 0
                assert m.recurrent_state.numel() > 0
                break

    def test_single_generation(self, engine):
        from nanovllm.sampling_params import SamplingParams
        results = engine.generate(["你好"], SamplingParams(max_tokens=10), use_tqdm=False)
        assert len(results) == 1
        assert len(results[0]["token_ids"]) <= 10
        assert results[0]["text"]

    def test_concurrent_requests_state_isolation(self, engine):
        """并发 2 路请求的输出应与单独推理一致（状态不串）。"""
        from nanovllm.sampling_params import SamplingParams
        sp = SamplingParams(max_tokens=20, temperature=0.0, ignore_eos=True)
        prompt = "The capital of France is"

        # 单独推理两次
        r1 = engine.generate([prompt], sp, use_tqdm=False)[0]["token_ids"]
        r2 = engine.generate([prompt], sp, use_tqdm=False)[0]["token_ids"]

        # 并发推理
        r_batch = engine.generate([prompt, prompt], sp, use_tqdm=False)

        assert r_batch[0]["token_ids"] == r1, "并发 slot 0 输出与单独推理不一致"
        assert r_batch[1]["token_ids"] == r2, "并发 slot 1 输出与单独推理不一致"

    def test_greedy_alignment_with_hf(self, engine):
        """greedy decoding 前 20 token 应与 HuggingFace 参考实现一致。"""
        pytest.importorskip("transformers")
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from nanovllm.sampling_params import SamplingParams

        prompt = "Once upon a time"
        sp = SamplingParams(max_tokens=20, temperature=0.0, ignore_eos=True)

        # nano-vllm 推理
        nano_result = engine.generate([prompt], sp, use_tqdm=False)[0]["token_ids"]

        # HuggingFace 参考
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        hf_model  = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cuda"
        )
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            hf_out = hf_model.generate(
                **inputs, max_new_tokens=20, do_sample=False,
                temperature=1.0, repetition_penalty=1.0,
            )
        hf_tokens = hf_out[0][inputs["input_ids"].shape[1]:].tolist()

        assert nano_result == hf_tokens, (
            f"nano-vllm 输出与 HF 不一致\n"
            f"  nano: {nano_result}\n"
            f"  HF:   {hf_tokens}"
        )
        del hf_model
        torch.cuda.empty_cache()
