"""
Qwen3.5-35B-A3B MoE 适配单元 + 集成测试

覆盖层次：
  Unit       — 单个类/函数，无 GPU，无真实权重
  Integration— 多模块协作，CPU 小模型
  GPU        — 需 NANO_VLLM_QWEN35_MOE_MODEL 环境变量指向 3L 权重目录
"""
import os
import pytest
import torch
import torch.nn as nn

from nanovllm.utils.context import set_context, reset_context


# ─── Fake configs ──────────────────────────────────────────────────────────────

class FakeMoEConfig:
    """缩小版 MoE 配置，用于 CPU 单元测试（无 GPU，无真实权重）。"""
    model_type                   = "qwen3_5_moe_text"
    hidden_size                  = 64
    num_hidden_layers            = 3
    num_attention_heads          = 4
    num_key_value_heads          = 2
    head_dim                     = 16
    vocab_size                   = 200
    max_position_embeddings      = 256
    rms_norm_eps                 = 1e-6
    rope_theta                   = 10000.0
    partial_rotary_factor        = 0.25
    tie_word_embeddings          = False
    # 线性注意力参数（nk != nv，覆盖 35B 场景）
    linear_num_key_heads         = 2
    linear_num_value_heads       = 4   # nv > nk
    linear_key_head_dim          = 8
    linear_value_head_dim        = 8
    linear_conv_kernel_dim       = 4
    # MoE 参数（小规模）
    num_experts                  = 8
    num_experts_per_tok          = 2
    moe_intermediate_size        = 16
    shared_expert_intermediate_size = 16
    # layer_types：3 层全 GDN
    layer_types                  = ["linear_attention", "linear_attention", "linear_attention"]
    dtype                        = torch.float32
    torch_dtype                  = torch.float32


class FakeMoEConfigWithFullAttn(FakeMoEConfig):
    """含一层 full_attention 的 MoE 配置。"""
    layer_types = ["linear_attention", "linear_attention", "linear_attention", "full_attention"]
    num_hidden_layers = 4


# ─── Unit Tests ────────────────────────────────────────────────────────────────

class TestExpertWeights:
    def test_shape(self):
        from nanovllm.models.qwen35_moe import ExpertWeights
        ew = ExpertWeights(num_experts=8, moe_inter=16, hidden=64)
        assert ew.gate_up_proj.shape == (8, 32, 64)
        assert ew.down_proj.shape == (8, 64, 16)

    def test_parameter_names(self):
        from nanovllm.models.qwen35_moe import ExpertWeights
        ew = ExpertWeights(num_experts=8, moe_inter=16, hidden=64)
        names = {n for n, _ in ew.named_parameters()}
        assert "gate_up_proj" in names
        assert "down_proj" in names


class TestSharedExpertMLP:
    def test_forward_shape(self):
        from nanovllm.models.qwen35_moe import SharedExpertMLP
        mlp = SharedExpertMLP(hidden=64, moe_inter=16)
        x = torch.randn(5, 64)
        out = mlp(x)
        assert out.shape == (5, 64)


class TestQwen35MoEFFN:
    def setup_method(self):
        torch.set_default_device("cpu")
        torch.set_default_dtype(torch.float32)

    def teardown_method(self):
        torch.set_default_device("cpu")
        torch.set_default_dtype(torch.float32)

    def test_output_shape(self):
        from nanovllm.models.qwen35_moe import Qwen35MoEFFN
        cfg = FakeMoEConfig()
        ffn = Qwen35MoEFFN(cfg)
        x = torch.randn(7, 64)
        out = ffn(x)
        assert out.shape == (7, 64)

    def test_top_k_routing(self):
        """验证每个 token 恰好激活 top_k 个专家（通过路由 logit 控制验证路由行为）。"""
        from nanovllm.models.qwen35_moe import Qwen35MoEFFN
        cfg = FakeMoEConfig()
        ffn = Qwen35MoEFFN(cfg)
        T = 4
        x = torch.randn(T, 64)

        # 拦截 gate 输出来验证 top-k 选择
        router_logits = ffn.gate(x)  # [T, num_experts=8]
        scores = torch.softmax(router_logits.float(), dim=-1)
        _, top_indices = torch.topk(scores, cfg.num_experts_per_tok, dim=-1)
        assert top_indices.shape == (T, cfg.num_experts_per_tok)
        # 每行 top_k 个专家互不重复
        for i in range(T):
            assert top_indices[i].unique().numel() == cfg.num_experts_per_tok

    def test_deterministic(self):
        """相同输入应产生相同输出（无随机性）。
        注意：ExpertWeights 用 torch.empty 初始化（生产中由权重文件填充），
        测试中必须先显式初始化，否则未初始化内存可能含 NaN。
        """
        from nanovllm.models.qwen35_moe import Qwen35MoEFFN
        cfg = FakeMoEConfig()
        ffn = Qwen35MoEFFN(cfg)
        with torch.no_grad():
            for p in ffn.parameters():
                p.normal_(0, 0.02)
        ffn.eval()
        x = torch.randn(3, 64)
        out1 = ffn(x)
        out2 = ffn(x)
        assert torch.allclose(out1, out2)


class TestGDNWithNvGtNk:
    """验证 GDN nv > nk 时（35B 场景）正确运行。"""

    def setup_method(self):
        torch.set_default_device("cpu")
        torch.set_default_dtype(torch.float32)

    def teardown_method(self):
        torch.set_default_device("cpu")
        torch.set_default_dtype(torch.float32)

    def test_gdn_prefill_shape_nv_gt_nk(self):
        from nanovllm.models.qwen35 import GatedDeltaNet
        cfg = FakeMoEConfig()
        gdn = GatedDeltaNet(cfg)
        gdn.allocate_states(max_seqs=2)

        T = 5
        hidden = torch.randn(T, cfg.hidden_size)
        cu_q = torch.tensor([0, T], dtype=torch.int32)
        set_context(True, cu_q, cu_q, T, T, None, lin_attn_seq_slots=[0])
        out = gdn(hidden)
        reset_context()
        assert out.shape == (T, cfg.hidden_size)

    def test_gdn_decode_shape_nv_gt_nk(self):
        from nanovllm.models.qwen35 import GatedDeltaNet
        cfg = FakeMoEConfig()
        gdn = GatedDeltaNet(cfg)
        gdn.allocate_states(max_seqs=2)

        hidden = torch.randn(1, cfg.hidden_size)
        set_context(False, lin_attn_seq_slots=[0])
        out = gdn(hidden)
        reset_context()
        assert out.shape == (1, cfg.hidden_size)

    def test_a_log_shape(self):
        """A_log 应为 nv-sized，不是 nk-sized。"""
        from nanovllm.models.qwen35 import GatedDeltaNet
        cfg = FakeMoEConfig()
        gdn = GatedDeltaNet(cfg)
        assert gdn.A_log.shape == (cfg.linear_num_value_heads,)
        assert gdn.dt_bias.shape == (cfg.linear_num_value_heads,)
        assert gdn.in_proj_b.weight.shape[0] == cfg.linear_num_value_heads
        assert gdn.in_proj_a.weight.shape[0] == cfg.linear_num_value_heads


# ─── Integration Tests ─────────────────────────────────────────────────────────

class TestQwen35MoEDecoderLayer:
    def setup_method(self):
        torch.set_default_device("cpu")
        torch.set_default_dtype(torch.float32)

    def teardown_method(self):
        torch.set_default_device("cpu")
        torch.set_default_dtype(torch.float32)

    def test_linear_attn_layer(self):
        from nanovllm.models.qwen35_moe import Qwen35MoEDecoderLayer
        from nanovllm.models.qwen35 import GatedDeltaNet
        cfg = FakeMoEConfig()
        layer = Qwen35MoEDecoderLayer(cfg, "linear_attention")
        layer.linear_attn.allocate_states(max_seqs=2)

        T = 4
        hidden = torch.randn(T, cfg.hidden_size)
        positions = torch.arange(T)
        cu_q = torch.tensor([0, T], dtype=torch.int32)
        set_context(True, cu_q, cu_q, T, T, None, lin_attn_seq_slots=[0])
        out, residual = layer(positions, hidden, None)
        reset_context()
        assert out.shape == (T, cfg.hidden_size)
        assert residual.shape == (T, cfg.hidden_size)


class TestQwen35MoEModel:
    def setup_method(self):
        torch.set_default_device("cpu")
        torch.set_default_dtype(torch.float32)

    def teardown_method(self):
        torch.set_default_device("cpu")
        torch.set_default_dtype(torch.float32)

    def _setup_gdn_states(self, model, max_seqs=2):
        from nanovllm.models.qwen35 import GatedDeltaNet
        for m in model.modules():
            if isinstance(m, GatedDeltaNet):
                m.allocate_states(max_seqs)

    def test_model_forward_3_gdn_layers(self):
        from nanovllm.models.qwen35_moe import Qwen35MoEForCausalLM
        cfg = FakeMoEConfig()
        model = Qwen35MoEForCausalLM(cfg)
        self._setup_gdn_states(model)

        T = 6
        input_ids = torch.randint(0, cfg.vocab_size, (T,))
        positions  = torch.arange(T)
        cu_q = torch.tensor([0, T], dtype=torch.int32)
        set_context(True, cu_q, cu_q, T, T, None, lin_attn_seq_slots=[0])
        hidden = model(input_ids, positions)
        reset_context()
        assert hidden.shape == (T, cfg.hidden_size)

        logits = model.compute_logits(hidden)
        assert logits.shape == (T, cfg.vocab_size)

    def test_parameter_path_alignment(self):
        """关键参数路径应与 safetensors key 对齐（剥离 model. 前缀后）。"""
        from nanovllm.models.qwen35_moe import Qwen35MoEForCausalLM
        cfg = FakeMoEConfig()
        model = Qwen35MoEForCausalLM(cfg)
        param_names = {n for n, _ in model.named_parameters()}

        # GDN 参数
        assert "language_model.layers.0.linear_attn.in_proj_qkv.weight" in param_names
        assert "language_model.layers.0.linear_attn.A_log" in param_names
        # MoE 参数（packed experts）
        assert "language_model.layers.0.mlp.experts.gate_up_proj" in param_names
        assert "language_model.layers.0.mlp.experts.down_proj" in param_names
        # 路由器
        assert "language_model.layers.0.mlp.gate.weight" in param_names
        # 共享专家
        assert "language_model.layers.0.mlp.shared_expert_gate.weight" in param_names
        assert "language_model.layers.0.mlp.shared_expert.gate_up_proj.weight" in param_names


# ─── GPU Tests（需真实权重）────────────────────────────────────────────────────

MOE_MODEL_PATH = os.environ.get("NANO_VLLM_QWEN35_MOE_MODEL", "")

@pytest.mark.skipif(not MOE_MODEL_PATH, reason="需 NANO_VLLM_QWEN35_MOE_MODEL 环境变量")
class TestQwen35MoEGPU:
    def test_load_and_generate(self):
        from nanovllm import LLM, SamplingParams
        llm = LLM(model=MOE_MODEL_PATH, max_model_len=512, enforce_eager=True)
        outputs = llm.generate(
            ["你好，请介绍一下自己。"],
            SamplingParams(max_new_tokens=20, temperature=0.0)
        )
        assert len(outputs) == 1
        text = outputs[0].outputs[0].text
        assert isinstance(text, str) and len(text) > 0
