"""
Qwen3ForCausalLM 结构单元测试（微型 CPU 模型）
"""
import pytest
import torch

from nanovllm.utils.context import set_context, reset_context
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.models.qwen3 import Qwen3ForCausalLM


class FakeQwen3Config:
    hidden_size = 32
    intermediate_size = 64
    num_hidden_layers = 2
    num_attention_heads = 4
    num_key_value_heads = 2
    head_dim = 8
    vocab_size = 100
    max_position_embeddings = 128
    rms_norm_eps = 1e-6
    hidden_act = "silu"
    rope_theta = 10000.0
    rope_scaling = None
    attention_bias = False
    tie_word_embeddings = False


def _fresh_model():
    get_rope.cache_clear()
    return Qwen3ForCausalLM(FakeQwen3Config())


class TestQwen3Structure:

    @pytest.mark.unit
    def test_forward_output_shape(self):
        reset_context()
        model = _fresh_model()
        N = 5
        input_ids = torch.randint(0, 100, (N,))
        positions = torch.arange(N)
        set_context(is_prefill=True, cu_seqlens_q=torch.tensor([0, N], dtype=torch.int32))
        hidden = model(input_ids, positions)
        assert hidden.shape == (N, 32)
        reset_context()

    @pytest.mark.unit
    def test_compute_logits_shape_per_seq(self):
        reset_context()
        model = _fresh_model()
        N, batch = 7, 3
        input_ids = torch.randint(0, 100, (N,))
        positions = torch.arange(N)
        cu_q = torch.tensor([0, 2, 5, 7], dtype=torch.int32)
        set_context(is_prefill=True, cu_seqlens_q=cu_q)
        hidden = model(input_ids, positions)
        logits = model.compute_logits(hidden)
        assert logits.shape == (batch, 100)
        reset_context()

    @pytest.mark.unit
    def test_packed_modules_mapping_keys(self):
        model = _fresh_model()
        mapping = model.packed_modules_mapping
        for key in ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"):
            assert key in mapping

    @pytest.mark.unit
    def test_tie_word_embeddings(self):
        get_rope.cache_clear()
        cfg = FakeQwen3Config()
        cfg.tie_word_embeddings = True
        model = Qwen3ForCausalLM(cfg)
        assert (model.lm_head.weight.data_ptr() ==
                model.model.embed_tokens.weight.data_ptr())

    @pytest.mark.unit
    def test_num_decoder_layers(self):
        model = _fresh_model()
        assert len(model.model.layers) == 2
