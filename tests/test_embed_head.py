"""
VocabEmbedding / LMHead 单元测试（Phase 2：基础 Embedding 和 LMHead）
"""
import pytest
import torch
import torch.nn as nn

from nanovllm.utils.context import set_context, reset_context
from nanovllm.layers.embed_head import VocabEmbedding, LMHead


# ─── Phase 2：基础前向计算 ──────────────────────────────────────────────────────


class TestVocabEmbedding:

    @pytest.mark.unit
    def test_output_shape(self):
        emb = VocabEmbedding(100, 16)
        x = torch.randint(0, 100, (5,))
        y = emb(x)
        assert y.shape == (5, 16)

    @pytest.mark.unit
    def test_equivalence_to_nn_embedding(self):
        emb = VocabEmbedding(64, 8)
        nn.init.normal_(emb.weight)
        ref = nn.Embedding(64, 8)
        ref.weight.data.copy_(emb.weight.data)
        token_ids = torch.tensor([0, 10, 63])
        assert torch.allclose(emb(token_ids), ref(token_ids))

    @pytest.mark.unit
    def test_weight_loader_copies_correctly(self):
        emb = VocabEmbedding(32, 8)
        w = torch.randn(32, 8)
        emb.weight_loader(emb.weight, w)
        assert torch.allclose(emb.weight.data, w)


class TestLMHead:

    @pytest.mark.unit
    def test_decode_mode_full_output(self):
        reset_context()
        set_context(is_prefill=False)
        head = LMHead(50, 16)
        nn.init.normal_(head.weight)
        y = head(torch.randn(7, 16))
        assert y.shape == (7, 50)
        reset_context()

    @pytest.mark.unit
    def test_prefill_mode_extracts_last_tokens(self):
        reset_context()
        cu_q = torch.tensor([0, 3, 7], dtype=torch.int32)
        set_context(is_prefill=True, cu_seqlens_q=cu_q)
        head = LMHead(50, 16)
        nn.init.normal_(head.weight)
        y = head(torch.randn(7, 16))
        assert y.shape == (2, 50)   # 2 个 seq → 各取最后 1 个 token
        reset_context()

    @pytest.mark.unit
    def test_prefill_correct_last_token_indices(self):
        reset_context()
        cu_q = torch.tensor([0, 3, 5], dtype=torch.int32)
        set_context(is_prefill=True, cu_seqlens_q=cu_q)
        head = LMHead(10, 4)
        nn.init.normal_(head.weight)
        x = torch.randn(5, 4)
        out = head(x)
        # last indices = [2, 4]
        expected_0 = torch.nn.functional.linear(x[2:3], head.weight)
        assert torch.allclose(out[0:1], expected_0, atol=1e-5)
        reset_context()

