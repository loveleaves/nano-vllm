"""
VocabEmbedding / LMHead 单元测试（Phase 2：基础 Embedding 和 LMHead）
"""
import pytest
import torch
import torch.nn as nn

from nanovllm.utils.context import AttentionMetadata
from nanovllm.layers.embed_head import VocabEmbedding, LMHead


def _md(cu_q):
    cu_q = torch.as_tensor(cu_q, dtype=torch.int32)
    return AttentionMetadata(query_start_loc=cu_q, cu_seqlens_k=cu_q.clone())


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
    def test_no_metadata_full_output(self):
        # attn_md=None → 不做选取，输出全部行（等价旧 decode 全输出）
        head = LMHead(50, 16)
        nn.init.normal_(head.weight)
        y = head(torch.randn(7, 16), None)
        assert y.shape == (7, 50)

    @pytest.mark.unit
    def test_decode_metadata_is_identity(self):
        # decode：query_start_loc=[0,1,2] → last_indices=[0,1] 选取退化为恒等
        head = LMHead(50, 16)
        nn.init.normal_(head.weight)
        x = torch.randn(2, 16)
        y = head(x, _md([0, 1, 2]))
        assert y.shape == (2, 50)
        assert torch.allclose(y, torch.nn.functional.linear(x, head.weight), atol=1e-5)

    @pytest.mark.unit
    def test_prefill_mode_extracts_last_tokens(self):
        head = LMHead(50, 16)
        nn.init.normal_(head.weight)
        y = head(torch.randn(7, 16), _md([0, 3, 7]))
        assert y.shape == (2, 50)   # 2 个 seq → 各取最后 1 个 token

    @pytest.mark.unit
    def test_prefill_correct_last_token_indices(self):
        head = LMHead(10, 4)
        nn.init.normal_(head.weight)
        x = torch.randn(5, 4)
        out = head(x, _md([0, 3, 5]))
        # last indices = [2, 4]
        expected_0 = torch.nn.functional.linear(x[2:3], head.weight)
        assert torch.allclose(out[0:1], expected_0, atol=1e-5)


# ─── Phase 4：VocabParallelEmbedding / ParallelLMHead（TP=1 等价验证） ──────────


class TestParallelEmbedHeadPhase4:

    @pytest.mark.unit
    def test_vocab_parallel_embedding_forward_shape(self):
        from nanovllm.layers.embed_head import VocabParallelEmbedding
        emb = VocabParallelEmbedding(100, 16)
        out = emb(torch.tensor([0, 5, 99, 42, 7]))
        assert out.shape == (5, 16)

    @pytest.mark.unit
    def test_vocab_parallel_embedding_equivalence(self):
        from nanovllm.layers.embed_head import VocabParallelEmbedding
        emb = VocabParallelEmbedding(64, 8)
        nn.init.normal_(emb.weight)
        ref = nn.Embedding(64, 8)
        ref.weight.data.copy_(emb.weight.data)
        token_ids = torch.tensor([0, 10, 63])
        assert torch.allclose(emb(token_ids), ref(token_ids))

    @pytest.mark.unit
    def test_parallel_lm_head_decode(self):
        from nanovllm.layers.embed_head import ParallelLMHead
        head = ParallelLMHead(64, 8)
        nn.init.normal_(head.weight)
        out = head(torch.randn(3, 8), None)
        assert out.shape == (3, 64)

    @pytest.mark.unit
    def test_parallel_lm_head_prefill_extracts_last(self):
        from nanovllm.layers.embed_head import ParallelLMHead
        head = ParallelLMHead(64, 8)
        nn.init.normal_(head.weight)
        out = head(torch.randn(5, 8), _md([0, 3, 5]))
        assert out.shape == (2, 64)
