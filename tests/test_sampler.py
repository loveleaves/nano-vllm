"""
Sampler 单元测试

Phase 2：Gumbel-max 采样（形状、范围、温度效应）
Phase 4：追加 @torch.compile 兼容性测试
"""
import pytest
import torch

import torch._dynamo
torch._dynamo.config.suppress_errors = True

from nanovllm.layers.sampler import Sampler


# ─── Phase 2：基础采样行为 ──────────────────────────────────────────────────────


class TestSamplerBasic:

    @pytest.mark.unit
    def test_output_shape(self):
        sampler = Sampler()
        tokens = sampler(torch.randn(4, 1000), torch.ones(4))
        assert tokens.shape == (4,)

    @pytest.mark.unit
    def test_output_dtype_is_int64(self):
        sampler = Sampler()
        tokens = sampler(torch.randn(3, 50), torch.ones(3))
        assert tokens.dtype == torch.int64

    @pytest.mark.unit
    def test_output_in_vocab_range(self):
        sampler = Sampler()
        vocab_size = 500
        tokens = sampler(torch.randn(8, vocab_size), torch.ones(8))
        assert (tokens >= 0).all() and (tokens < vocab_size).all()

    @pytest.mark.unit
    def test_low_temperature_approx_argmax(self):
        sampler = Sampler()
        logits = torch.zeros(5, 20)
        expected = torch.tensor([3, 7, 12, 0, 19])
        for i, idx in enumerate(expected):
            logits[i, idx] = 100.0
        tokens = sampler(logits, torch.full((5,), 1e-6))
        assert torch.equal(tokens, expected)

    @pytest.mark.unit
    def test_temperature_affects_distribution(self):
        torch.manual_seed(0)
        sampler = Sampler()
        logits = torch.randn(1000, 50)
        max_tokens = logits.argmax(-1)
        agree_low = (sampler(logits.clone(), torch.full((1000,), 0.1)) == max_tokens).float().mean()
        agree_high = (sampler(logits.clone(), torch.full((1000,), 10.0)) == max_tokens).float().mean()
        assert agree_low > agree_high

    @pytest.mark.unit
    def test_single_token_vocab(self):
        sampler = Sampler()
        tokens = sampler(torch.zeros(3, 1), torch.ones(3))
        assert (tokens == 0).all()


# ─── Phase 4：@torch.compile 兼容性 ────────────────────────────────────────────


class TestSamplerCompile:

    @pytest.mark.unit
    def test_compile_compatible_on_cpu(self):
        sampler = Sampler()
        logits = torch.randn(2, 30)
        temps = torch.ones(2) * 0.8
        out = sampler(logits, temps)
        assert out.shape == (2,)
        assert out.dtype == torch.int64

    @pytest.mark.unit
    def test_compile_low_temperature_argmax(self):
        sampler = Sampler()
        logits = torch.zeros(3, 10)
        logits[0, 5] = 100.0
        logits[1, 2] = 100.0
        logits[2, 8] = 100.0
        temps = torch.full((3,), 1e-6)
        out = sampler(logits, temps)
        assert torch.equal(out, torch.tensor([5, 2, 8]))
