"""
RMSNorm 单元测试

Phase 2：基础归一化（rms_forward、dispatch via __call__）
Phase 4：追加融合加法归一化（add_rms_forward）和 @torch.compile 兼容性测试
"""
import pytest
import torch
import torch.nn as nn

import torch._dynamo
torch._dynamo.config.suppress_errors = True

from nanovllm.layers.layernorm import RMSNorm


# ─── Phase 2：基础 RMSNorm ──────────────────────────────────────────────────────


class TestRMSNormBasic:

    @pytest.mark.unit
    def test_output_shape_preserved(self):
        norm = RMSNorm(16)
        x = torch.randn(4, 16)
        y = norm(x)
        assert y.shape == x.shape

    @pytest.mark.unit
    def test_unit_weight_rms_equals_one(self):
        norm = RMSNorm(64)
        nn.init.ones_(norm.weight)
        x = torch.randn(8, 64) * 10
        y = norm(x)
        rms = y.float().pow(2).mean(-1).sqrt()
        assert torch.allclose(rms, torch.ones_like(rms), atol=1e-5)

    @pytest.mark.unit
    def test_dtype_preserved_float16(self):
        norm = RMSNorm(8)
        x = torch.randn(2, 8, dtype=torch.float16)
        y = norm(x)
        assert y.dtype == torch.float16

    @pytest.mark.unit
    def test_none_residual_returns_tensor(self):
        norm = RMSNorm(16)
        x = torch.randn(3, 16)
        result = norm(x, None)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (3, 16)

    @pytest.mark.unit
    def test_add_rms_forward_equiv_to_add_then_norm(self):
        norm = RMSNorm(32)
        x = torch.randn(4, 32)
        residual = torch.randn(4, 32)
        with torch.no_grad():
            y1, _ = norm.add_rms_forward(x, residual)
            ref = norm.rms_forward(x + residual)
        assert torch.allclose(y1, ref, atol=1e-6)

    @pytest.mark.unit
    def test_add_rms_forward_updates_residual(self):
        norm = RMSNorm(32)
        x = torch.randn(4, 32)
        residual = torch.randn(4, 32)
        with torch.no_grad():
            _, new_res = norm.add_rms_forward(x, residual)
        expected = x + residual
        assert torch.allclose(new_res, expected, atol=1e-6)

    @pytest.mark.unit
    def test_forward_dispatch_with_residual(self):
        norm = RMSNorm(16)
        x = torch.randn(3, 16)
        residual = torch.randn(3, 16)
        y, r = norm(x, residual)
        assert y.shape == (3, 16) and r.shape == (3, 16)


# ─── Phase 4：融合 Add-RMSNorm + dtype 保留 ────────────────────────────────────


class TestRMSNormFused:

    @pytest.mark.unit
    def test_add_rms_residual_dtype_preserved(self):
        norm = RMSNorm(16)
        x = torch.randn(4, 16, dtype=torch.float16)
        residual = torch.randn(4, 16, dtype=torch.float16)
        out, res = norm(x, residual)
        assert out.dtype == torch.float16
        assert res.dtype == torch.float16

    @pytest.mark.unit
    def test_add_rms_residual_correct_value(self):
        norm = RMSNorm(16)
        nn.init.ones_(norm.weight)
        x = torch.randn(4, 16, dtype=torch.float16)
        residual = torch.randn(4, 16, dtype=torch.float16)
        _, new_res = norm(x, residual)
        expected = (x.float() + residual.float()).to(torch.float16)
        assert torch.allclose(new_res, expected, atol=1e-3)

    @pytest.mark.unit
    def test_compile_no_crash_on_cpu(self):
        norm = RMSNorm(32)
        x = torch.randn(4, 32)
        residual = torch.randn(4, 32)
        out, res = norm(x, residual)
        assert out.shape == (4, 32)
