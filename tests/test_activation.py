"""
激活函数单元测试（SiluAndMul / SwiGLU）
"""
import pytest
import torch
import torch.nn.functional as F

from nanovllm.layers.activation import SiluAndMul


class TestSiluAndMul:

    @pytest.mark.unit
    def test_output_half_the_input_dim(self):
        act = SiluAndMul()
        x = torch.randn(8, 64)
        y = act(x)
        assert y.shape == (8, 32)

    @pytest.mark.unit
    def test_correctness_known_input(self):
        act = SiluAndMul()
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        y = act(x)
        gate = torch.tensor([[1.0, 2.0]])
        up = torch.tensor([[3.0, 4.0]])
        expected = F.silu(gate) * up
        assert torch.allclose(y, expected, atol=1e-6)

    @pytest.mark.unit
    def test_dtype_preserved(self):
        act = SiluAndMul()
        x = torch.randn(4, 8, dtype=torch.float32)
        assert act(x).dtype == torch.float32

    @pytest.mark.unit
    def test_batch_independence(self):
        act = SiluAndMul()
        x = torch.randn(4, 16)
        y = act(x)
        for i in range(4):
            y_single = act(x[i:i+1])
            assert torch.allclose(y[i:i+1], y_single, atol=1e-6)
