"""
Linear 层族单元测试

Phase 2：基础线性层（ReplicatedLinear, ColumnParallel, RowParallel, QKV, Merged）
Phase 4：追加 TP weight_loader 精确切片测试
"""
import pytest
import torch

from nanovllm.layers.linear import (
    ReplicatedLinear, ColumnParallelLinear,
    MergedColumnParallelLinear, QKVParallelLinear, RowParallelLinear,
)


# ─── Phase 2：基础前向计算 ──────────────────────────────────────────────────────


class TestLinearForward:

    @pytest.mark.unit
    def test_replicated_linear_shape(self):
        linear = ReplicatedLinear(16, 32)
        y = linear(torch.randn(4, 16))
        assert y.shape == (4, 32)

    @pytest.mark.unit
    def test_column_parallel_shape(self):
        linear = ColumnParallelLinear(16, 32)
        y = linear(torch.randn(4, 16))
        assert y.shape == (4, 32)   # TP=1，输出完整

    @pytest.mark.unit
    def test_row_parallel_shape(self):
        linear = RowParallelLinear(32, 16)
        y = linear(torch.randn(4, 32))
        assert y.shape == (4, 16)

    @pytest.mark.unit
    def test_row_parallel_no_dist_no_crash(self):
        linear = RowParallelLinear(8, 4)
        y = linear(torch.randn(3, 8))
        assert y.shape == (3, 4)

    @pytest.mark.unit
    def test_merged_column_linear_shape(self):
        linear = MergedColumnParallelLinear(8, [16, 16])
        y = linear(torch.randn(4, 8))
        assert y.shape == (4, 32)

    @pytest.mark.unit
    def test_qkv_parallel_shape(self):
        qkv = QKVParallelLinear(16, 4, total_num_heads=4, total_num_kv_heads=2)
        y = qkv(torch.randn(5, 16))
        # output = (4+2+2)*4 = 32
        assert y.shape[0] == 5

    @pytest.mark.unit
    def test_replicated_weight_loader_copies(self):
        linear = ReplicatedLinear(4, 8)
        w = torch.randn(8, 4)
        linear.weight_loader(linear.weight, w)
        assert torch.allclose(linear.weight.data, w)


# ─── Phase 2：weight_loader 正确性（TP=1，单进程） ───────────────────────────────


class TestLinearWeightLoader:

    @pytest.mark.unit
    def test_qkv_loader_q_writes_to_correct_slice(self):
        num_heads, head_dim, hidden, num_kv_heads = 4, 8, 16, 2
        qkv = QKVParallelLinear(hidden, head_dim, num_heads, num_kv_heads)
        w = torch.arange(num_heads * head_dim * hidden,
                         dtype=torch.float).reshape(num_heads * head_dim, hidden)
        qkv.weight_loader(qkv.weight, w, "q")
        q_size = num_heads * head_dim
        assert torch.allclose(qkv.weight.data[:q_size], w)

    @pytest.mark.unit
    def test_qkv_loader_k_writes_to_correct_slice(self):
        num_heads, head_dim, hidden, num_kv_heads = 4, 8, 16, 2
        qkv = QKVParallelLinear(hidden, head_dim, num_heads, num_kv_heads)
        q_size = num_heads * head_dim
        kv_size = num_kv_heads * head_dim
        w = torch.randn(kv_size, hidden)
        qkv.weight_loader(qkv.weight, w, "k")
        assert torch.allclose(qkv.weight.data[q_size: q_size + kv_size], w)

    @pytest.mark.unit
    def test_qkv_loader_v_writes_to_correct_slice(self):
        num_heads, head_dim, hidden, num_kv_heads = 4, 8, 16, 2
        qkv = QKVParallelLinear(hidden, head_dim, num_heads, num_kv_heads)
        q_size = num_heads * head_dim
        kv_size = num_kv_heads * head_dim
        w = torch.randn(kv_size, hidden)
        qkv.weight_loader(qkv.weight, w, "v")
        assert torch.allclose(qkv.weight.data[q_size + kv_size:], w)

    @pytest.mark.unit
    def test_qkv_loader_all_three_slices_correct(self):
        hidden, head_dim, num_heads, num_kv_heads = 8, 4, 2, 2
        qkv = QKVParallelLinear(hidden, head_dim, num_heads, num_kv_heads)
        q_w = torch.randn(num_heads * head_dim, hidden)
        k_w = torch.randn(num_kv_heads * head_dim, hidden)
        v_w = torch.randn(num_kv_heads * head_dim, hidden)
        qkv.weight_loader(qkv.weight, q_w, "q")
        qkv.weight_loader(qkv.weight, k_w, "k")
        qkv.weight_loader(qkv.weight, v_w, "v")
        q_size = num_heads * head_dim
        kv_size = num_kv_heads * head_dim
        assert torch.allclose(qkv.weight.data[:q_size], q_w)
        assert torch.allclose(qkv.weight.data[q_size: q_size + kv_size], k_w)
        assert torch.allclose(qkv.weight.data[q_size + kv_size:], v_w)

    @pytest.mark.unit
    def test_qkv_loader_invalid_shard_raises(self):
        qkv = QKVParallelLinear(8, 4, 2, 2)
        with pytest.raises(ValueError):
            qkv.weight_loader(qkv.weight, torch.randn(8, 8), "z")

    @pytest.mark.unit
    def test_merged_column_loader_gate_up(self):
        sizes = [8, 8]
        linear = MergedColumnParallelLinear(4, sizes)
        w0 = torch.ones(8, 4)
        w1 = torch.zeros(8, 4) + 2
        linear.weight_loader(linear.weight, w0, 0)
        linear.weight_loader(linear.weight, w1, 1)
        assert torch.allclose(linear.weight.data[:8], w0)
        assert torch.allclose(linear.weight.data[8:16], w1)
