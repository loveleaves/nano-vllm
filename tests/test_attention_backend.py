"""
AttentionBackend 抽象与 selector 单元测试。
"""
import os
import pytest
import torch

from nanovllm.layers.attention import (
    Attention, get_attn_backend, AttentionBackend, AttentionImpl,
    AttentionMetadataBuilder, FlashAttentionBackend, TorchSDPABackend,
)
from nanovllm.layers.attention.common import CommonAttentionMetadata


class TestBackendTriad:

    @pytest.mark.unit
    @pytest.mark.parametrize("backend", [FlashAttentionBackend, TorchSDPABackend])
    def test_backend_triad_types(self, backend):
        assert issubclass(backend, AttentionBackend)
        assert isinstance(backend.get_name(), str)
        assert issubclass(backend.get_impl_cls(), AttentionImpl)
        assert issubclass(backend.get_builder_cls(), AttentionMetadataBuilder)

    @pytest.mark.unit
    def test_kv_cache_shape(self):
        shape = FlashAttentionBackend.get_kv_cache_shape(10, 256, 2, 64)
        assert shape == (2, 10, 256, 2, 64)

    @pytest.mark.unit
    def test_builder_is_identity(self):
        md = CommonAttentionMetadata(max_query_len=1)
        for backend in (FlashAttentionBackend, TorchSDPABackend):
            builder = backend.get_builder_cls()()
            assert builder.build(md) is md


class TestSelector:

    @pytest.mark.unit
    def test_cpu_selects_sdpa(self):
        os.environ.pop("NANOVLLM_ATTN_BACKEND", None)
        assert get_attn_backend(is_cuda=False) is TorchSDPABackend

    @pytest.mark.unit
    def test_cuda_with_flash_selects_flash(self):
        os.environ.pop("NANOVLLM_ATTN_BACKEND", None)
        # flash_attn 已安装的环境下，cuda → flash
        from nanovllm.layers.attention.flash_attn import HAS_FLASH_ATTN
        expected = FlashAttentionBackend if HAS_FLASH_ATTN else TorchSDPABackend
        assert get_attn_backend(is_cuda=True) is expected

    @pytest.mark.unit
    def test_env_override_forces_backend(self):
        try:
            os.environ["NANOVLLM_ATTN_BACKEND"] = "torch_sdpa"
            assert get_attn_backend(is_cuda=True) is TorchSDPABackend
            os.environ["NANOVLLM_ATTN_BACKEND"] = "flash_attn"
            assert get_attn_backend(is_cuda=False) is FlashAttentionBackend
        finally:
            os.environ.pop("NANOVLLM_ATTN_BACKEND", None)

    @pytest.mark.unit
    def test_env_override_invalid_raises(self):
        try:
            os.environ["NANOVLLM_ATTN_BACKEND"] = "nope"
            with pytest.raises(ValueError):
                get_attn_backend()
        finally:
            os.environ.pop("NANOVLLM_ATTN_BACKEND", None)


class TestAttentionLayerBinding:

    @pytest.mark.unit
    def test_layer_binds_backend_at_init(self):
        # CPU 默认设备 → SDPA impl
        attn = Attention(num_heads=4, head_dim=8, scale=8 ** -0.5, num_kv_heads=2)
        from nanovllm.layers.attention.torch_sdpa import TorchSDPAImpl
        assert isinstance(attn.impl, TorchSDPAImpl)

    @pytest.mark.unit
    def test_layer_forward_delegates(self):
        attn = Attention(num_heads=2, head_dim=4, scale=4 ** -0.5, num_kv_heads=2)
        N = 4
        q, k, v = (torch.randn(N, 2, 4) for _ in range(3))
        cu = torch.tensor([0, N], dtype=torch.int32)
        md = CommonAttentionMetadata(query_start_loc=cu, cu_seqlens_k=cu.clone(),
                                     max_query_len=N, max_seq_len=N)
        o = attn(q, k, v, md)
        assert o.shape == (N, 2, 4)
