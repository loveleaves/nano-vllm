"""
AttentionBackend 抽象与 selector 单元测试。
"""
import os
import pytest
import torch

from nanovllm.attention import (
    Attention, get_attn_backend, AttentionBackend, AttentionImpl,
    AttentionMetadataBuilder, FlashAttentionBackend, TorchSDPABackend,
    AttentionBackendEnum, register_backend,
)
from nanovllm.attention.common import CommonAttentionMetadata


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
        assert get_attn_backend(device_type="cpu") is TorchSDPABackend

    @pytest.mark.unit
    def test_cuda_with_flash_selects_flash(self):
        os.environ.pop("NANOVLLM_ATTN_BACKEND", None)
        # flash_attn 已安装的环境下，cuda → flash
        from nanovllm.attention.flash_attn import HAS_FLASH_ATTN
        expected = FlashAttentionBackend if HAS_FLASH_ATTN else TorchSDPABackend
        assert get_attn_backend(device_type="cuda") is expected

    @pytest.mark.unit
    def test_env_override_forces_backend(self):
        try:
            os.environ["NANOVLLM_ATTN_BACKEND"] = "torch_sdpa"
            assert get_attn_backend(device_type="cuda") is TorchSDPABackend
            os.environ["NANOVLLM_ATTN_BACKEND"] = "flash_attn"
            assert get_attn_backend(device_type="cpu") is FlashAttentionBackend
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


class TestCapabilitySelection:
    """按 head_size / dtype 能力筛选（对齐 V1 supports_head_size / supports_dtype）。"""

    @pytest.mark.unit
    def test_flash_capability_queries(self):
        assert FlashAttentionBackend.supports_head_size(128)
        assert not FlashAttentionBackend.supports_head_size(320)   # >256
        assert not FlashAttentionBackend.supports_head_size(100)   # 非 8 倍数
        assert FlashAttentionBackend.supports_dtype(torch.bfloat16)
        assert not FlashAttentionBackend.supports_dtype(torch.float32)
        assert not FlashAttentionBackend.is_available("cpu")

    @pytest.mark.unit
    def test_sdpa_is_permissive(self):
        assert TorchSDPABackend.is_available("cpu") and TorchSDPABackend.is_available("cuda")
        assert TorchSDPABackend.supports_head_size(123)            # 不限
        assert TorchSDPABackend.supports_dtype(torch.float32)

    @pytest.mark.unit
    def test_cuda_unsupported_headsize_falls_back_to_sdpa(self):
        os.environ.pop("NANOVLLM_ATTN_BACKEND", None)
        # cuda 上 head_size=300（flash 不支持）→ 回退 SDPA
        assert get_attn_backend(head_size=300, dtype=torch.bfloat16,
                                device_type="cuda") is TorchSDPABackend

    @pytest.mark.unit
    def test_cuda_fp32_falls_back_to_sdpa(self):
        os.environ.pop("NANOVLLM_ATTN_BACKEND", None)
        from nanovllm.attention.flash_attn import HAS_FLASH_ATTN
        # cuda + fp32（flash 仅 fp16/bf16）→ 回退 SDPA
        assert get_attn_backend(head_size=128, dtype=torch.float32,
                                device_type="cuda") is TorchSDPABackend
        # 对照：bf16 + 合法 head_size → flash（若已安装）
        if HAS_FLASH_ATTN:
            assert get_attn_backend(head_size=128, dtype=torch.bfloat16,
                                    device_type="cuda") is FlashAttentionBackend


class TestRegistry:
    """枚举 + register_backend 动态覆盖（对齐 V1 registry）。"""

    @pytest.mark.unit
    def test_enum_get_class_resolves(self):
        assert AttentionBackendEnum.FLASH_ATTN.get_class() is FlashAttentionBackend
        assert AttentionBackendEnum.TORCH_SDPA.get_class() is TorchSDPABackend

    @pytest.mark.unit
    def test_from_name_case_insensitive_and_invalid(self):
        assert AttentionBackendEnum.from_name("flash_attn") is AttentionBackendEnum.FLASH_ATTN
        assert AttentionBackendEnum.from_name("TORCH_SDPA") is AttentionBackendEnum.TORCH_SDPA
        with pytest.raises(ValueError):
            AttentionBackendEnum.from_name("nope")

    @pytest.mark.unit
    def test_register_backend_override_and_clear(self):
        member = AttentionBackendEnum.FLASH_ATTN
        try:
            register_backend(
                member, "nanovllm.attention.torch_sdpa.TorchSDPABackend")
            assert member.is_overridden()
            assert member.get_class() is TorchSDPABackend   # 覆盖后解析到替身
        finally:
            member.clear_override()
        assert not member.is_overridden()
        assert member.get_class() is FlashAttentionBackend  # 还原默认


class TestAttentionLayerBinding:

    @pytest.mark.unit
    def test_layer_binds_backend_at_init(self):
        # CPU 默认设备 → SDPA impl
        attn = Attention(num_heads=4, head_dim=8, scale=8 ** -0.5, num_kv_heads=2)
        from nanovllm.attention.torch_sdpa import TorchSDPAImpl
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
