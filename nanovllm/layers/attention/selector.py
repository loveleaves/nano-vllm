"""注意力后端选择器（对齐 vLLM V1 `v1/attention/selector.py::get_attn_backend`）。

按 (平台可用性, head_size, dtype) 在优先级列表中筛选首个满足的后端；
`NANOVLLM_ATTN_BACKEND` 环境变量可显式强制（绕过能力检查，便于测试/调试）。
"""
import os

import torch

from nanovllm.layers.attention.backend import AttentionBackend
from nanovllm.layers.attention.registry import AttentionBackendEnum

# 优先级：flash 优于 sdpa（满足能力时优先选 flash）
_PRIORITY = [AttentionBackendEnum.FLASH_ATTN, AttentionBackendEnum.TORCH_SDPA]


def get_attn_backend(head_size: int | None = None,
                     dtype: torch.dtype | None = None,
                     device_type: str | None = None) -> type[AttentionBackend]:
    """选择注意力后端。

    1. `NANOVLLM_ATTN_BACKEND={flash_attn,torch_sdpa}` → 显式强制（绕过能力筛选）。
    2. 否则按优先级 [flash, sdpa] 选首个满足 (is_available(device) ∧ supports_head_size
       ∧ supports_dtype) 的后端。head_size/dtype 为 None 时跳过对应检查。
    3. 均不满足 → ValueError。

    device_type 未给时由当前默认设备推断。后端在 Attention.__init__ 绑定，故以
    **当前默认设备/dtype**判定，graph 捕获期不再分发。
    """
    forced = os.getenv("NANOVLLM_ATTN_BACKEND")
    if forced:
        return AttentionBackendEnum.from_name(forced).get_class()

    if device_type is None:
        device_type = torch.get_default_device().type

    for member in _PRIORITY:
        backend = member.get_class()
        if not backend.is_available(device_type):
            continue
        if head_size is not None and not backend.supports_head_size(head_size):
            continue
        if dtype is not None and not backend.supports_dtype(dtype):
            continue
        return backend

    raise ValueError(
        f"无可用注意力后端 (device={device_type}, head_size={head_size}, dtype={dtype})")
