"""注意力后端选择器（对齐 vLLM V1 的 get_attn_backend）。"""
import os
import torch

from nanovllm.layers.attention.backend import AttentionBackend
from nanovllm.layers.attention.flash_attn import FlashAttentionBackend, HAS_FLASH_ATTN
from nanovllm.layers.attention.torch_sdpa import TorchSDPABackend

_BACKENDS = {
    "flash_attn": FlashAttentionBackend,
    "torch_sdpa": TorchSDPABackend,
}


def get_attn_backend(is_cuda: bool | None = None) -> type[AttentionBackend]:
    """
    选择注意力后端：
      1. 环境变量 NANOVLLM_ATTN_BACKEND={flash_attn,torch_sdpa} 强制覆盖（便于测试）
      2. CUDA（默认设备为 cuda）且 flash_attn 已安装 → FlashAttentionBackend
      3. 否则 → TorchSDPABackend

    后端在 Attention.__init__ 时绑定，故以**当前默认设备**判定（模型在 cuda 上构建 →
    flash；CPU 单测在 cpu 上构建 → sdpa），与旧代码按 q.is_cuda 运行时分发等价。
    """
    forced = os.getenv("NANOVLLM_ATTN_BACKEND")
    if forced:
        if forced not in _BACKENDS:
            raise ValueError(
                f"未知 NANOVLLM_ATTN_BACKEND={forced!r}，可选 {list(_BACKENDS)}")
        return _BACKENDS[forced]
    if is_cuda is None:
        is_cuda = torch.get_default_device().type == "cuda"
    if is_cuda and HAS_FLASH_ATTN:
        return FlashAttentionBackend
    return TorchSDPABackend
