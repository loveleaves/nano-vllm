"""
CPU 执行后端（对齐 vLLM V1 `v1/worker/cpu_model_runner.py::CPUModelRunner`）。

设计哲学与 vLLM 一致——**复用 GPU 执行器，仅中和 CUDA 专属操作**：
  - 基类 ModelRunner 已按 `self.device` / `self.is_cuda` 参数化（NCCL / CUDA graph /
    显存估算 / pinned 异步拷贝在 is_cuda=False 时整段跳过，退化为 CPU 同步路径）；
  - 本子类只覆写**唯一真正不同**的一处：可用内存估算——GPU 调 `mem_get_info`，
    CPU 无此接口，改由 `config.cpu_kvcache_gb` 显式给定（对齐 V1
    `CPUWorker.determine_available_memory` 返回 `cpu_kvcache_space_bytes`）。

由此 nano 在无 GPU、无 flash-attn、无 Triton 的机器上也能跑通完整推理：
注意力走 SDPA 后端（selector 按 device_type='cpu' 自动选择），KV 写入走 naive scatter。
"""
import torch

from nanovllm.config import Config
from nanovllm.engine.model_runner import ModelRunner


class CPUModelRunner(ModelRunner):
    """CPU 推理执行器：device='cpu'，无 NCCL / 无 CUDA graph / 内存按配置预留。"""

    def __init__(self, config: Config, rank: int = 0):
        assert config.device == "cpu"
        super().__init__(config, rank)
        assert self.device.type == "cpu"

    def _available_kvcache_bytes(self) -> int:
        """CPU 后端：KV cache 内存由 config.cpu_kvcache_gb 显式给定（无 mem_get_info）。"""
        return int(self.config.cpu_kvcache_gb * (1024 ** 3))
