import os
import torch
from dataclasses import dataclass, field


@dataclass
class Config:
    """
    全局推理配置。

    字段说明：
      model                  — 模型权重目录路径（safetensors 格式）
      max_num_batched_tokens — 单步最多处理的 token 总数
      max_num_seqs           — 单步最多并发序列数
      max_model_len          — 支持的最大序列长度
      gpu_memory_utilization — GPU 显存用于 KV cache 的比例
      tensor_parallel_size   — 张量并行 GPU 数量
      enforce_eager          — 禁用 CUDA graph（调试用）
      kvcache_block_size     — 每个 KV cache 物理块包含的 token 数（256 的倍数）
      num_kvcache_blocks     — KV cache 物理块总数（运行时由 ModelRunner 填入）
      hf_config              — transformers AutoConfig 对象（运行时加载）
      eos                    — EOS token id（由 LLMEngine 从 tokenizer 填入）
    """
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    num_lin_attn_slots: int = 0   # 运行时由 ModelRunner 填入（混合模型）
    hf_config: object = field(default=None, repr=False)
    eos: int = -1

    def __post_init__(self):
        assert os.path.isdir(self.model), f"模型路径不存在: {self.model}"
        assert self.kvcache_block_size % 256 == 0, "kvcache_block_size 必须是 256 的倍数"
        assert 1 <= self.tensor_parallel_size <= 8
        assert 0.0 < self.gpu_memory_utilization <= 1.0

        # 延迟导入 transformers，避免在纯 Python 测试中不必要的依赖
        try:
            from transformers import AutoConfig
            try:
                hf = AutoConfig.from_pretrained(self.model)
            except (ValueError, KeyError):
                # transformers 版本过旧不认识该 model_type，直接解析 config.json
                import json
                from types import SimpleNamespace
                with open(os.path.join(self.model, 'config.json')) as f:
                    cfg = json.load(f)
                # VLM 包装：顶层 model_type='qwen3_5'/'qwen3_5_moe'，语言骨干在 text_config 下
                if cfg.get('model_type') in ('qwen3_5', 'qwen3_5_moe') and 'text_config' in cfg:
                    cfg = cfg['text_config']
                # 展平 rope_parameters（含 partial_rotary_factor / rope_theta）
                rope_params = cfg.pop('rope_parameters', {})
                cfg.update({k: v for k, v in rope_params.items() if k not in cfg})
                hf = SimpleNamespace(**cfg)

            # AutoConfig 路径：处理 VLM 包装层
            if getattr(hf, 'model_type', '') in ('qwen3_5', 'qwen3_5_moe'):
                hf = hf.text_config

            # dtype 字符串 → torch.dtype（兼容 torch_dtype 和 dtype 字段）
            raw_dtype = getattr(hf, 'torch_dtype', None) or getattr(hf, 'dtype', None)
            if isinstance(raw_dtype, str):
                resolved = getattr(torch, raw_dtype, torch.bfloat16)
            else:
                resolved = raw_dtype if raw_dtype is not None else torch.bfloat16
            hf.torch_dtype = resolved
            hf.dtype = resolved

            self.hf_config = hf
            self.max_model_len = min(self.max_model_len, hf.max_position_embeddings)
        except Exception:
            pass
