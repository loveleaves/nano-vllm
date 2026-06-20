import os
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
      device                 — 执行设备："cuda"(GPU，默认) / "cpu"(无 GPU 也能跑，对齐 V1
                               CpuPlatform + CPUModelRunner)。cpu 下强制 enforce_eager（无
                               CUDA graph）、仅 TP=1 / UniProc、禁 swap 抢占；模型以 float32
                               运行（CPU SDPA 对 fp16 支持不全）
      cpu_kvcache_gb         — device="cpu" 时为 KV cache 预留的内存（GB）。GPU 路径按剩余显存
                               自动估算块数，CPU 无 mem_get_info，故由此显式给定（对齐 V1
                               VLLM_CPU_KVCACHE_SPACE）
      gpu_memory_utilization — GPU 显存用于 KV cache 的比例（仅 device="cuda"）
      tensor_parallel_size   — 张量并行 GPU 数量
      distributed_executor_backend — 执行器后端："uni"(单进程内联) / "mp"(各 rank 子进程隔离)；
                               None 时按 TP 自动选（TP=1→uni，TP>1→mp）。显式 "mp" 可让
                               TP=1 也走进程隔离（对齐 V1，单卡可测隔离机制）
      enforce_eager          — 禁用 CUDA graph（调试用）
      kvcache_block_size     — 每个 KV cache 物理块包含的 token 数（256 的倍数）
      num_kvcache_blocks     — KV cache 物理块总数（运行时由 ModelRunner 填入）
      scheduling_policy      — waiting 队列排队策略："fcfs" 或 "priority"
      hf_config              — transformers AutoConfig 对象（运行时加载）
      eos                    — EOS token id（由 LLMEngine 从 tokenizer 填入）
    """
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    device: str = "cuda"
    cpu_kvcache_gb: float = 4.0
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    distributed_executor_backend: str | None = None
    enforce_eager: bool = False
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    num_swap_blocks: int = 0   # CPU swap 区块数（>0 时抢占走 swap 而非 recompute；仅 TP=1 内联支持）
    async_scheduling: bool = False   # 异步调度：step N 的 GPU 计算与 step N+1 的 CPU 调度重叠（仅 TP=1 内联，与 swap 互斥）
    multiproc_engine_core: bool = False   # EngineCore 进程化：调度+执行核心跑在独立子进程（busy-loop + mp.Queue），
                                          # 前端（tokenize/detokenize/HTTP）与 GPU 调度解耦；默认 False 走同进程 InprocClient
    speculative_num_tokens: int = 0       # 投机解码深度 k（>0 开启 n-gram 投机；一步多 token verify）；仅 UniProc
    speculative_ngram_max: int = 3        # n-gram proposer 的最大匹配阶
    scheduling_policy: str = "fcfs"
    hf_config: object = field(default=None, repr=False)
    eos: int = -1

    def __post_init__(self):
        assert os.path.isdir(self.model), f"模型路径不存在: {self.model}"
        assert self.kvcache_block_size % 256 == 0, "kvcache_block_size 必须是 256 的倍数"
        assert self.scheduling_policy in ("fcfs", "priority")
        assert self.distributed_executor_backend in (None, "uni", "mp")
        assert 1 <= self.tensor_parallel_size <= 8
        assert 0.0 < self.gpu_memory_utilization <= 1.0
        assert self.device in ("cuda", "cpu")
        if self.device == "cpu":
            # CPU 后端：无 CUDA graph（强制 eager）；TP/进程隔离/swap 抢占均依赖 GPU 语义，限定单进程内联
            self.enforce_eager = True
            assert self.tensor_parallel_size == 1, "device='cpu' 仅支持 TP=1"
            assert self.distributed_executor_backend in (None, "uni"), \
                "device='cpu' 仅支持 UniProc（不支持 mp 进程隔离）"
            assert not self.multiproc_engine_core, "device='cpu' 暂不支持 EngineCore 进程化"
            assert self.num_swap_blocks == 0, "device='cpu' 无 GPU↔CPU swap 概念"
            assert self.cpu_kvcache_gb > 0, "cpu_kvcache_gb 必须 > 0"
        if self.async_scheduling:
            # 异步调度的 token 前向依赖采样 token 留在 GPU + 单进程内联流水，
            # 暂不支持进程隔离（mp）与 swap 抢占（换出会让在飞 token 张量失效）
            assert self.tensor_parallel_size == 1, "async_scheduling 仅支持 TP=1"
            assert self.distributed_executor_backend in (None, "uni"), \
                "async_scheduling 仅支持 UniProc（不支持 mp 进程隔离）"
            assert self.num_swap_blocks == 0, "async_scheduling 与 swap 抢占互斥"
        if self.speculative_num_tokens > 0:
            # 投机解码的 KV 自愈依赖 grammar/Sequence 同进程 + 多位置 verify 留在 rank0，
            # 仅 UniProc；与 async（占位 token 语义冲突）互斥
            assert self.tensor_parallel_size == 1, "投机解码仅支持 TP=1"
            assert self.distributed_executor_backend in (None, "uni"), \
                "投机解码仅支持 UniProc（不支持 mp 进程隔离）"
            assert not self.async_scheduling, "投机解码与 async_scheduling 互斥"
            assert self.speculative_ngram_max >= 1

        # 延迟导入 transformers，避免在纯 Python 测试中不必要的依赖
        try:
            from transformers import AutoConfig
            self.hf_config = AutoConfig.from_pretrained(self.model)
            self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        except Exception:
            pass
