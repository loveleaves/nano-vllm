"""
Worker：单 rank 的纯执行包装（对齐 vLLM V1 `v1/worker/gpu_worker.py`）。

只负责"在本 rank 上执行一条指令"，不含任何跨 rank 协调（广播 / barrier / shm）——
那些进程编排逻辑上移到 Executor（见 engine/executor/）。Worker 持有本 rank 的
ModelRunner，execute() 把方法分派到它。

  Worker       — 单 rank 执行器（run / exit）
  ModelRunner  — 纯 GPU 执行器（前向 + 采样），NCCL 进程组在其 __init__ 内建立
"""
from nanovllm.config import Config
from nanovllm.engine.model_runner import ModelRunner


class Worker:

    def __init__(self, config: Config, rank: int = 0):
        self.rank = rank
        self.model_runner = ModelRunner(config, rank)

    def execute(self, method: str, seqs=None, finished_seq_ids=None):
        """在本 rank 上执行一条指令。"""
        if method == "run":
            return self.model_runner.run(seqs, finished_seq_ids)
        if method == "exit":
            return self.model_runner.exit()   # del graphs + destroy_process_group
        raise ValueError(f"未知 RPC 方法: {method}")
