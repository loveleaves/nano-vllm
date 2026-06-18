"""
Worker：单 rank 的生命周期与 RPC 驱动（对齐 vLLM V1 的 Worker/ModelRunner 分层）。

职责分离：
  Worker        — 进程编排：持有 ShmTransport，rank>0 跑 loop()，rank0 broadcast 后本地执行
  ModelRunner   — 纯 GPU 执行器（前向 + 采样），不含任何进程间通信

TP=1 时 transport 为 None，call() 直接本地执行（零通信开销）。
"""
import torch.distributed as dist

from nanovllm.config import Config
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.rpc import ShmTransport


class Worker:

    def __init__(self, config: Config, rank: int = 0, events=None):
        self.rank = rank
        self.world_size = config.tensor_parallel_size
        self.model_runner = ModelRunner(config, rank)
        self.transport: ShmTransport | None = None

        if self.world_size > 1:
            # barrier 保证 rank0 先创建共享内存段，rank>0 再打开
            if rank == 0:
                self.transport = ShmTransport(rank, events, create=True)
                dist.barrier()
            else:
                dist.barrier()
                self.transport = ShmTransport(rank, events, create=False)
                self.loop()  # rank>0 阻塞于此，直到收到 "exit"

    def loop(self):
        """rank>0：循环接收并执行指令。"""
        while True:
            method, seqs = self.transport.recv()
            self.execute(method, seqs)
            if method == "exit":
                break

    def execute(self, method: str, seqs=None):
        """本地分派到 ModelRunner。"""
        if method == "run":
            return self.model_runner.run(seqs)
        if method == "exit":
            return self._exit()
        raise ValueError(f"未知 RPC 方法: {method}")

    def call(self, method: str, seqs=None):
        """rank0 入口：先广播给子进程，再本地执行并返回结果。"""
        if self.transport is not None:
            self.transport.broadcast(method, seqs)
        return self.execute(method, seqs)

    def _exit(self):
        """先关闭传输（带 barrier 协调 unlink），再释放 ModelRunner 资源。"""
        if self.transport is not None:
            self.transport.close()
            dist.barrier()            # 等所有 rank close 完毕
            if self.rank == 0:
                self.transport.unlink()
        self.model_runner.exit()      # del graphs + destroy_process_group
