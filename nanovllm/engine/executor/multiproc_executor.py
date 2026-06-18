"""
多进程执行器（对齐 vLLM V1 `v1/executor/multiproc_executor.py::MultiprocExecutor`）。

TP>1：rank1..N 为 spawn 子进程，rank0 跑在本（EngineCore）进程内。rank0 经 ShmTransport
向子进程广播指令、并在本地执行同一步，最后返回 rank0 的采样结果。NCCL all_reduce 在
ModelRunner 内部完成张量同步。

进程编排（从原 Worker.__init__/_exit 上移至此）：
  启动：子进程建 Worker(NCCL init) → barrier 等 rank0 建 shm → 进入收发循环
  退出：广播 "exit" → 各 rank close 传输 → barrier → rank0 unlink → 各 rank 销毁进程组
"""
import torch.distributed as dist
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.engine.executor.abstract import Executor
from nanovllm.engine.rpc import ShmTransport
from nanovllm.engine.worker import Worker


def _worker_proc_main(config: Config, rank: int, event):
    """子进程入口（rank>0）：建 Worker → barrier → 开 shm → 收发循环 → 清理。"""
    worker = Worker(config, rank)             # ModelRunner: NCCL init_process_group
    dist.barrier()                            # 等 rank0 创建共享内存段
    transport = ShmTransport(rank, event, create=False)
    while True:
        method, seqs = transport.recv()
        if method == "exit":
            break
        worker.execute(method, seqs)
    # 清理顺序与 rank0 对齐：close → barrier → 销毁进程组
    transport.close()
    dist.barrier()
    worker.execute("exit")                    # del graphs + destroy_process_group


class MultiProcExecutor(Executor):

    def _init_executor(self) -> None:
        config = self.config
        self.ps: list = []
        self.events: list = []
        ctx = mp.get_context("spawn")
        for rank in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=_worker_proc_main, args=(config, rank, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)

        # rank0 Worker（NCCL init），再建 shm，barrier 通知子进程 shm 就绪
        self.worker = Worker(config, rank=0)
        self.transport = ShmTransport(0, self.events, create=True)
        dist.barrier()

    def collective_rpc(self, method: str, seqs=None) -> list:
        self.transport.broadcast(method, seqs)
        return [self.worker.execute(method, seqs)]

    def execute_model(self, seqs) -> list[int] | None:
        self.transport.broadcast("run", seqs)
        return self.worker.execute("run", seqs)

    def shutdown(self) -> None:
        self.transport.broadcast("exit")      # 通知子进程退出循环
        self.transport.close()
        dist.barrier()                         # 等所有 rank close 完毕
        self.transport.unlink()                # rank0 释放共享内存段
        self.worker.execute("exit")            # rank0 销毁进程组
        for p in self.ps:
            p.join()
