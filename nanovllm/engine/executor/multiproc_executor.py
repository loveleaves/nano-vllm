"""
多进程执行器（对齐 vLLM V1 `v1/executor/multiproc_executor.py::MultiprocExecutor`）。

**进程隔离**：所有 rank（含 rank0）都是 spawn 子进程，executor 进程本身**不内联 Worker、
不加入 NCCL 组**——只通过两条共享内存通道与各 worker 通信：
  - 广播（ShmTransport）：executor → 所有 worker，下发同一指令；
  - 回传（ResultChannel）：输出 rank（rank0，唯一采样者）→ executor，取回结果。

与对齐前（rank0 内联在引擎进程）的差异见 docs/arch_worker_isolation/design.md。

进程编排：
  启动：executor 先建两条通道 → spawn rank0..N-1（各自 Worker：NCCL init + warmup +
        allocate_kv_cache + cudagraph）→ collective_rpc("num_kvcache_blocks") 把 rank0
        算出的块数回传、填入引擎侧 config（供 EngineCore 建 Scheduler）。
  退出：广播 "exit" → 各 worker 跳出循环、销毁进程组 → join → executor 关闭/释放通道。
"""
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.engine.executor.abstract import Executor
from nanovllm.engine.rpc import ResultChannel, ShmTransport
from nanovllm.engine.worker import Worker

OUTPUT_RANK = 0   # 唯一采样、回传结果的 rank


def _worker_proc_main(config: Config, rank: int, broadcast_event, result_event):
    """子进程入口（每个 rank）：建 Worker → 开通道 → 收发循环 → 清理。"""
    worker = Worker(config, rank)                    # ModelRunner: NCCL init + warmup + allocate
    broadcast = ShmTransport(rank, broadcast_event, create=False)
    result = ResultChannel(result_event, create=False) if rank == OUTPUT_RANK else None
    while True:
        method, seqs, finished = broadcast.recv()
        if method == "exit":
            break
        out = worker.execute(method, seqs, finished)
        if rank == OUTPUT_RANK:
            result.send(out)                         # 回传 token_ids / num_kvcache_blocks
    broadcast.close()
    if result is not None:
        result.close()
    worker.execute("exit")                           # destroy_process_group（各 rank 集体）


class MultiProcExecutor(Executor):

    def _init_executor(self) -> None:
        config = self.config
        n = config.tensor_parallel_size
        ctx = mp.get_context("spawn")

        # 先建通道（子进程启动后即可打开），再 spawn 各 rank
        self.worker_events = [ctx.Event() for _ in range(n)]
        self.result_event = ctx.Event()
        self.broadcast = ShmTransport(rank=-1, events=self.worker_events, create=True)
        self.result = ResultChannel(self.result_event, create=True)

        self.ps: list = []
        for rank in range(n):
            p = ctx.Process(target=_worker_proc_main,
                            args=(config, rank, self.worker_events[rank], self.result_event))
            p.start()
            self.ps.append(p)

        # 阻塞直到各 worker 完成 __init__（warmup+allocate）并响应：把 rank0 算出的
        # num_kvcache_blocks 回填引擎侧 config，供 EngineCore 据此构建 Scheduler。
        self.config.num_kvcache_blocks = self.collective_rpc("num_kvcache_blocks")[0]

    def _check_workers_alive(self) -> None:
        """探测 worker 子进程存活；任一异常退出则终止其余并抛错（避免永久阻塞）。"""
        dead = [(rank, p.exitcode) for rank, p in enumerate(self.ps) if not p.is_alive()]
        if dead:
            for p in self.ps:
                if p.is_alive():
                    p.terminate()
            raise RuntimeError(f"worker 子进程异常退出，ranks/exitcodes={dead}")

    def collective_rpc(self, method: str, seqs=None, finished_seq_ids=None) -> list:
        """向所有 rank 广播指令，返回 [输出 rank 的结果]（与 UniProc 的列表契约一致）。

        轮询等待回传，期间探测 worker 存活——worker 崩溃时立即抛错而非永久阻塞。
        """
        self.broadcast.broadcast(method, seqs, finished_seq_ids)
        return [self.result.recv(alive_check=self._check_workers_alive)]

    def execute_model(self, seqs, finished_seq_ids=None) -> list[int] | None:
        return self.collective_rpc("run", seqs, finished_seq_ids)[0]

    def shutdown(self) -> None:
        if not getattr(self, "ps", None):
            return
        self.broadcast.broadcast("exit")             # 通知各 rank 跳出循环并销毁进程组
        for p in self.ps:
            p.join()
        self.ps = []
        self.broadcast.close()
        self.broadcast.unlink()
        self.result.close()
        self.result.unlink()
