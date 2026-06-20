"""
执行器抽象（对齐 vLLM V1 `v1/executor/abstract.py::Executor`）。

Executor 夹在 EngineCore 与 Worker 之间，封装"把一步推理分发到所有 rank 并收集结果"
的进程编排。EngineCore 只依赖 Executor 接口（execute_model / shutdown），不感知单进程
还是多进程 TP。

  collective_rpc(method, seqs) — 向所有 rank 下发同一指令，返回各 rank 结果列表
  execute_model(seqs)          — 跑一步推理，返回 rank0 的采样 token（其余 rank 返回 None）
"""
from abc import ABC, abstractmethod

from nanovllm.config import Config


class Executor(ABC):

    def __init__(self, config: Config):
        self.config = config
        self._init_executor()

    @staticmethod
    def get_class(config: Config) -> type["Executor"]:
        """按 distributed_executor_backend 选择执行器后端（对齐 V1 分派）。

        backend 为 None 时按 TP 自动选：TP=1→uni(单进程内联)，TP>1→mp(各 rank 子进程隔离)；
        显式 "mp" 可让 TP=1 也走进程隔离。
        """
        backend = config.distributed_executor_backend
        if backend is None:
            backend = "uni" if config.tensor_parallel_size == 1 else "mp"
        if backend == "uni":
            from nanovllm.engine.executor.uniproc_executor import UniProcExecutor
            return UniProcExecutor
        from nanovllm.engine.executor.multiproc_executor import MultiProcExecutor
        return MultiProcExecutor

    @abstractmethod
    def _init_executor(self) -> None:
        """构造各 rank 的 Worker 并完成进程编排初始化。"""
        ...

    @abstractmethod
    def collective_rpc(self, method: str, seqs=None, finished_seq_ids=None) -> list:
        """向所有 rank 下发同一方法，返回各 rank 结果列表（rank0 在首位）。"""
        ...

    def execute_model(self, seqs, finished_seq_ids=None) -> list[int] | None:
        """跑一步推理，返回 rank0 采样出的 token_ids。"""
        return self.collective_rpc("run", seqs, finished_seq_ids)[0]

    def execute_swap(self, blocks_to_swap_in, blocks_to_swap_out) -> None:
        """执行本步 KV 块搬运（抢占换出 / 换回）。默认不支持（仅 UniProc 内联实现）。"""
        if blocks_to_swap_in or blocks_to_swap_out:
            raise NotImplementedError(
                "swap 抢占仅 UniProc（TP=1 内联）支持；MultiProc 需扩展 RPC 载荷")

    @abstractmethod
    def shutdown(self) -> None:
        ...
