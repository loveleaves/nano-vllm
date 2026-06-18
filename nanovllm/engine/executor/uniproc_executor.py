"""
单进程执行器（对齐 vLLM V1 `v1/executor/uniproc_executor.py::UniProcExecutor`）。

TP=1：单个 Worker 跑在本进程内，无 RPC、无 barrier。collective_rpc 直接本地调用。
"""
from nanovllm.engine.executor.abstract import Executor
from nanovllm.engine.worker import Worker


class UniProcExecutor(Executor):

    def _init_executor(self) -> None:
        self.worker = Worker(self.config, rank=0)

    def collective_rpc(self, method: str, seqs=None) -> list:
        return [self.worker.execute(method, seqs)]

    def execute_model(self, seqs) -> list[int] | None:
        return self.worker.execute("run", seqs)

    def shutdown(self) -> None:
        self.worker.execute("exit")
