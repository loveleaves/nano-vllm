"""
单进程执行器（对齐 vLLM V1 `v1/executor/uniproc_executor.py::UniProcExecutor`）。

TP=1：单个 Worker 跑在本进程内，无 RPC、无 barrier。collective_rpc 直接本地调用。
"""
from nanovllm.engine.executor.abstract import Executor
from nanovllm.engine.worker import Worker


class UniProcExecutor(Executor):

    def _init_executor(self) -> None:
        self.worker = Worker(self.config, rank=0)

    def collective_rpc(self, method: str, seqs=None, finished_seq_ids=None) -> list:
        return [self.worker.execute(method, seqs, finished_seq_ids)]

    def execute_model(self, seqs, finished_seq_ids=None) -> list[int] | None:
        return self.worker.execute("run", seqs, finished_seq_ids)

    def execute_swap(self, blocks_to_swap_in, blocks_to_swap_out) -> None:
        if blocks_to_swap_in or blocks_to_swap_out:
            self.worker.model_runner.execute_swap(blocks_to_swap_in, blocks_to_swap_out)

    def execute_model_async(self, seqs, finished_seq_ids=None) -> None:
        self.worker.model_runner.execute_model_async(seqs, finished_seq_ids)

    def resolve_inflight(self):
        return self.worker.model_runner.resolve_inflight()

    def promote_async(self) -> None:
        self.worker.model_runner.promote_async()

    def shutdown(self) -> None:
        self.worker.execute("exit")
