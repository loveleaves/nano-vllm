"""
EngineCore：调度 + 执行循环（对齐 vLLM V1 `v1/engine/core.py`）。

V1 把"调度器 + 执行器"封装进 EngineCore，并跑在独立进程里（前端经 ZMQ client
通信），目的是让 GPU 调度循环不被 tokenize / detokenize / HTTP 阻塞。nano 选择
**组件拆分但同进程**（见 docs/arch_engine/design.md 的范围决策）：EngineCore 仍是
一个独立、自洽的类，对外只暴露 add_request / step / abort / has_unfinished_requests，
不含任何 tokenize 或文本处理逻辑——这部分留给 Processor / OutputProcessor。

step() 产出 EngineCoreOutputs（每请求的新 token + 是否结束 + 结束原因），不产出文本。
"""
import atexit

import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.engine.core_types import (
    EngineCoreOutput,
    EngineCoreOutputs,
    EngineCoreRequest,
    FinishReason,
)
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.worker import Worker


class EngineCore:
    """持有 Scheduler + Worker 的调度执行核心。

    进程编排（多进程 TP）从原 LLMEngine 迁移至此：
      rank 0（本进程）：Worker(rank=0)，broadcast + 本地推理 + 采样
      rank 1..N（子进程）：Worker.loop()，经 ShmTransport 等待指令
    """

    def __init__(self, config: Config):
        Sequence.block_size = config.kvcache_block_size
        self.config = config

        self.ps: list = []
        self.events: list = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=Worker, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)

        # rank0 Worker 构造时完成 warmup → 填好 config.num_kvcache_blocks，
        # 之后才能据此构建 Scheduler 的 BlockManager。
        self.worker = Worker(config, 0, self.events)

        self.scheduler = Scheduler(
            num_kvcache_blocks=config.num_kvcache_blocks,
            block_size=config.kvcache_block_size,
            max_num_seqs=config.max_num_seqs,
            max_num_batched_tokens=config.max_num_batched_tokens,
            eos=config.eos,
        )
        # request_id ↔ Sequence，供 abort / 结束清理
        self.requests: dict[str, Sequence] = {}
        atexit.register(self.exit)

    # ── 生命周期 ──────────────────────────────────────────────────────────────
    def exit(self):
        if getattr(self, "worker", None) is None:
            return
        self.worker.call("exit")
        self.worker = None
        for p in self.ps:
            p.join()

    # ── 请求管理 ──────────────────────────────────────────────────────────────
    def add_request(self, request: EngineCoreRequest):
        seq = Sequence(request.prompt_token_ids, request.sampling_params)
        seq.request_id = request.request_id
        self.requests[request.request_id] = seq
        self.scheduler.add(seq)

    def abort_requests(self, request_ids: list[str]):
        for rid in request_ids:
            seq = self.requests.pop(rid, None)
            if seq is not None:
                self.scheduler.abort(seq)

    def has_unfinished_requests(self) -> bool:
        return not self.scheduler.is_finished()

    # ── 单步推理 ──────────────────────────────────────────────────────────────
    def step(self) -> EngineCoreOutputs:
        """调度一批 → 执行 → 后处理 → 收集每请求增量。"""
        seqs, num_scheduled = self.scheduler.schedule()
        if not seqs:
            return EngineCoreOutputs()

        # 记录每个 seq 本步前已产出的 completion 数，用于判定是否真的吐了新 token
        prev_completion = {seq.seq_id: seq.num_completion_tokens for seq in seqs}
        token_ids = self.worker.call("run", seqs)
        self.scheduler.postprocess(seqs, token_ids, num_scheduled)

        outputs: list[EngineCoreOutput] = []
        for seq in seqs:
            # prefill chunk 未覆盖完整 prompt 的步不产 token（postprocess 已跳过追加）
            if seq.num_completion_tokens == prev_completion[seq.seq_id]:
                continue
            finished = seq.is_finished
            finish_reason = None
            if finished:
                finish_reason = (FinishReason.LENGTH
                                 if seq.num_completion_tokens >= seq.max_tokens
                                 else FinishReason.STOP)
                self.requests.pop(seq.request_id, None)
            outputs.append(EngineCoreOutput(
                request_id=seq.request_id,
                new_token_ids=[seq.last_token],
                finished=finished,
                finish_reason=finish_reason,
            ))

        # 吞吐提示：任一 seq 调度 >1 token 记为含 prefill，否则纯 decode
        total = sum(num_scheduled.values())
        is_prefill_step = any(n > 1 for n in num_scheduled.values())
        num_tokens = total if is_prefill_step else -len(seqs)
        return EngineCoreOutputs(outputs=outputs, num_tokens=num_tokens)
