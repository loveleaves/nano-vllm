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

from nanovllm.config import Config
from nanovllm.engine.core_types import (
    EngineCoreOutput,
    EngineCoreOutputs,
    EngineCoreRequest,
    FinishReason,
)
from nanovllm.engine.executor import Executor
from nanovllm.engine.sched import Scheduler, SchedulingPolicy
from nanovllm.engine.sequence import Sequence


class EngineCore:
    """持有 Scheduler + Executor 的调度执行核心。

    进程编排下沉到 Executor（UniProc / MultiProc，按 TP 规模选择）：
      EngineCore.step → executor.execute_model(seqs) → rank0 采样 token_ids
    """

    def __init__(self, config: Config):
        Sequence.block_size = config.kvcache_block_size
        self.config = config

        # Executor 构造时完成各 rank Worker 初始化（含 rank0 warmup → 填好
        # config.num_kvcache_blocks），之后才能据此构建 Scheduler 的 BlockManager。
        executor_class = Executor.get_class(config)
        self.executor = executor_class(config)

        self.scheduler = Scheduler(
            num_kvcache_blocks=config.num_kvcache_blocks,
            block_size=config.kvcache_block_size,
            max_num_seqs=config.max_num_seqs,
            max_num_batched_tokens=config.max_num_batched_tokens,
            eos=config.eos,
            policy=SchedulingPolicy(config.scheduling_policy),
        )
        # request_id ↔ Sequence，供 abort / 结束清理
        self.requests: dict[str, Sequence] = {}
        atexit.register(self.exit)

    # ── 生命周期 ──────────────────────────────────────────────────────────────
    def exit(self):
        if getattr(self, "executor", None) is None:
            return
        self.executor.shutdown()
        self.executor = None

    # ── 请求管理 ──────────────────────────────────────────────────────────────
    def add_request(self, request: EngineCoreRequest):
        seq = Sequence(request.prompt_token_ids, request.sampling_params,
                       priority=request.priority)
        seq.request_id = request.request_id
        self.requests[request.request_id] = seq
        self.scheduler.add_request(seq)

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
        sched_output = self.scheduler.schedule()
        if sched_output.is_empty:
            return EngineCoreOutputs()

        seqs = sched_output.scheduled_seqs
        # 记录每个 seq 本步前已产出的 completion 数，用于判定是否真的吐了新 token
        prev_completion = {seq.seq_id: seq.num_completion_tokens for seq in seqs}
        token_ids = self.executor.execute_model(seqs)
        self.scheduler.update_from_output(sched_output, token_ids)

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
        is_prefill_step = any(n > 1 for n in sched_output.num_scheduled_tokens.values())
        num_tokens = (sched_output.total_num_scheduled_tokens
                      if is_prefill_step else -len(seqs))
        return EngineCoreOutputs(outputs=outputs, num_tokens=num_tokens)
