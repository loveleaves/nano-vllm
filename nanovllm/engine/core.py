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
from nanovllm.engine.sequence import Sequence, SequenceStatus


class EngineCore:
    """持有 Scheduler + Executor 的调度执行核心。

    进程编排下沉到 Executor（UniProc / MultiProc，按 TP 规模选择）：
      EngineCore.step → executor.execute_model(seqs) → rank0 采样 token_ids
    """

    def __init__(self, config: Config):
        Sequence.block_size = config.kvcache_block_size
        self.config = config

        # Executor 构造时完成各 rank Worker 初始化（含 rank0 warmup → 填好
        # config.num_kvcache_blocks），之后才能据此构建 Scheduler 的 KVCacheManager。
        executor_class = Executor.get_class(config)
        self.executor = executor_class(config)

        self.scheduler = Scheduler(
            num_kvcache_blocks=config.num_kvcache_blocks,
            block_size=config.kvcache_block_size,
            max_num_seqs=config.max_num_seqs,
            max_num_batched_tokens=config.max_num_batched_tokens,
            eos=config.eos,
            policy=SchedulingPolicy(config.scheduling_policy),
            num_swap_blocks=config.num_swap_blocks,
        )
        # request_id ↔ Sequence，供 abort / 结束清理
        self.requests: dict[str, Sequence] = {}
        # 最近一步的调度统计（可观测性；get_stats 取用）
        self.scheduler_stats = None
        # 异步调度：已下发 GPU 但结果尚未回收的"在飞"步（SchedulerOutput），None 表示无
        self.async_scheduling = config.async_scheduling
        self._inflight = None

        # 投机解码（仅 UniProc）：n-gram 草案 + 一步多 token verify
        self.use_spec = config.speculative_num_tokens > 0
        self.spec_decoder = None
        if self.use_spec:
            from nanovllm.spec_decode import NgramProposer, SpeculativeDecoder
            self.spec_decoder = SpeculativeDecoder(NgramProposer(
                max_n=config.speculative_ngram_max, k=config.speculative_num_tokens))
        atexit.register(self.exit)

    def get_stats(self):
        """返回最近一步的 SchedulerStats（None 表示尚未 step）。"""
        return self.scheduler_stats

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
        seq.grammar = request.grammar   # 引导解码 Grammar（None 表示无约束）
        self.requests[request.request_id] = seq
        self.scheduler.add_request(seq)

    def abort_requests(self, request_ids: list[str]):
        for rid in request_ids:
            seq = self.requests.pop(rid, None)
            if seq is not None:
                self.scheduler.abort(seq)

    def has_unfinished_requests(self) -> bool:
        # 异步调度下还需排空"在飞"步（其结果尚未回收）
        return not self.scheduler.is_finished() or self._inflight is not None

    # ── 单步推理 ──────────────────────────────────────────────────────────────
    def step(self) -> EngineCoreOutputs:
        if self.async_scheduling:
            return self._step_async()
        if getattr(self, "use_spec", False):
            return self._step_spec()
        return self._step_sync()

    def _step_sync(self) -> EngineCoreOutputs:
        """调度一批 → 执行 → 后处理 → 收集每请求增量。"""
        sched_output = self.scheduler.schedule()
        self.scheduler_stats = self.scheduler.make_stats(
            sched_output.total_num_scheduled_tokens)
        if sched_output.is_empty:
            return EngineCoreOutputs()

        seqs = sched_output.scheduled_seqs
        # 抢占换出 / 换回的 KV 块搬运，须在 execute_model 之前（swap_out 读旧 KV）
        if sched_output.blocks_to_swap_in or sched_output.blocks_to_swap_out:
            self.executor.execute_swap(
                sched_output.blocks_to_swap_in, sched_output.blocks_to_swap_out)
        # 记录每个 seq 本步前已产出的 completion 数，用于判定是否真的吐了新 token
        prev_completion = {seq.seq_id: seq.num_completion_tokens for seq in seqs}
        token_ids, step_logprobs = self.executor.execute_model(
            seqs, sched_output.finished_seq_ids)
        self.scheduler.update_from_output(sched_output, token_ids)

        outputs: list[EngineCoreOutput] = []
        for i, seq in enumerate(seqs):
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
                logprobs=step_logprobs[i] if step_logprobs is not None else None,
            ))

        return EngineCoreOutputs(outputs=outputs,
                                 num_tokens=self._throughput_hint(sched_output))

    # ── 投机解码：基础步 + 多 token 扩展 ────────────────────────────────────────
    def _step_spec(self) -> EngineCoreOutputs:
        """投机解码步：先跑一次普通同步步（推进 prefill / 产 1 个基准 token），再对每个
        decode 序列做一步**多 token** 投机扩展（propose → verify → reject → 追加接受 token）。

        正常路径（_step_sync）完全不变 → 零回归；spec 仅作为 decode 序列的附加扩展。
        """
        base = self._step_sync()
        if not base.outputs:
            return base
        for out in base.outputs:
            if out.finished:
                continue
            seq = self.requests.get(out.request_id)
            if seq is None or seq.is_prefill:   # 仅对已进入 decode 的序列做投机
                continue
            extra, finished, reason = self._extend_with_spec(seq)
            if extra:
                out.new_token_ids.extend(extra)
            if finished:
                out.finished = True
                out.finish_reason = reason
                self.requests.pop(out.request_id, None)
        return base

    def _extend_with_spec(self, seq):
        """对一个 decode 序列做一步投机扩展，返回 (接受的额外 token, 是否结束, 结束原因)。

        KV 自愈（为何无需显式回滚 KV）：verify 在位置 L0..L0+k-1 写入草案 token 的 KV。
        被接受的匹配位（草案==目标）KV 正确；唯一可能"脏"的是最后那个 token——修正位的 KV 是
        草案值（错）、或奖励位根本没写 KV。但我们把 num_cached 设为 num_tokens-1，即把"最新
        token"标记为"KV 未就绪"，下一步前向它时会**覆写**成正确 KV——这与普通 decode 中"刚生成
        的 token 其 KV 要等下一步才写"完全一致。故只需释放尾部多余块（前部块 KV 不动）即可。
        """
        bm = self.scheduler.block_manager
        L0 = seq.num_tokens
        drafts = self.spec_decoder.proposer.propose(seq.token_ids)
        if not drafts:
            return [], False, None

        # 1) 投机追加草案 token（分配块，使 verify 的 KV 落点存在）
        for d in drafts:
            seq.append_token(d)
            bm.may_append(seq)
        # 2) 目标模型并行验证 k+1 个位置（GPU；返回 k+1 个 argmax）
        targets = self.executor.verify_spec(seq, len(drafts))
        accepted = self.spec_decoder.rejection_sampler.verify_greedy(drafts, targets)

        # 3) 在接受序列内做 EOS / max_tokens 终止判定（截到首个结束处）
        final, finished, reason = self._apply_finish(seq, L0, accepted)

        # 4) 回滚到 L0 + final：覆写接受 token 值、释放尾部块（前部块 KV 完好）
        seq.token_ids = seq.token_ids[:L0] + final
        seq.num_tokens = L0 + len(final)
        seq.last_token = seq.token_ids[-1]
        # 与普通 decode 一致的不变式：最新 token 的 KV 尚未写入（修正位 KV 为草案值/奖励位无 KV）
        seq.num_cached_tokens = seq.num_tokens - 1
        bm.truncate_blocks(seq)

        if finished:
            seq.status = SequenceStatus.FINISHED
            bm.deallocate(seq)
            if seq in self.scheduler.running:
                self.scheduler.running.remove(seq)
            self.scheduler.finished_req_ids.add(seq.seq_id)
        return final, finished, reason

    def _apply_finish(self, seq, L0, accepted):
        """逐个接受 token 判定终止，返回 (截断后的接受序列, 是否结束, 结束原因)。"""
        result = []
        for i, tok in enumerate(accepted):
            result.append(tok)
            completion = (L0 + i + 1) - seq.num_prompt_tokens
            if (not seq.ignore_eos and tok == self.scheduler.eos):
                return result, True, FinishReason.STOP
            if completion >= seq.max_tokens:
                return result, True, FinishReason.LENGTH
        return result, False, None

    def _throughput_hint(self, sched_output) -> int:
        """吞吐提示：任一 seq 调度 >1 token 记为含 prefill（正数 token 数），否则纯 decode（负 seq 数）。"""
        is_prefill_step = any(n > 1 for n in sched_output.num_scheduled_tokens.values())
        return (sched_output.total_num_scheduled_tokens if is_prefill_step
                else -len(sched_output.scheduled_seqs))

    # ── 异步调度：step N 的 GPU 计算与 step N+1 的 CPU 调度重叠 ──────────────────
    def _step_async(self) -> EngineCoreOutputs:
        """流水深度 1 的异步调度（仅 UniProc）：

          1. schedule 本步并**非阻塞**下发 GPU（采样 token 留在 GPU，前向喂给本步输入，
             不做 D2H 同步）——此时上一步的 GPU 计算与本步的 CPU 调度已重叠；
          2. 回收上一步（在飞）结果：D2H 同步取回 token、回填占位、判定 EOS、产出增量；
          3. 推进本步记账（在 resolve 之后，故已结束序列被跳过、哈希见已回填 token），
             promote 本步采样张量为下一步的前向源。
        """
        sched_output = self.scheduler.schedule()
        self.scheduler_stats = self.scheduler.make_stats(
            sched_output.total_num_scheduled_tokens)

        launched = not sched_output.is_empty
        if launched:
            # 非阻塞下发：内部用上一步留在 GPU 的采样 token 前向回填本步 decode 输入
            self.executor.execute_model_async(
                sched_output.scheduled_seqs, sched_output.finished_seq_ids)

        outputs = EngineCoreOutputs()
        if self._inflight is not None:
            tok_by_id, lp_by_id = self.executor.resolve_inflight()   # D2H 同步取回上一步
            token_ids = [tok_by_id[s.seq_id] for s in self._inflight.scheduled_seqs]
            produced = self.scheduler.resolve_output(self._inflight, token_ids)
            outputs = self._build_async_outputs(produced, lp_by_id, self._inflight)

        if launched:
            # 推进必须在 resolve 之后：已结束序列被 advance 跳过；哈希见已回填的真实 token
            self.scheduler.advance_after_schedule(sched_output)
            self.executor.promote_async()
            self._inflight = sched_output
        else:
            # 空调度步未下发 model → 本步排空的 finished_seq_ids 无处投递给 InputBatch
            # 回收行槽位，退回累积器留待下一次 launch（可能是下个 generate）处理。
            self.scheduler.finished_req_ids |= sched_output.finished_seq_ids
            self._inflight = None
        return outputs

    def _build_async_outputs(self, produced, lp_by_id, sched_output) -> EngineCoreOutputs:
        outputs: list[EngineCoreOutput] = []
        for seq in produced:
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
                logprobs=lp_by_id.get(seq.seq_id) if lp_by_id is not None else None,
            ))
        return EngineCoreOutputs(outputs=outputs,
                                 num_tokens=self._throughput_hint(sched_output))
