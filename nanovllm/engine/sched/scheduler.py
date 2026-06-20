from collections import deque

from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.kv_cache import KVCacheManager
from nanovllm.engine.sched.interface import SchedulerInterface
from nanovllm.engine.sched.output import SchedulerOutput
from nanovllm.engine.sched.request_queue import (
    RequestQueue,
    SchedulingPolicy,
    create_request_queue,
)


class Scheduler(SchedulerInterface):
    """
    请求调度器（统一连续批 + Chunked Prefill + 前缀缓存 + 抢占）。

    对齐 V1：实现 SchedulerInterface，schedule() 产出结构化 SchedulerOutput，
    waiting 队列经可插拔 RequestQueue（FCFS / PRIORITY）承载。

    策略：
      - 优先 prefill：waiting 非空时先做 prefill
      - Chunked Prefill：长 prompt 按 token 预算分块（任意 seq 可分块）
      - 前缀缓存：can_allocate() 探测缓存命中块数，allocate() 直接复用
      - 抢占：decode 内存不足时，将 running 末尾 seq 推回 waiting

    队列：
      waiting — 待 prefill 的序列（RequestQueue，策略决定出队顺序）
      running — 已完成 prefill、正在 decode 的序列（FCFS deque）
    """

    def __init__(self, num_kvcache_blocks: int, block_size: int,
                 max_num_seqs: int = 512, max_num_batched_tokens: int = 16384,
                 eos: int = -1,
                 policy: SchedulingPolicy = SchedulingPolicy.FCFS,
                 num_swap_blocks: int = 0):
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.eos = eos
        self.block_size = block_size
        self.block_manager = KVCacheManager(num_kvcache_blocks, block_size, num_swap_blocks)
        self.swap_enabled = num_swap_blocks > 0
        self.waiting: RequestQueue = create_request_queue(policy)
        self.running: deque[Sequence] = deque()
        self.swapped: deque[Sequence] = deque()   # 已换出到 CPU、待换回的序列（FIFO）
        # 上一步结束 / 中止、需在执行器持久批回收行槽位的 seq_id（schedule 时随
        # 本步被抢占者一并下发，清空累积器）；对齐 V1 Scheduler.finished_req_ids
        self.finished_req_ids: set[int] = set()

    # ── 接口实现 ──────────────────────────────────────────────────────────────
    def add_request(self, seq: Sequence):
        self.waiting.add_request(seq)

    def get_num_unfinished_requests(self) -> int:
        return len(self.waiting) + len(self.running) + len(self.swapped)

    def make_stats(self, num_scheduled_tokens: int = 0):
        """聚合一份调度快照（运行/等待数 + KV 利用率），供 metrics 使用。"""
        from nanovllm.engine.metrics import SchedulerStats
        pool = self.block_manager.block_pool
        total = len(pool.blocks)
        return SchedulerStats(
            num_running=len(self.running),
            num_waiting=len(self.waiting),
            num_scheduled_tokens=num_scheduled_tokens,
            num_gpu_blocks=total,
            num_gpu_blocks_used=total - pool.get_num_free_blocks(),
        )

    def is_finished(self) -> bool:
        return not self.waiting and not self.running and not self.swapped

    def schedule(self) -> SchedulerOutput:
        """
        统一连续批调度。返回结构化 SchedulerOutput。

        无 prefill/decode 阶段切换：一个 batch 内可同时包含正在 decode 的 running
        序列（query 长度 1）与正在 prefill chunk 的 waiting 序列（query 长度 >1）。
        """
        scheduled_seqs: list[Sequence] = []
        num_scheduled: dict[int, int] = {}
        produced_seq_ids: set[int] = set()   # 本步产出 token 的 seq（decode + prefill 收尾）
        preempted_seq_ids: set[int] = set()
        swap_out_list: list[tuple[int, int]] = []
        swap_in_list: list[tuple[int, int]] = []
        num_batched_tokens = 0
        # 排空上一步累积的结束/中止 seq_id，本步随被抢占者一并下发给 InputBatch 回收
        finished_seq_ids = self.finished_req_ids
        self.finished_req_ids = set()

        # ── 0) SWAPPED：换回（恢复 decode，先于本步 RUNNING）──────────────────
        # 已换出到 CPU 的序列按 FIFO 优先换回 GPU；换回得到的块槽位与抢占（phase 1）
        # 新释放的块不相交，故同一步内无冲突。
        if self.swapped:
            while (self.swapped and len(self.running) < self.max_num_seqs
                   and self.block_manager.can_swap_in(self.swapped[0])):
                seq = self.swapped.popleft()
                swap_in_list.extend(self.block_manager.swap_in(seq))
                seq.status = SequenceStatus.RUNNING
                self.running.append(seq)

        # ── 1) RUNNING：decode ───────────────────────────────────────────────
        decode_scheduled = []
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            if num_batched_tokens + 1 > self.max_num_batched_tokens:
                break
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    victim = self.running.pop()
                    self.preempt(victim, swap_out_list)
                    preempted_seq_ids.add(victim.seq_id)
                else:
                    self.preempt(seq, swap_out_list)
                    preempted_seq_ids.add(seq.seq_id)
                    seq = None
                    break
            if seq is None:
                break
            seq.num_scheduled_tokens = 1
            self.block_manager.may_append(seq)
            scheduled_seqs.append(seq)
            decode_scheduled.append(seq)
            num_scheduled[seq.seq_id] = 1
            produced_seq_ids.add(seq.seq_id)   # decode 步必产 token
            num_batched_tokens += 1
        # 本步 decode 的 running seq 放回 running 队列（保持原序）
        self.running.extendleft(reversed(decode_scheduled))

        # ── 2) WAITING：prefill chunk（剩余预算）─────────────────────────────
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting.peek_request()
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining <= 0:
                break

            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                # 分块 prefill 的后续 chunk：继续处理剩余部分
                num_tokens = seq.num_tokens - seq.num_cached_tokens

            n = min(num_tokens, remaining)
            if n <= 0:
                break

            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)

            seq.num_scheduled_tokens = n
            num_batched_tokens += n

            if seq.num_cached_tokens + n == seq.num_tokens:
                # 本步完成整个 prompt（或最后一个 chunk）→ 转入 decode 队列，产出首 token
                seq.status = SequenceStatus.RUNNING
                self.waiting.pop_request()
                self.running.append(seq)
                produced_seq_ids.add(seq.seq_id)
            # 否则 chunk 未完成，seq 留在 waiting 队首，下一步续算

            scheduled_seqs.append(seq)
            num_scheduled[seq.seq_id] = n

        return SchedulerOutput(
            scheduled_seqs=scheduled_seqs,
            num_scheduled_tokens=num_scheduled,
            total_num_scheduled_tokens=num_batched_tokens,
            preempted_seq_ids=preempted_seq_ids,
            finished_seq_ids=finished_seq_ids | preempted_seq_ids,
            blocks_to_swap_out=swap_out_list,
            blocks_to_swap_in=swap_in_list,
            produced_token_seq_ids=produced_seq_ids,
        )

    def preempt(self, seq: Sequence, swap_out_list: list[tuple[int, int]] | None = None):
        """
        将 seq 从 running 撤回。两种策略：

          - swap（num_swap_blocks>0 且 swap 区有空槽）：把 KV 块搬到 CPU swap 区，
            **保留** num_cached_tokens，seq 进入 self.swapped 待换回（恢复 decode）。
            搬运 (gpu_block_id, swap_slot) 收集进 swap_out_list 交执行器做 D2H。
          - recompute（默认 / swap 区满）：deallocate 释放 KV 块、num_cached_tokens
            归零（is_prefill 自动恢复 True），seq 推回 waiting 队首重算。

        两种策略前缀缓存的块 hash 都保留，下次大概率再次命中。

        异步调度：被抢占 recompute 的序列丢弃其未回填的占位 token（其在飞结果将在
        resolve_output 中按"非运行态"跳过），从最后一个已回填 token 干净重算。
        """
        if self.swap_enabled and swap_out_list is not None \
                and self.block_manager.can_swap_out(seq):
            swap_out_list.extend(self.block_manager.swap_out(seq))
            seq.status = SequenceStatus.WAITING
            self.swapped.append(seq)
            return
        if seq.num_pending:
            seq.truncate_pending()   # 同步模式恒为 0（no-op）
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.prepend_request(seq)

    def abort(self, seq: Sequence):
        """显式中止一个请求：从队列移除并释放其 KV 块（停止串命中 / 外部 abort）。

        deallocate 对空 block_table 是安全的 no-op，故 waiting 中尚未分配的 seq 也可中止。
        """
        seq.status = SequenceStatus.FINISHED
        if seq in self.swapped:
            # 已换出：归还 swap 槽位到空闲池（无 GPU 块可释放）
            slots = self.block_manager.swapped_slots.pop(seq.seq_id, [])
            self.block_manager.free_swap_slots.extend(slots)
            self.swapped.remove(seq)
        else:
            self.block_manager.deallocate(seq)
            if seq in self.running:
                self.running.remove(seq)
            if seq in self.waiting:
                self.waiting.remove_request(seq)
        self.finished_req_ids.add(seq.seq_id)

    def update_from_output(self, output: SchedulerOutput, token_ids: list[int]):
        """
        每步推理后更新序列状态（对齐 V1 update_from_output，原 postprocess）：
          1. hash_blocks：注册本步新填满块的哈希
          2. 更新 num_cached_tokens，清零 num_scheduled_tokens
          3. 若仍处于 prefill（chunk 未覆盖完整 prompt），不追加 token
          4. 否则追加新 token，检查终止条件
        """
        for seq, token_id in zip(output.scheduled_seqs, token_ids):
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            # chunked prefill 未覆盖完整 prompt（is_prefill 由进度派生），本步不产 token
            if seq.is_prefill:
                continue
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or \
               seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                if seq in self.running:
                    self.running.remove(seq)
                self.finished_req_ids.add(seq.seq_id)

    # ── 异步调度：调度时推进记账 + 结果返回时回填 ────────────────────────────────
    def advance_after_schedule(self, output: SchedulerOutput):
        """异步调度：紧随 schedule() 推进序列记账（不等结果），使下一步可立即调度。

        镜像 update_from_output 的"长度推进"部分，但对产出 token 的步追加**占位** token
        （真实值经 GPU 前向喂给下一步、结果返回时由 resolve_output 回填），且不做 EOS 判定
        （EOS 在 resolve_output 检出）。已结束序列（上一步 EOS 检出）跳过。
        """
        for seq in output.scheduled_seqs:
            if seq.is_finished:
                continue
            self.block_manager.hash_blocks(seq)   # 仅哈希已喂入（已知）token 的满块
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if seq.seq_id in output.produced_token_seq_ids:
                seq.append_placeholder()

    def resolve_output(self, output: SchedulerOutput,
                       token_ids: list[int]) -> list[Sequence]:
        """异步调度：在飞步结果返回时回填占位 token + EOS/max 判定，返回产出 token 的序列。

        已结束序列（被多调度一步）丢弃其占位与垃圾结果。返回的序列供 EngineCore 产出增量。
        """
        produced: list[Sequence] = []
        for seq, token_id in zip(output.scheduled_seqs, token_ids):
            # 仅回填仍在 decode 的序列：跳过 partial prefill（无 token）、已结束（多调度
            # 一步）、被抢占/中止（status≠RUNNING 或无待回填占位）——其在飞结果丢弃。
            if seq.seq_id not in output.produced_token_seq_ids:
                continue
            if seq.status != SequenceStatus.RUNNING or seq.num_pending == 0:
                continue
            seq.resolve_placeholder(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or \
               seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                if seq in self.running:
                    self.running.remove(seq)
                self.finished_req_ids.add(seq.seq_id)
            produced.append(seq)
        return produced
