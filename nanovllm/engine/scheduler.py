from collections import deque

from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:
    """
    请求调度器（Phase 4：FCFS + Chunked Prefill + 前缀缓存 + 抢占）。

    策略：
      - 优先 prefill：waiting 非空时先做 prefill
      - Chunked Prefill：长 prompt 分块（只允许第一个 seq 分块，避免饥饿）
      - 前缀缓存：can_allocate() 探测缓存命中块数，allocate() 直接复用
      - 抢占：decode 内存不足时，将 running 末尾 seq 推回 waiting 队头

    队列：
      waiting — 待 prefill 的序列（FIFO）
      running — 已完成 prefill、正在 decode 的序列（FIFO）
    """

    def __init__(self, num_kvcache_blocks: int, block_size: int,
                 max_num_seqs: int = 512, max_num_batched_tokens: int = 16384,
                 eos: int = -1):
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.eos = eos
        self.block_size = block_size
        self.block_manager = BlockManager(num_kvcache_blocks, block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self) -> bool:
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        """
        核心调度逻辑，返回 (scheduled_seqs, is_prefill)。

        --- Prefill 阶段（优先） ---
          贪心调度 waiting 队列：
          1. 检查 token 预算（max_num_batched_tokens）
          2. can_allocate 检查内存 + 探测前缀缓存
          3. 只允许第一个 seq 分块（Chunked Prefill）
          4. 分配 KV 块，设置 num_scheduled_tokens
          5. 处理完整 prompt → 移入 running，否则留在 waiting

        --- Decode 阶段 ---
          对 running 每个 seq 调度 1 token：
          1. can_append 检查空闲块
          2. 不足则抢占 running 末尾 seq（preempt）
          3. may_append 按需分配新块
        """
        scheduled_seqs = []
        num_batched_tokens = 0

        # ── Prefill ──────────────────────────────────────────────────────────
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break

            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                # 分块 prefill 的后续 chunk：继续处理剩余部分
                num_tokens = seq.num_tokens - seq.num_cached_tokens

            # 只有第一个 seq（scheduled_seqs 为空）允许分块
            if remaining < num_tokens and scheduled_seqs:
                break

            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)

            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens

            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                # 本步完成整个 prompt（或最后一个 chunk）
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)

            scheduled_seqs.append(seq)

        if scheduled_seqs:
            return scheduled_seqs, True

        # waiting 中有 seq 但内存不足，且 running 为空 → 无法继续
        if self.waiting and not self.running:
            return [], True

        # ── Decode ───────────────────────────────────────────────────────────
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)

        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        """
        将 seq 从 running 撤回：释放 KV 块，重置状态，推回 waiting 头部。
        前缀缓存的块 hash 保留，下次调度大概率再次命中。
        """
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        """
        每步推理后更新序列状态：
          1. hash_blocks：注册本步新填满块的哈希
          2. 更新 num_cached_tokens，清零 num_scheduled_tokens
          3. 若是 chunked prefill 中间步，不追加 token（继续等待）
          4. 否则追加新 token，检查终止条件
        """
        for seq, token_id in zip(seqs, token_ids):
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            # chunked prefill 未完成整个 prompt，本步不产出新 token
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or \
               seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                if seq in self.running:
                    self.running.remove(seq)
