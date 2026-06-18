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

    def schedule(self) -> tuple[list[Sequence], dict[int, int]]:
        """
        统一连续批调度，返回 (scheduled_seqs, num_scheduled[seq_id]→本步 token 数)。

        无 prefill/decode 阶段切换：一个 batch 内可同时包含正在 decode 的 running
        序列（query 长度 1）与正在 prefill chunk 的 waiting 序列（query 长度 >1）。

        --- 1) RUNNING：decode ---
          每个 running seq 调度 1 token；can_append 不足则抢占 running 末尾。
        --- 2) WAITING：prefill chunk（用剩余预算）---
          can_allocate 探测前缀缓存；按剩余 token 预算切 chunk（任意 seq 可分块）；
          完成整个 prompt → 移入 running，否则留在 waiting 续算。
        """
        scheduled_seqs = []
        num_scheduled: dict[int, int] = {}
        num_batched_tokens = 0

        # ── 1) RUNNING：decode ───────────────────────────────────────────────
        decode_scheduled = []
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            if num_batched_tokens + 1 > self.max_num_batched_tokens:
                break
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    seq = None
                    break
            if seq is None:
                break
            seq.num_scheduled_tokens = 1
            self.block_manager.may_append(seq)
            scheduled_seqs.append(seq)
            decode_scheduled.append(seq)
            num_scheduled[seq.seq_id] = 1
            num_batched_tokens += 1
        # 本步 decode 的 running seq 放回 running 队列（保持原序）
        self.running.extendleft(reversed(decode_scheduled))

        # ── 2) WAITING：prefill chunk（剩余预算）─────────────────────────────
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
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
                # 本步完成整个 prompt（或最后一个 chunk）→ 转入 decode 队列
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            # 否则 chunk 未完成，seq 留在 waiting 队头，下一步续算

            scheduled_seqs.append(seq)
            num_scheduled[seq.seq_id] = n

        return scheduled_seqs, num_scheduled

    def preempt(self, seq: Sequence):
        """
        将 seq 从 running 撤回：释放 KV 块，重置状态，推回 waiting 头部。
        deallocate 会把 num_cached_tokens 归零，故 is_prefill 自动恢复为 True。
        前缀缓存的块 hash 保留，下次调度大概率再次命中。
        """
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int],
                    num_scheduled: dict[int, int]):
        """
        每步推理后更新序列状态：
          1. hash_blocks：注册本步新填满块的哈希
          2. 更新 num_cached_tokens，清零 num_scheduled_tokens
          3. 若仍处于 prefill（chunk 未覆盖完整 prompt），不追加 token
          4. 否则追加新 token，检查终止条件
        """
        for seq, token_id in zip(seqs, token_ids):
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
