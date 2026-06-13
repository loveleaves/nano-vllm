from collections import deque

from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:
    """
    请求调度器（阶段一：基础 FCFS，无 chunked prefill，无抢占）。

    策略：
      - 优先 prefill（waiting 非空时先做 prefill）
      - 每步 decode 对 running 中所有 seq 各处理 1 token
      - 内存不足时停止调度（不抢占）

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

        Prefill：遍历 waiting，贪心选取不超过 token 预算且有足够 KV 块的 seq。
        Decode：对所有 running seq 各调度 1 token（若无空闲块则停止）。
        """
        scheduled_seqs = []
        num_batched_tokens = 0

        # ── Prefill ──────────────────────────────────────────────────────────
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            num_tokens = seq.num_tokens - seq.num_cached_tokens
            if num_batched_tokens + num_tokens > self.max_num_batched_tokens:
                break
            num_cached_blocks = self.block_manager.can_allocate(seq)
            if num_cached_blocks == -1:
                break
            self.block_manager.allocate(seq, num_cached_blocks)
            seq.num_scheduled_tokens = num_tokens
            num_batched_tokens += num_tokens
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq)

        if scheduled_seqs:
            return scheduled_seqs, True

        # waiting 中有 seq 但全部受内存限制无法调度：返回空列表（调用方需处理）
        if self.waiting and not self.running:
            return [], True

        # ── Decode ───────────────────────────────────────────────────────────
        for seq in list(self.running):
            if len(scheduled_seqs) >= self.max_num_seqs:
                break
            if not self.block_manager.can_append(seq):
                break
            seq.num_scheduled_tokens = 1
            seq.is_prefill = False
            self.block_manager.may_append(seq)
            scheduled_seqs.append(seq)

        # 阶段一无抢占：若 running 非空却一个都排不进（队首就缺块），
        # 后续步也不可能腾出空间（没有 seq 能 decode 进而结束），会陷入死循环。
        # 这里直接报错而非静默空转，便于定位「KV cache 不足」。
        if not scheduled_seqs and self.running:
            raise RuntimeError(
                "KV cache 不足以继续 decode，且当前阶段未实现抢占；"
                "请调大 gpu_memory_utilization 或减小 max_num_seqs / max_model_len。"
            )

        return scheduled_seqs, False

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        """
        每步推理后更新序列状态：
          1. 更新 num_cached_tokens，清零 num_scheduled_tokens
          2. 追加新生成的 token
          3. 检查终止条件（EOS 或 max_tokens）
        """
        for seq, token_id in zip(seqs, token_ids):
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or \
               seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                if seq in self.running:
                    self.running.remove(seq)
