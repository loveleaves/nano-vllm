from dataclasses import dataclass
import torch


@dataclass
class AttentionMetadata:
    """
    单步推理的 attention 元数据，由 ModelRunner.prepare_inputs 构造，
    显式传入 model.forward → Attention.forward。取代旧的全局 Context 单例。

    统一连续批：prefill chunk 与 decode token 共用同一组字段，无 is_prefill 分支。
    decode 序列即 query 长度为 1 的退化情形。

    字段：
      query_start_loc — [num_seqs+1]，累计 query 长度（flash 的 cu_seqlens_q）。
                        decode 批每段步长为 1。
      cu_seqlens_k    — [num_seqs+1]，累计 KV 长度（flash 的 cu_seqlens_k）。
                        每段 = num_cached_tokens + num_scheduled_tokens。
                        （flash_attn 2.8.3 无 seqused_k 参数，故用累计形式。）
      max_query_len   — 批内最大 query 长度。==1 即纯 decode 批（CUDA graph 可用）。
      max_seq_len     — 批内最大 KV 总长度。CUDA graph 捕获时取 max_model_len
                        （经验证：高估 max_seqlen_k 对 varlen kernel 安全）。
      slot_mapping    — [total_tokens]，每个本步 token 写入 KV cache 的绝对 slot 编号。
      block_table     — [num_seqs, max_blocks]，分页 KV 地址。
                        None 表示无 KV cache（仅 warmup 阶段），attention 退回裸 k/v。
      state_slots     — [num_seqs] 的 list[int]，线性注意力（GatedDeltaNet）每个序列的
                        recurrent/conv 状态槽位（按批行序对齐 query_start_loc 的分段）。
                        仅混合模型（Qwen3.5）使用；纯全注意力模型恒为 None。
    """
    query_start_loc: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_query_len: int = 0
    max_seq_len: int = 0
    slot_mapping: torch.Tensor | None = None
    block_table: torch.Tensor | None = None
    state_slots: list | None = None

    @property
    def is_decode_only(self) -> bool:
        """批内所有 seq 的 query 长度均为 1 → 可走 CUDA graph。"""
        return self.max_query_len == 1
