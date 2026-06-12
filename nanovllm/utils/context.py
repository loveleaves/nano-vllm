from dataclasses import dataclass
import torch


@dataclass
class Context:
    """
    单步推理的全局上下文，在 model_runner.prepare_prefill/decode 中设置，
    在 Attention.forward 和 ParallelLMHead.forward 中读取。

    通过全局变量隐式传递，避免将推理元数据作为参数逐层传递。

    prefill 字段：
      cu_seqlens_q/k  — [num_seqs+1]，累计序列长度（flash_attn_varlen 接口）
      max_seqlen_q/k  — 本批次最大序列长度
      slot_mapping    — [total_tokens]，每个 token 写入 KV cache 的绝对 slot 编号
      block_tables    — [num_seqs, max_blocks]，前缀缓存时使用

    decode 字段：
      slot_mapping    — [num_seqs]，当前 token 写入的 slot
      context_lens    — [num_seqs]，每个 seq 的 KV 总长度
      block_tables    — [num_seqs, max_blocks]
    """
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None
    lin_attn_seq_slots: list | None = None


_CONTEXT = Context()


def get_context() -> Context:
    return _CONTEXT


def set_context(
    is_prefill: bool,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    max_seqlen_q: int = 0,
    max_seqlen_k: int = 0,
    slot_mapping=None,
    context_lens=None,
    block_tables=None,
    lin_attn_seq_slots=None,
):
    global _CONTEXT
    _CONTEXT = Context(
        is_prefill, cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlen_k,
        slot_mapping, context_lens, block_tables,
        lin_attn_seq_slots,
    )


def reset_context():
    """推理步结束后清空 Context，防止下步误读上步数据。"""
    global _CONTEXT
    _CONTEXT = Context()
