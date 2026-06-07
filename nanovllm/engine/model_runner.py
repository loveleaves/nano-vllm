import torch

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, reset_context
from nanovllm.utils.loader import load_model
from nanovllm.models.qwen3 import Qwen3ForCausalLM

# Phase 3 注意力层：扩展 Attention 以支持 KV cache（朴素版，无 FlashAttention）
import torch.nn.functional as F
from nanovllm.layers.attention import Attention
from nanovllm.utils.context import get_context


class Attention:
    """
    Phase 3 重新实现的 Attention，支持 KV cache 读写（无 FlashAttention，朴素实现）。
    注：此处通过 monkey-patch 替换 nanovllm.layers.attention.Attention。
    Phase 4 将用 FlashAttention 替换。
    """
    pass


# ─── 替换注意力为支持 KV cache 的版本 ────────────────────────────────────────

import torch.nn as nn


class AttentionWithKVCache(nn.Module):
    """
    支持 KV cache 的注意力层（Phase 3，朴素实现）。

    k_cache / v_cache：
      初始为空张量，由 ModelRunner.allocate_kv_cache 替换为全局 KV cache 的对应层切片。
      形状：[num_blocks, block_size, num_kv_heads, head_dim]

    forward 分两路：
      prefill: 所有 token 做 causal self-attention（朴素实现，不用 FlashAttention）
      decode:  从 KV cache 读历史，只计算 1 个 query token 的注意力
    """

    def __init__(self, num_heads: int, head_dim: int, scale: float, num_kv_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.num_kv_groups = num_heads // num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def _store_kv(self, k: torch.Tensor, v: torch.Tensor, slot_mapping: torch.Tensor):
        """将 k/v 写入 KV cache 的指定 slot（朴素 Python scatter）。"""
        if self.k_cache.numel() == 0:
            return
        block_size = self.k_cache.shape[1]
        for idx, slot in enumerate(slot_mapping):
            slot = slot.item()
            if slot < 0:
                continue
            block_id = slot // block_size
            offset = slot % block_size
            self.k_cache[block_id, offset] = k[idx]
            self.v_cache[block_id, offset] = v[idx]

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        context = get_context()

        # 写入 KV cache
        if context.slot_mapping is not None and self.k_cache.numel() > 0:
            self._store_kv(k, v, context.slot_mapping)

        if context.is_prefill:
            # GQA 扩展
            if self.num_kv_groups > 1:
                k = k.repeat_interleave(self.num_kv_groups, dim=1)
                v = v.repeat_interleave(self.num_kv_groups, dim=1)
            # [1, num_heads, N, head_dim]
            q_t = q.transpose(0, 1).unsqueeze(0)
            k_t = k.transpose(0, 1).unsqueeze(0)
            v_t = v.transpose(0, 1).unsqueeze(0)
            o = F.scaled_dot_product_attention(q_t, k_t, v_t, scale=self.scale, is_causal=True)
            return o.squeeze(0).transpose(0, 1)
        else:
            # decode: 从 KV cache 读历史 k/v
            bs = q.size(0)
            block_tables = context.block_tables    # [bs, max_blocks]
            context_lens = context.context_lens    # [bs]
            outputs = []
            for i in range(bs):
                seq_len = context_lens[i].item()
                num_blocks_needed = (seq_len + self.k_cache.shape[1] - 1) // self.k_cache.shape[1]
                blocks = block_tables[i, :num_blocks_needed]
                # 收集历史 k/v：[seq_len, num_kv_heads, head_dim]
                k_hist = torch.cat([self.k_cache[b] for b in blocks], dim=0)[:seq_len]
                v_hist = torch.cat([self.v_cache[b] for b in blocks], dim=0)[:seq_len]
                # GQA 扩展
                if self.num_kv_groups > 1:
                    k_hist = k_hist.repeat_interleave(self.num_kv_groups, dim=1)
                    v_hist = v_hist.repeat_interleave(self.num_kv_groups, dim=1)
                # q[i]: [num_heads, head_dim]
                qi = q[i].unsqueeze(1)                  # [num_heads, 1, head_dim]
                # k_hist: [seq_len, num_heads, head_dim] → [num_heads, seq_len, head_dim]
                ki = k_hist.transpose(0, 1)
                vi = v_hist.transpose(0, 1)
                # scaled dot product: [num_heads, 1, head_dim]
                oi = F.scaled_dot_product_attention(
                    qi.unsqueeze(0), ki.unsqueeze(0), vi.unsqueeze(0), scale=self.scale
                ).squeeze(0)                            # [num_heads, 1, head_dim]
                outputs.append(oi.squeeze(1))           # [num_heads, head_dim]
            return torch.stack(outputs, dim=0)          # [bs, num_heads, head_dim]


# 运行时 patch：将 Qwen3Attention 中的 self.attn 替换为 AttentionWithKVCache
import nanovllm.layers.attention as _attn_module
_attn_module.Attention = AttentionWithKVCache


class ModelRunner:
    """
    GPU 推理执行器（Phase 3：单进程，无 TP，无 CUDA graph）。

    职责：
      1. 初始化 CUDA 设备
      2. 构建并加载 Qwen3 模型
      3. warmup → 估算显存峰值 → allocate_kv_cache
      4. 提供 run() 接口（prefill + decode）
    """

    def __init__(self, config: Config):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size

        torch.cuda.set_device(0)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.dtype)
        torch.set_default_device("cuda")

        # 重新构建模型，使用 AttentionWithKVCache 层
        # 由于 monkey-patch 已在模块级执行，直接使用 Qwen3ForCausalLM 即可
        # 但需要替换已构建的 attn 实例
        self.model = self._build_model(hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()

        self.warmup_model()
        self.allocate_kv_cache()

        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

    def _build_model(self, hf_config):
        """构建模型，并将所有 Attention 层替换为 AttentionWithKVCache。"""
        from nanovllm.models.qwen3 import (
            Qwen3ForCausalLM, Qwen3Attention as _Qwen3Attention,
        )
        model = Qwen3ForCausalLM(hf_config)
        # 替换所有 Attention 层为 AttentionWithKVCache
        for module in model.modules():
            if isinstance(module, _Qwen3Attention):
                old_attn = module.attn
                module.attn = AttentionWithKVCache(
                    old_attn.num_heads, old_attn.head_dim,
                    old_attn.scale, old_attn.num_kv_heads,
                )
        return model

    def warmup_model(self):
        """运行一次最大批次 prefill，测量 GPU 峰值显存。"""
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        config = self.config
        seq_len = min(config.max_num_batched_tokens, config.max_model_len)
        num_seqs = min(config.max_num_batched_tokens // seq_len, config.max_num_seqs)
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        for seq in seqs:
            seq.num_scheduled_tokens = seq_len
        self._run_prefill_eager(seqs)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        """根据剩余显存计算并分配 KV cache 张量。"""
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads
        head_dim = getattr(hf_config, "head_dim",
                           hf_config.hidden_size // hf_config.num_attention_heads)
        block_bytes = (2 * hf_config.num_hidden_layers * self.block_size *
                       num_kv_heads * head_dim * hf_config.dtype.itemsize)
        config.num_kvcache_blocks = int(
            total * config.gpu_memory_utilization - used - peak + current
        ) // block_bytes
        assert config.num_kvcache_blocks > 0, "显存不足以分配 KV cache"

        self.kv_cache = torch.empty(
            2, hf_config.num_hidden_layers, config.num_kvcache_blocks,
            self.block_size, num_kv_heads, head_dim,
        )
        # 将 kv_cache 各层切片绑定到对应 AttentionWithKVCache 模块
        layer_id = 0
        for module in self.model.modules():
            if isinstance(module, AttentionWithKVCache):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def _run_prefill_eager(self, seqs: list[Sequence]):
        """准备 prefill 输入并执行 forward（不采样）。"""
        input_ids_list = []
        positions_list = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []

        for seq in seqs:
            start = seq.num_cached_tokens
            seqlen_q = seq.num_scheduled_tokens
            end = start + seqlen_q
            seqlen_k = end

            input_ids_list.extend(seq[start:end] if seq.token_ids else [0] * seqlen_q)
            positions_list.extend(range(start, end))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)

            if not seq.block_table:
                slot_mapping.extend([-1] * seqlen_q)
                continue
            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size
            for i in range(start_block, end_block):
                slot_start = seq.block_table[i] * self.block_size
                if i == start_block:
                    slot_start += start % self.block_size
                if i != end_block - 1:
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:
                    slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
                slot_mapping.extend(range(slot_start, slot_end))

        input_ids = torch.tensor(input_ids_list, dtype=torch.int64).cuda()
        positions = torch.tensor(positions_list, dtype=torch.int64).cuda()
        cu_q = torch.tensor(cu_seqlens_q, dtype=torch.int32).cuda()
        cu_k = torch.tensor(cu_seqlens_k, dtype=torch.int32).cuda()
        sm = torch.tensor(slot_mapping, dtype=torch.int32).cuda() if slot_mapping else None
        set_context(True, cu_q, cu_k, max_seqlen_q, max_seqlen_k, sm)
        with torch.inference_mode():
            hidden = self.model(input_ids, positions)
        reset_context()
        return hidden

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids_list = []
        positions_list = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []

        for seq in seqs:
            start = seq.num_cached_tokens
            seqlen_q = seq.num_scheduled_tokens
            end = start + seqlen_q
            seqlen_k = end

            input_ids_list.extend(seq[start:end])
            positions_list.extend(range(start, end))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)

            if not seq.block_table:
                slot_mapping.extend([-1] * seqlen_q)
                continue
            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size
            for i in range(start_block, end_block):
                slot_start = seq.block_table[i] * self.block_size
                if i == start_block:
                    slot_start += start % self.block_size
                if i != end_block - 1:
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:
                    slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
                slot_mapping.extend(range(slot_start, slot_end))

        input_ids = torch.tensor(input_ids_list, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions_list, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        sm = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_q, cu_k, max_seqlen_q, max_seqlen_k, sm)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids_list = []
        positions_list = []
        slot_mapping = []
        context_lens = []

        for seq in seqs:
            input_ids_list.append(seq.last_token)
            positions_list.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(
                seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1
            )

        input_ids = torch.tensor(input_ids_list, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions_list, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        sm = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cl = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        max_len = max(len(seq.block_table) for seq in seqs)
        bt = [[*seq.block_table, *[-1] * (max_len - len(seq.block_table))] for seq in seqs]
        bt = torch.tensor(bt, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(False, slot_mapping=sm, context_lens=cl, block_tables=bt)
        return input_ids, positions

    @torch.inference_mode()
    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        """单步推理接口。"""
        if is_prefill:
            input_ids, positions = self.prepare_prefill(seqs)
        else:
            input_ids, positions = self.prepare_decode(seqs)
        temperatures = torch.tensor([seq.temperature for seq in seqs],
                                    dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        hidden = self.model(input_ids, positions)
        logits = self.model.compute_logits(hidden)
        token_ids = self.sampler(logits, temperatures).tolist()
        reset_context()
        return token_ids
