import torch

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, reset_context
from nanovllm.utils.loader import load_model

import torch.nn.functional as F
from nanovllm.utils.context import get_context


# ─── 替换注意力为支持 KV cache 的版本 ────────────────────────────────────────

import torch.nn as nn


class AttentionWithKVCache(nn.Module):
    """
    支持 KV cache 的注意力层（Phase 3，朴素实现，无 FlashAttention）。

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
        """
        shape 记号约定：
          N   — prefill 时拉平拼接的全部 token 数（= Σ 各序列 q 长度）
          Ls  — 单条序列的 token 数（prefill 逐序列循环内）
          bs  — decode 时的序列数（每序列仅 1 个 query token）
          H   — query 头数 self.num_heads
          Hkv — KV 头数 self.num_kv_heads（GQA 下 Hkv < H）
          G   — GQA 组数 self.num_kv_groups = H // Hkv
          D   — head_dim
          S   — block_size（每个物理块的 token 容量）

        入参（已是「头维展开」后的 3D 张量）：
          q: [N, H,   D]   k/v: [N, Hkv, D]   （decode 时首维为 bs）
        返回：
          o: [N, H, D]     （decode 时 [bs, H, D]）
        SDPA 要求 4D 布局 [batch, heads, seq, D]，故下面频繁 transpose/unsqueeze。
        """
        context = get_context()

        # 写入 KV cache：把本步算出的 k/v 按 slot_mapping scatter 到分页 cache
        if context.slot_mapping is not None and self.k_cache.numel() > 0:
            self._store_kv(k, v, context.slot_mapping)

        if context.is_prefill:
            # 按序列边界逐条计算注意力，避免不同序列间的跨序列 attend 污染。
            # is_causal=True 的下三角 mask 作用于拼接后的全局 token 序列，
            # 若多条序列拼在一起，后序序列会 attend 到前序序列，导致 KV cache 污染。
            cu_q = context.cu_seqlens_q  # [num_seqs+1]，前缀和，相邻差即各序列 q 长度
            if cu_q is None:
                # 单序列 fallback（无 context 信息时）：整个 N 当一条序列处理
                if self.num_kv_groups > 1:
                    # GQA：把 Hkv 个 KV 头各复制 G 份，对齐到 H 个 query 头
                    k = k.repeat_interleave(self.num_kv_groups, dim=1)  # [N, Hkv, D] → [N, H, D]
                    v = v.repeat_interleave(self.num_kv_groups, dim=1)  # [N, Hkv, D] → [N, H, D]
                # [N, H, D] → transpose → [H, N, D] → unsqueeze(0) → [1, H, N, D]
                q = q.transpose(0, 1).unsqueeze(0)
                k = k.transpose(0, 1).unsqueeze(0)
                v = v.transpose(0, 1).unsqueeze(0)
                o = F.scaled_dot_product_attention(q, k, v, scale=self.scale, is_causal=True)  # [1, H, N, D]
                return o.squeeze(0).transpose(0, 1)  # [1,H,N,D] → [H,N,D] → [N, H, D]
            out_parts = []
            for s in range(cu_q.shape[0] - 1):
                s0, s1 = cu_q[s].item(), cu_q[s + 1].item()  # 第 s 条序列在拼接维上的 [s0, s1)
                q_s = q[s0:s1]                # [Ls, H,   D]
                k_s = k[s0:s1]                # [Ls, Hkv, D]
                v_s = v[s0:s1]                # [Ls, Hkv, D]
                if self.num_kv_groups > 1:
                    k_s = k_s.repeat_interleave(self.num_kv_groups, dim=1)  # → [Ls, H, D]
                    v_s = v_s.repeat_interleave(self.num_kv_groups, dim=1)  # → [Ls, H, D]
                q_t = q_s.transpose(0, 1).unsqueeze(0)  # [Ls,H,D] → [1, H, Ls, D]
                k_t = k_s.transpose(0, 1).unsqueeze(0)  # [1, H, Ls, D]
                v_t = v_s.transpose(0, 1).unsqueeze(0)  # [1, H, Ls, D]
                o_s = F.scaled_dot_product_attention(q_t, k_t, v_t, scale=self.scale, is_causal=True)  # [1, H, Ls, D]
                out_parts.append(o_s.squeeze(0).transpose(0, 1))  # → [Ls, H, D]
            return torch.cat(out_parts, dim=0)  # 拼回 [N, H, D]
        else:
            # decode: 每序列只有 1 个 query token，K/V 从分页 cache 读全部历史
            bs = q.size(0)                          # q: [bs, H, D]
            block_tables = context.block_tables     # [bs, max_blocks]，物理块号，-1 为 padding
            context_lens = context.context_lens     # [bs]，各序列历史 KV 长度（含当前 token）
            outputs = []
            for i in range(bs):
                seq_len = context_lens[i].item()    # 标量 Ls_i
                # 该序列历史占用的块数（向上取整）；k_cache.shape[1] 即 S
                num_blocks_needed = (seq_len + self.k_cache.shape[1] - 1) // self.k_cache.shape[1]
                blocks = block_tables[i, :num_blocks_needed]   # [num_blocks_needed]
                # 逐块取出再拼接：每块 k_cache[b] 形如 [S, Hkv, D]，
                # cat 后 [num_blocks_needed*S, Hkv, D]，截到 [:seq_len] → [Ls_i, Hkv, D]
                k_hist = torch.cat([self.k_cache[b] for b in blocks], dim=0)[:seq_len]
                v_hist = torch.cat([self.v_cache[b] for b in blocks], dim=0)[:seq_len]
                if self.num_kv_groups > 1:
                    k_hist = k_hist.repeat_interleave(self.num_kv_groups, dim=1)  # [Ls_i, Hkv, D] → [Ls_i, H, D]
                    v_hist = v_hist.repeat_interleave(self.num_kv_groups, dim=1)  # [Ls_i, Hkv, D] → [Ls_i, H, D]
                qi = q[i].unsqueeze(1)                  # q[i]:[H, D] → [H, 1, D]（1 个 query token）
                ki = k_hist.transpose(0, 1)             # [Ls_i, H, D] → [H, Ls_i, D]
                vi = v_hist.transpose(0, 1)             # [H, Ls_i, D]
                # SDPA：输入补 batch 维 → q[1,H,1,D] / k,v[1,H,Ls_i,D]，无 causal（单 token attend 全历史）
                oi = F.scaled_dot_product_attention(
                    qi.unsqueeze(0), ki.unsqueeze(0), vi.unsqueeze(0), scale=self.scale
                ).squeeze(0)                            # [1,H,1,D] → [H, 1, D]
                outputs.append(oi.squeeze(1))           # [H, 1, D] → [H, D]
            return torch.stack(outputs, dim=0)          # 堆叠 → [bs, H, D]


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
        """构建模型。Attention 层已在模块级 monkey-patch 为 AttentionWithKVCache。"""
        from nanovllm.models.qwen3 import Qwen3ForCausalLM
        return Qwen3ForCausalLM(hf_config)

    def warmup_model(self):
        """
        运行一次最大批次 prefill，测量 GPU 峰值显存。
        Note：测到的不一定是真实峰值，只是当前给定的条件下的峰值，这里由以下三个参数影响：
            1. max_num_batched_tokens
            2. max_model_len
            3. max_num_seqs
        """
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        config = self.config
        seq_len = min(config.max_num_batched_tokens, config.max_model_len)
        num_seqs = min(config.max_num_batched_tokens // seq_len, config.max_num_seqs)
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        for seq in seqs:
            seq.num_scheduled_tokens = seq_len
        self.run(seqs, True)
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

    def prepare_prefill(self, seqs: list[Sequence]):
        """
        构造 prefill 一步的模型输入与全局 Context。

        多条序列的 token 被「拉平拼接」成一维张量（varlen 布局，无 padding），
        序列边界靠 cu_seqlens 前缀和切分。每个 token 还需算出它在全局 KV cache
        里的写入 slot（slot_mapping），forward 时 Attention 据此 scatter 写入。

        关键量：
          start / end — 本步要处理的 token 在 seq 中的 [start, end) 区间。
                        阶段一 num_cached_tokens 恒为 0，故 start=0、end=整段 prompt。
          seqlen_q    — query 长度（本步新算的 token 数）
          seqlen_k    — key 长度（= end，含此前已缓存部分；阶段一与 seqlen_q 相等）
        """
        input_ids_list = []
        positions_list = []
        cu_seqlens_q = [0]   # query 累计长度前缀和，[num_seqs+1]，相邻差即各 seq 的 q 长度
        cu_seqlens_k = [0]   # key 累计长度前缀和，供变长注意力切分每条序列的 KV 范围
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []    # 每个 query token 写入 KV cache 的绝对槽位（block_id*block_size+offset）

        for seq in seqs:
            start = seq.num_cached_tokens
            seqlen_q = seq.num_scheduled_tokens
            end = start + seqlen_q
            seqlen_k = end

            # 拉平拼接：input_ids/positions 直接 extend，无 padding；位置从 start 开始连续递增
            input_ids_list.extend(seq[start:end])
            positions_list.extend(range(start, end))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)

            # warmup 阶段 KV cache 尚未分配、seq 无 block_table：用 -1 占位，
            # Attention._store_kv 见 -1 / 空 cache 会跳过写入。
            if not seq.block_table:
                slot_mapping.extend([-1] * seqlen_q)
                continue
            # 把逻辑区间 [start, end) 映射到物理 slot。token 可能横跨多个物理块，
            # 故逐块计算该块覆盖的 slot 子区间再拼接。
            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size   # 向上取整，开区间右端
            for i in range(start_block, end_block):
                slot_start = seq.block_table[i] * self.block_size
                if i == start_block:
                    # 首块可能从块中间开始（start 未对齐块边界）
                    slot_start += start % self.block_size
                if i != end_block - 1:
                    # 中间块：整块写满
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:
                    # 末块：只写到 end，end - i*block_size 为末块内已用 token 数
                    slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
                slot_mapping.extend(range(slot_start, slot_end))

        # pin_memory + non_blocking：锁页内存上异步 H2D 拷贝，与后续 CPU 工作重叠
        input_ids = torch.tensor(input_ids_list, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions_list, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        sm = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_q, cu_k, max_seqlen_q, max_seqlen_k, sm)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        """
        构造 decode 一步的模型输入与全局 Context。

        decode 每条序列只输入「最后一个 token」（上一步刚生成的），算它的 KV 并写入
        cache，再对全部历史 KV 做注意力。因此 query 长度恒为 1，batch 维即序列数。

        与 prefill 的关键差异：
          - 输入是 1 个 token（seq.last_token），不是整段区间
          - 不传 cu_seqlens；改传 context_lens（各序列 KV 总长）+ block_tables，
            供 Attention 从分页 cache 里按块取回历史 K/V
        """
        input_ids_list = []
        positions_list = []
        slot_mapping = []
        context_lens = []

        for seq in seqs:
            input_ids_list.append(seq.last_token)        # 仅输入最后一个 token
            positions_list.append(len(seq) - 1)          # 其位置 = 序列当前长度 - 1（0-based）
            context_lens.append(len(seq))                # 历史 KV 长度，含本 token
            # 本 token 写入末块内偏移 last_block_num_tokens-1 处；
            # may_append 已在调度时确保末块容得下它（len%block_size==1 时新开块）。
            slot_mapping.append(
                seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1
            )

        input_ids = torch.tensor(input_ids_list, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions_list, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        sm = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cl = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        # block_tables 是规整二维张量 [num_seqs, max_blocks]，短序列用 -1 右侧补齐对齐，
        # Attention 按 context_lens 只取前若干块，padding 的 -1 不会被读到。
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
