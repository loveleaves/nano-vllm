import gc

import torch
import torch.distributed as dist

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.input_batch import InputBatch
from nanovllm.engine.kv_cache import FullAttentionSpec
from nanovllm.attention import Attention
from nanovllm.sample import Sampler, SamplingMetadata
from nanovllm.utils.context import AttentionMetadata
from nanovllm.utils.loader import load_model
from nanovllm.models.registry import resolve_model_cls


class ModelRunner:
    """
    GPU 推理执行器（Phase 4：完整版，含 TP + CUDA Graph）。

    职责：
      1. 初始化 NCCL 进程组、设置 CUDA 设备
      2. 构建并加载 Qwen3 模型
      3. warmup → 估算显存峰值 → allocate_kv_cache
      4. 捕获 CUDA graph（decode 阶段加速）
      5. 提供 run() 接口（prefill + decode 统一入口）
      6. rank>0 进入 loop()，等待 rank 0 通过 SharedMemory 广播指令

    多进程通信：
      SharedMemory（nanovllm）：rank 0 写入 pickle 数据，rank i 读取
      multiprocessing.Event：rank 0 设置 event 通知 rank i 有新数据
      NCCL all_reduce：GPU 间实际张量同步

    CUDA Graph：
      graph_bs = [1,2,4,8,16,...,512]，为每个 batch size 录制独立 graph。
      共享同一 CUDA memory pool，减少显存碎片。
    """

    def __init__(self, config: Config, rank: int = 0):
        """纯 GPU 执行器：NCCL init + 建模 + warmup + KV cache + CUDA graph。

        多进程 RPC（loop/recv/broadcast）已抽到 engine.worker.Worker + engine.rpc.ShmTransport，
        本类不再涉及进程间通信。
        """
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank

        dist.init_process_group("nccl", "tcp://localhost:2333",
                                world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)

        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.dtype)
        torch.set_default_device("cuda")

        # 动态解析架构 → 模型类（惰性导入）：从 HF config 的 architectures 字段查注册表，
        # 命中后才 import 对应模块，避免主进程过早初始化 CUDA / 导入全部模型。
        architectures = getattr(hf_config, "architectures", None) or ["Qwen3ForCausalLM"]
        model_cls, _arch = resolve_model_cls(architectures)
        self.model = model_cls(hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()
        # 按请求 seed 持久化的随机数生成器（seq_id → Generator），跨步续流
        self.generators: dict[int, torch.Generator] = {}
        # 异步调度的两槽采样状态：inflight=上一步（待回收 + 本步前向源）/ pending=本步刚算出
        self._ai = None   # dict(sampled[GPU], ordered, index{seq_id→row}, logprobs) | None
        self._ap = None

        # 跨步常驻的输入批：持久行槽位 + 增量块表 + 每步展开缓冲（对齐 V1 InputBatch）
        max_num_blocks_per_req = (config.max_model_len + self.block_size - 1) // self.block_size
        self.input_batch = InputBatch(
            max_num_reqs=config.max_num_seqs,
            max_num_blocks_per_req=max_num_blocks_per_req,
            max_num_batched_tokens=config.max_num_batched_tokens,
            block_size=self.block_size, device="cuda", pin_memory=True,
        )

        self.warmup_model()
        self.allocate_kv_cache()

        if not self.enforce_eager:
            self.capture_cudagraph()

        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

    def exit(self):
        """释放 GPU 资源（graph / KV cache / 模型 / 进程组）。RPC 传输的关闭由 Worker 负责。

        显式释放显存并 empty_cache，使同进程可干净重建引擎（否则残留显存会让下个引擎
        的 num_kvcache_blocks 估算 ≤ 0 而断言失败）。退出后本 runner 不应再被使用。
        """
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        # 解除 Attention 层对 KV cache 切片的引用，再释放 KV cache / 模型 / 输入批缓冲
        for module in self.model.modules():
            if isinstance(module, Attention):
                module.k_cache = module.v_cache = None
        self.kv_cache = None
        self.cpu_kv_cache = None
        self._ai = self._ap = None
        self.model = None
        self.input_batch = None
        # nn.Module 间存在引用环，需 gc.collect() 才能释放模型权重显存，否则 empty_cache 无效
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        dist.destroy_process_group()

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
        self.run(seqs)
        self.input_batch.clear()   # 释放 warmup 占用的行，真正推理从空批开始
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        """根据剩余显存计算并分配 KV cache 张量（块字节/块数计算由 KVCacheSpec 承担）。"""
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim",
                           hf_config.hidden_size // hf_config.num_attention_heads)
        num_layers = hf_config.num_hidden_layers

        # KVCacheSpec 封装单层单块字节数与"显存 → 块数"反推（对齐 V1）
        self.kv_cache_spec = FullAttentionSpec(
            block_size=self.block_size, num_kv_heads=num_kv_heads,
            head_dim=head_dim, dtype=hf_config.dtype,
        )
        available = int(total * config.gpu_memory_utilization - used - peak + current)
        config.num_kvcache_blocks = self.kv_cache_spec.num_blocks_for_memory(
            available, num_layers)
        assert config.num_kvcache_blocks > 0

        self.kv_cache = torch.empty(
            2, num_layers, *self.kv_cache_spec.kv_cache_shape(config.num_kvcache_blocks)[1:],
        )
        layer_id = 0
        for module in self.model.modules():
            if isinstance(module, Attention):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

        # CPU swap 区（pinned 内存，按 num_swap_blocks 分配；抢占换出/换入的落脚点）
        self.cpu_kv_cache = None
        if config.num_swap_blocks > 0:
            self.cpu_kv_cache = torch.empty(
                2, num_layers,
                *self.kv_cache_spec.kv_cache_shape(config.num_swap_blocks)[1:],
                device="cpu", pin_memory=True,
            )

    @torch.inference_mode()
    def swap_out(self, blocks: list[tuple[int, int]]):
        """抢占换出：把 GPU 块 D2H 拷到 CPU swap 槽（blocks=[(gpu_block_id, swap_slot)]）。
        须在 execute_model 之前调用——此时块内仍是被换出序列的旧 KV。"""
        if not blocks:
            return
        gpu_ids = torch.tensor([g for g, _ in blocks], device="cuda")
        slots = torch.tensor([s for _, s in blocks], device="cpu")
        # kv_cache/cpu_kv_cache 形状 [2, L, num_blocks, ...]，按 block 维 gather/scatter
        gathered = self.kv_cache[:, :, gpu_ids].to("cpu")   # D2H（同步，落地后再写）
        self.cpu_kv_cache[:, :, slots] = gathered

    @torch.inference_mode()
    def swap_in(self, blocks: list[tuple[int, int]]):
        """换回：把 CPU swap 槽 H2D 拷到新分配的 GPU 块（blocks=[(gpu_block_id, swap_slot)]）。"""
        if not blocks:
            return
        gpu_ids = torch.tensor([g for g, _ in blocks], device="cuda")
        slots = torch.tensor([s for _, s in blocks], device="cpu")
        gathered = self.cpu_kv_cache[:, :, slots].to("cuda")   # H2D（同步）
        self.kv_cache[:, :, gpu_ids] = gathered

    def execute_swap(self, blocks_to_swap_in, blocks_to_swap_out):
        """成对执行本步 KV 搬运：先 swap_out（读旧 KV）再 swap_in。"""
        self.swap_out(blocks_to_swap_out)
        self.swap_in(blocks_to_swap_in)

    def _to_cuda(self, data, dtype) -> torch.Tensor:
        return torch.tensor(data, dtype=dtype, pin_memory=True).cuda(non_blocking=True)

    def prepare_sample(self, seqs: list[Sequence]) -> SamplingMetadata:
        """从行序序列构造结构化 SamplingMetadata（仅 rank0 调用）。

        整批无某项配置时该字段置 None / no_penalties，Sampler 据此整段跳过，
        使 greedy / 纯温度采样的常见路径零额外开销。
        """
        vocab_size = self.config.hf_config.vocab_size
        eps = 1e-5

        temps = [seq.temperature for seq in seqs]
        greedy = [t < eps for t in temps]
        all_greedy = all(greedy)
        all_random = not any(greedy)
        temperature = self._to_cuda(temps, torch.float32)

        tp = [seq.top_p for seq in seqs]
        top_p = None if all(x >= 1.0 for x in tp) else self._to_cuda(tp, torch.float32)

        # top_k：<=0 或 >=vocab 视为关闭，关闭行填 vocab_size（apply_top_k_only 不掩码）
        tk = [k if 0 < k < vocab_size else vocab_size for k in (seq.top_k for seq in seqs)]
        top_k = None if all(k == vocab_size for k in tk) else self._to_cuda(tk, torch.int32)

        mp = [seq.min_p for seq in seqs]
        min_p = None if all(x <= 0.0 for x in mp) else self._to_cuda(mp, torch.float32)

        freq = [seq.frequency_penalty for seq in seqs]
        pres = [seq.presence_penalty for seq in seqs]
        rep = [seq.repetition_penalty for seq in seqs]
        no_penalties = all(f == 0.0 and p == 0.0 and r == 1.0
                           for f, p, r in zip(freq, pres, rep))

        # bad_words：行 → 该请求的禁止 token 序列
        bad_words = {i: seq.bad_words_token_ids for i, seq in enumerate(seqs)
                     if seq.bad_words_token_ids}
        bad_words = bad_words or None

        # 惩罚需 prompt+output 历史；bad_words 仅需 output 历史
        prompt_ids = output_ids = None
        freq_t = pres_t = rep_t = None
        if not no_penalties:
            prompt_ids = [seq.prompt_token_ids for seq in seqs]
            output_ids = [seq.completion_token_ids for seq in seqs]
            freq_t = self._to_cuda(freq, torch.float32)
            pres_t = self._to_cuda(pres, torch.float32)
            rep_t = self._to_cuda(rep, torch.float32)
        elif bad_words is not None:
            output_ids = [seq.completion_token_ids for seq in seqs]

        # 持久 generator：按请求 seed 建一次，跨步续流（行 → generator）
        generators = {}
        for row, seq in enumerate(seqs):
            if seq.seed is None:
                continue
            gen = self.generators.get(seq.seq_id)
            if gen is None:
                gen = torch.Generator(device="cuda")
                gen.manual_seed(seq.seed)
                self.generators[seq.seq_id] = gen
            generators[row] = gen
        generators = generators or None

        lp = [seq.logprobs for seq in seqs if seq.logprobs is not None]
        max_num_logprobs = min(max(lp), vocab_size - 1) if lp else None

        # LogitsProcessor 框架字段：logit_bias / min_tokens / 引导 grammar
        logit_bias = {i: seq.logit_bias for i, seq in enumerate(seqs) if seq.logit_bias}
        min_tokens = {i: seq.min_tokens for i, seq in enumerate(seqs) if seq.min_tokens}
        grammars = {i: seq.grammar for i, seq in enumerate(seqs)
                    if getattr(seq, "grammar", None) is not None}
        # min_tokens / grammar 需 output 历史长度判定；确保 output_ids 已就绪
        if (min_tokens or grammars) and output_ids is None:
            output_ids = [seq.completion_token_ids for seq in seqs]
        eos = self.config.eos if self.config.eos != -1 else None

        return SamplingMetadata(
            temperature=temperature, all_greedy=all_greedy, all_random=all_random,
            top_p=top_p, top_k=top_k, min_p=min_p,
            generators=generators, bad_words_token_ids=bad_words,
            no_penalties=no_penalties, prompt_token_ids=prompt_ids,
            output_token_ids=output_ids, frequency_penalties=freq_t,
            presence_penalties=pres_t, repetition_penalties=rep_t,
            logit_bias=logit_bias or None, min_tokens=min_tokens or None,
            eos_token_id=eos, grammars=grammars or None,
            max_num_logprobs=max_num_logprobs,
        )

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor,
                  attn_md: AttentionMetadata):
        """
        执行模型 forward，两种路径：
          1. 含 prefill chunk 的混合批 / enforce_eager / bs>512 → eager
          2. 纯 decode 批（attn_md.is_decode_only，每 seq query 长度 1）且 bs≤512
             → CUDA graph replay（零 Python overhead）

        graph 捕获时 max_seqlen_k 取 max_model_len（高估对 varlen kernel 安全，
        已验证数值一致），故同一 graph 可服务任意 KV 长度的 decode。
        """
        if not attn_md.is_decode_only or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(
                self.model(input_ids, positions, attn_md), attn_md)

        bs = input_ids.size(0)
        graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
        gv = self.graph_vars
        gv["input_ids"][:bs] = input_ids
        gv["positions"][:bs] = positions
        gv["slot_mapping"].fill_(-1)
        gv["slot_mapping"][:bs] = attn_md.slot_mapping
        gv["cu_seqlens_k"].zero_()
        gv["cu_seqlens_k"][:bs + 1] = attn_md.cu_seqlens_k
        gv["block_tables"][:bs, :attn_md.block_table.size(1)] = attn_md.block_table
        graph.replay()
        return self.model.compute_logits(gv["outputs"][:bs], attn_md)

    @torch.inference_mode()
    def verify_spec(self, seq: Sequence, num_drafts: int) -> list[int]:
        """投机解码验证（GPU，仅 UniProc）：目标模型并行前向 num_drafts+1 个位置，
        返回各位置贪心 argmax（共 num_drafts+1 个 token）。

        调用前 seq 已投机追加 num_drafts 个草案 token、块表覆盖到末位。前向 query 为
        [token@(L0-1), 草案0..草案k-1]（k+1 个位置），写入它们的 KV（被拒绝位的 KV 由后续
        步覆盖），并对**全部** k+1 个位置取 lm_head（不做末位聚合）后 argmax。
        """
        # 位置算术：调用前已把 k 个草案 append 到 seq，故 num_tokens = L0 + k（L0=投机前长度）。
        # 要预测位置 L0..L0+k（共 k+1 个），需以位置 L0-1..L0+k-1 的 token 作为 query 前向：
        #   query[0]=原最后一个真 token(位置 L0-1) → 预测 L0；query[1..k]=草案(位置 L0..L0+k-1) → 预测 L0+1..L0+k
        k = num_drafts
        n_kv = seq.num_tokens                       # = L0 + k（已含草案）
        start = n_kv - (k + 1)                      # L0 - 1：最后一个真 token 的位置
        positions = list(range(start, n_kv))        # k+1 个 query 位置：L0-1 .. L0+k-1
        block_size = self.block_size
        bt = seq.block_table
        slots = [bt[p // block_size] * block_size + (p % block_size) for p in positions]

        dev = "cuda"
        input_ids = torch.tensor(seq.token_ids[start:n_kv], dtype=torch.int64, device=dev)
        pos_t = torch.tensor(positions, dtype=torch.int64, device=dev)
        attn_md = AttentionMetadata(
            query_start_loc=torch.tensor([0, k + 1], dtype=torch.int32, device=dev),
            cu_seqlens_k=torch.tensor([0, n_kv], dtype=torch.int32, device=dev),
            max_query_len=k + 1, max_seq_len=n_kv,
            slot_mapping=torch.tensor(slots, dtype=torch.int64, device=dev),
            block_table=torch.tensor([bt], dtype=torch.int32, device=dev),
        )
        hidden = self.model(input_ids, pos_t, attn_md)
        # 全位置 logits（attn_md=None → lm_head 不做末位聚合）
        logits = self.model.lm_head(hidden, None)
        return logits.argmax(dim=-1).tolist()

    def run(self, seqs: list[Sequence], finished_seq_ids: set[int] | None = None):
        """单步推理接口。统一连续批，经常驻 InputBatch 增量构造输入。

        finished_seq_ids — 上一步结束 / 本步被抢占的 seq_id，用于回收其持久行槽位。
        模型按行序前向/采样，得到的行序 token 再按 seq_id 映射回入参 seqs 的顺序返回，
        使上层 update_from_output 可直接与 scheduled_seqs zip。

        返回 (token_ids, step_logprobs)（rank>0 返回 None）。step_logprobs 为按 seqs 对齐的
        list[dict[int,float] | None]，整批无 logprobs 请求时为 None。
        """
        if finished_seq_ids:                       # 回收已结束请求的持久 generator
            for sid in finished_seq_ids:
                self.generators.pop(sid, None)
        self.input_batch.update(seqs, finished_seq_ids)
        input_ids, positions, attn_md, ordered = self.input_batch.make_inputs(seqs)
        logits = self.run_model(input_ids, positions, attn_md)
        if self.rank != 0:
            return None
        sampling_metadata = self.prepare_sample(ordered)
        sampler_output = self.sampler(logits, sampling_metadata)
        row_tokens = sampler_output.sampled_token_ids.tolist()
        tok_by_id = {seq.seq_id: tok for seq, tok in zip(ordered, row_tokens)}
        token_ids = [tok_by_id[seq.seq_id] for seq in seqs]

        # logprobs：按 seq_id 映射回入参顺序；未请求 logprobs 的 seq 置 None（整批未请求则为 None）
        step_logprobs = None
        lt = sampler_output.logprobs_tensors
        if lt is not None:
            ids = lt.logprob_token_ids.tolist()   # [n, 1+k]
            vals = lt.logprobs.tolist()            # [n, 1+k]
            lp_by_id = {ordered[r].seq_id: dict(zip(ids[r], vals[r]))
                        for r in range(len(ordered))}
            step_logprobs = [lp_by_id[s.seq_id] if s.logprobs is not None else None
                             for s in seqs]
        return token_ids, step_logprobs

    # ── 异步调度：非阻塞下发 + 采样 token 留 GPU 跨步前向 ─────────────────────────
    @torch.inference_mode()
    def execute_model_async(self, seqs, finished_seq_ids=None):
        """非阻塞下发一步推理：构造输入时用上一步留在 GPU 的采样 token 前向回填 decode 行
        （避免 D2H 同步），前向 + 采样后把采样张量暂存到 pending 槽，**不** .tolist()。"""
        if finished_seq_ids:
            for sid in finished_seq_ids:
                self.generators.pop(sid, None)
        self.input_batch.update(seqs, finished_seq_ids)
        input_ids, positions, attn_md, ordered = self.input_batch.make_inputs(seqs)

        # 前向：把上一步采样 token（GPU 张量）就地写入本步 decode 行的输入位（无 D2H）
        if self._ai is not None:
            prev_sampled, prev_index = self._ai["sampled"], self._ai["index"]
            cu = self.input_batch.query_start_loc.np
            dst, src = [], []
            for r, seq in enumerate(ordered):
                # 仅"喂生成 token"的 decode 行需前向（q==1 且喂的是已生成位而非 prompt 位）
                if seq.num_scheduled_tokens == 1 \
                        and seq.num_cached_tokens >= seq.num_prompt_tokens:
                    idx = prev_index.get(seq.seq_id)
                    if idx is not None:
                        dst.append(int(cu[r]))
                        src.append(idx)
            if dst:
                dst_t = torch.tensor(dst, device="cuda")
                src_t = torch.tensor(src, device="cuda")
                input_ids[dst_t] = prev_sampled[src_t]

        logits = self.run_model(input_ids, positions, attn_md)
        sampling_metadata = self.prepare_sample(ordered)
        sampler_output = self.sampler(logits, sampling_metadata)
        self._ap = {
            "sampled": sampler_output.sampled_token_ids,   # GPU 张量 [n]，不同步
            "ordered": ordered,
            "index": {seq.seq_id: r for r, seq in enumerate(ordered)},
            "logprobs": sampler_output.logprobs_tensors,
        }

    @torch.inference_mode()
    def resolve_inflight(self):
        """D2H 同步取回 inflight（上一步）结果：返回 (tok_by_id, lp_by_id)。"""
        ai = self._ai
        ordered = ai["ordered"]
        row_tokens = ai["sampled"].tolist()
        tok_by_id = {seq.seq_id: tok for seq, tok in zip(ordered, row_tokens)}
        lp_by_id = None
        lt = ai["logprobs"]
        if lt is not None:
            ids = lt.logprob_token_ids.tolist()
            vals = lt.logprobs.tolist()
            lp_by_id = {ordered[r].seq_id: dict(zip(ids[r], vals[r]))
                        for r in range(len(ordered)) if ordered[r].logprobs is not None}
        return tok_by_id, lp_by_id

    def promote_async(self):
        """把本步刚算出的采样张量提升为 inflight（下一步前向源 + 下一步待回收）。"""
        self._ai = self._ap
        self._ap = None

    @torch.inference_mode()
    def capture_cudagraph(self):
        """
        为 decode 阶段各 batch size 录制 CUDA graph。

        录制顺序：从大到小（第一次创建 pool，后续共享）。
        静态张量在录制期间预分配，replay 时修改数据即可。
        """
        config = self.config
        hf_config = config.hf_config
        max_bs = min(config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size

        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        # decode：每 seq query 长度恒为 1，cu_seqlens_q 即 arange（常量，replay 不更新）
        cu_seqlens_q = torch.arange(max_bs + 1, dtype=torch.int32)
        cu_seqlens_k = torch.zeros(max_bs + 1, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)

        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            # max_seq_len 取 max_model_len（高估安全），使同一 graph 服务任意 KV 长度
            attn_md = AttentionMetadata(
                query_start_loc=cu_seqlens_q[:bs + 1],
                cu_seqlens_k=cu_seqlens_k[:bs + 1],
                max_query_len=1, max_seq_len=config.max_model_len,
                slot_mapping=slot_mapping[:bs], block_table=block_tables[:bs],
            )
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs], attn_md)  # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs], attn_md)
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()

        self.graph_vars = dict(
            input_ids=input_ids, positions=positions,
            slot_mapping=slot_mapping, cu_seqlens_k=cu_seqlens_k,
            block_tables=block_tables, outputs=outputs,
        )
