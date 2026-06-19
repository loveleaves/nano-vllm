import torch
import torch.distributed as dist

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.input_batch import InputBatch
from nanovllm.engine.kv_cache import FullAttentionSpec
from nanovllm.layers.attention import Attention
from nanovllm.layers.sample import Sampler, SamplingMetadata
from nanovllm.utils.context import AttentionMetadata
from nanovllm.utils.loader import load_model
from nanovllm.models.qwen3 import Qwen3ForCausalLM


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

        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()

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
        """释放 GPU 资源（graph/进程组）。RPC 传输的关闭由 Worker 负责。"""
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
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

        freq = [seq.frequency_penalty for seq in seqs]
        pres = [seq.presence_penalty for seq in seqs]
        rep = [seq.repetition_penalty for seq in seqs]
        no_penalties = all(f == 0.0 and p == 0.0 and r == 1.0
                           for f, p, r in zip(freq, pres, rep))
        prompt_ids = output_ids = None
        freq_t = pres_t = rep_t = None
        if not no_penalties:
            prompt_ids = [seq.prompt_token_ids for seq in seqs]
            output_ids = [seq.completion_token_ids for seq in seqs]
            freq_t = self._to_cuda(freq, torch.float32)
            pres_t = self._to_cuda(pres, torch.float32)
            rep_t = self._to_cuda(rep, torch.float32)

        lp = [seq.logprobs for seq in seqs if seq.logprobs is not None]
        max_num_logprobs = min(max(lp), vocab_size - 1) if lp else None

        return SamplingMetadata(
            temperature=temperature, all_greedy=all_greedy, all_random=all_random,
            top_p=top_p, top_k=top_k,
            no_penalties=no_penalties, prompt_token_ids=prompt_ids,
            output_token_ids=output_ids, frequency_penalties=freq_t,
            presence_penalties=pres_t, repetition_penalties=rep_t,
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

    def run(self, seqs: list[Sequence],
            finished_seq_ids: set[int] | None = None) -> list[int] | None:
        """单步推理接口。统一连续批，经常驻 InputBatch 增量构造输入。

        finished_seq_ids — 上一步结束 / 本步被抢占的 seq_id，用于回收其持久行槽位。
        模型按行序前向/采样，得到的行序 token 再按 seq_id 映射回入参 seqs 的顺序返回，
        使上层 update_from_output 可直接与 scheduled_seqs zip。
        """
        self.input_batch.update(seqs, finished_seq_ids)
        input_ids, positions, attn_md, ordered = self.input_batch.make_inputs(seqs)
        logits = self.run_model(input_ids, positions, attn_md)
        if self.rank != 0:
            return None
        sampling_metadata = self.prepare_sample(ordered)
        sampler_output = self.sampler(logits, sampling_metadata)
        row_tokens = sampler_output.sampled_token_ids.tolist()
        tok_by_id = {seq.seq_id: tok for seq, tok in zip(ordered, row_tokens)}
        return [tok_by_id[seq.seq_id] for seq in seqs]

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
