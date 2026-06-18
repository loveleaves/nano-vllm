import torch
import torch.distributed as dist

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.layers.attention import Attention
from nanovllm.layers.sampler import Sampler
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
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        """根据剩余显存计算并分配 KV cache 张量。"""
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim",
                           hf_config.hidden_size // hf_config.num_attention_heads)
        block_bytes = (2 * hf_config.num_hidden_layers * self.block_size *
                       num_kv_heads * head_dim * hf_config.dtype.itemsize)
        config.num_kvcache_blocks = int(
            total * config.gpu_memory_utilization - used - peak + current
        ) // block_bytes
        assert config.num_kvcache_blocks > 0

        self.kv_cache = torch.empty(
            2, hf_config.num_hidden_layers, config.num_kvcache_blocks,
            self.block_size, num_kv_heads, head_dim,
        )
        layer_id = 0
        for module in self.model.modules():
            if isinstance(module, Attention):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]) -> torch.Tensor:
        """
        将“logical token → physical KV block”的映射表记录到context中，
        attention层查找kv cache table找到所有kv
        FlashAttention kernel 要求：
            - batch 内所有 sequence 的 block_table shape 必须一致
            - 所以必须用-1 padding
        使用自定义SPDA可以不用padding
        """
        max_len = max(len(seq.block_table) for seq in seqs)
        bt = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        return torch.tensor(bt, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)

    def prepare_inputs(self, seqs: list[Sequence]):
        """
        统一连续批输入构造（合并旧 prepare_prefill/prepare_decode）。

        每个 seq 取其 query 段 [num_cached_tokens, num_cached_tokens+num_scheduled_tokens)：
          - prefill chunk：num_scheduled_tokens 个 prompt token
          - decode：num_scheduled_tokens==1，即最后一个 token
        prefill chunk 与 decode token 混排在同一批，无 is_prefill 分支。

        block_table：任一 seq 已分配 KV 块时构造（覆盖 prefill/decode/前缀缓存，
        attention 统一从分页 cache 读）；仅 warmup（无 KV 块）时为 None，走裸 k/v。
        """
        input_ids, positions = [], []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = max_seqlen_k = 0
        slot_mapping = []
        has_cache = any(seq.block_table for seq in seqs)

        for seq in seqs:
            start = seq.num_cached_tokens
            seqlen_q = seq.num_scheduled_tokens
            end = start + seqlen_q
            seqlen_k = end  # KV 总长 = 已缓存 + 本步

            if seq.token_ids:
                input_ids.extend(seq[start:end])
            else:
                # rank>0 decode：仅 last_token 可用（seqlen_q==1）
                input_ids.append(seq.last_token)
            positions.extend(range(start, end))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)

            if not seq.block_table:
                slot_mapping.extend([-1] * seqlen_q)
            else:
                for pos in range(start, end):
                    block_id = seq.block_table[pos // self.block_size]
                    slot_mapping.append(block_id * self.block_size + pos % self.block_size)

        block_tables = self.prepare_block_tables(seqs) if has_cache else None

        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        sm = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        attn_md = AttentionMetadata(
            query_start_loc=cu_q, cu_seqlens_k=cu_k,
            max_query_len=max_seqlen_q, max_seq_len=max_seqlen_k,
            slot_mapping=sm, block_table=block_tables,
        )
        return input_ids, positions, attn_md

    def prepare_sample(self, seqs: list[Sequence]) -> torch.Tensor:
        temps = [seq.temperature for seq in seqs]
        return torch.tensor(temps, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)

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

    def run(self, seqs: list[Sequence]) -> list[int] | None:
        """单步推理接口（供 call() 调用）。统一连续批，无 is_prefill。"""
        input_ids, positions, attn_md = self.prepare_inputs(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, attn_md)
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        return token_ids

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
