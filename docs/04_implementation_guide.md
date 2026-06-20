# 实现指南

## 一、权重加载机制

### 1.1 问题：HF 权重格式 vs nano-vllm 参数格式

HuggingFace 将 Q/K/V 分别存储为独立权重，而 nano-vllm 将它们合并以减少 kernel launch 次数：

```
HuggingFace 格式：
  layers.0.self_attn.q_proj.weight   [2048, 2048]
  layers.0.self_attn.k_proj.weight   [1024, 2048]   ← GQA，比 Q 小
  layers.0.self_attn.v_proj.weight   [1024, 2048]
  layers.0.mlp.gate_proj.weight      [11008, 2048]
  layers.0.mlp.up_proj.weight        [11008, 2048]

nano-vllm 格式（合并后）：
  layers.0.self_attn.qkv_proj.weight [4096, 2048]   ← Q+K+V 拼接
  layers.0.mlp.gate_up_proj.weight   [22016, 2048]  ← gate+up 拼接
```

**合并收益**：3 次矩阵乘（q/k/v_proj）→ 1 次更大的矩阵乘（qkv_proj），减少 kernel launch + 更好利用 GPU SM 并行性。

### 1.2 packed_modules_mapping 设计

```python
# models/qwen3.py
packed_modules_mapping = {
    # HF 权重名中的子串 → (nano-vllm 参数名, shard_id)
    "q_proj":    ("qkv_proj", "q"),    # str shard_id 表示 QKV 中的哪部分
    "k_proj":    ("qkv_proj", "k"),
    "v_proj":    ("qkv_proj", "v"),
    "gate_proj": ("gate_up_proj", 0),  # int shard_id 表示合并参数中的位置索引
    "up_proj":   ("gate_up_proj", 1),
}
```

### 1.3 weight_loader 机制

每个参数（`nn.Parameter`）在初始化时就注册一个 `weight_loader` 函数属性：

```python
# linear.py（示意）
param.weight_loader = self.weight_loader   # 绑定到参数对象
```

加载时，调用 `param.weight_loader(param, loaded_weight, shard_id)` ——**参数自己负责处理如何接收 HF 权重**，而不是由加载器决定。这是关注点分离的好实践。

### 1.4 完整加载流程（`utils/loader.py`）

```python
def load_model(model, model_path):
    packed_mapping = model.packed_modules_mapping   # 从模型类获取
    state_dict = {n: p for n, p in model.named_parameters()}

    for file in sorted(glob(f"{model_path}/*.safetensors")):
        with safetensors.open(file) as f:
            for weight_name in f.keys():
                loaded = f.get_tensor(weight_name)  # 直接 mmap，不拷贝

                # 检查是否需要 packed 重映射
                for hf_key, (nano_key, shard_id) in packed_mapping.items():
                    if hf_key in weight_name:
                        # 将 weight_name 中的 hf_key 替换为 nano_key
                        param_name = weight_name.replace(hf_key, nano_key)
                        param = state_dict[param_name]
                        param.weight_loader(param, loaded, shard_id)
                        break
                else:
                    # 普通权重（归一化层、layernorm 等）
                    param = state_dict.get(weight_name)
                    if param is not None:
                        param.weight_loader(param, loaded)
```

### 1.5 TP 切分示例（QKVParallelLinear，tp=2，Qwen3-1.7B）

```
HF 存储：
  q_proj.weight: [2048, 2048]   (num_heads * head_dim, hidden)
  k_proj.weight: [1024, 2048]   (num_kv_heads * head_dim, hidden)
  v_proj.weight: [1024, 2048]

nano-vllm param（per GPU，tp=2）：
  qkv_proj.weight: [(1024+512+512), 2048] = [2048, 2048]
  内部布局：
    [0:1024]      → Q 部分（rank 0: heads 0~7，rank 1: heads 8~15）
    [1024:1536]   → K 部分（rank 0: kv_heads 0~3，rank 1: kv_heads 4~7）
    [1536:2048]   → V 部分（同 K）

weight_loader("q")：
  shard_offset = 0
  shard_size = num_heads / tp * head_dim = 8 * 128 = 1024
  # 从 HF q_proj.weight 取本 rank 的行：
  loaded_weight = q_proj.weight[rank*1024 : (rank+1)*1024, :]  → [1024, 2048]
  param.data[0:1024] = loaded_weight

weight_loader("k")：
  shard_offset = 1024
  shard_size = num_kv_heads / tp * head_dim = 4 * 128 = 512
  loaded_weight = k_proj.weight[rank*512 : (rank+1)*512, :]   → [512, 2048]
  param.data[1024:1536] = loaded_weight
```

---

## 二、GPU 显存估算公式详解

### 2.1 为什么不直接用 `free_memory`？

PyTorch 的 CUDA caching allocator（CachingAllocator）会保留已释放的内存供复用，`mem_get_info()` 返回的"空闲"是**分配器外的**空闲：

```
物理显存：8 GB
CachingAllocator 持有：4 GB（含 1 GB 已释放但缓存的块）
mem_get_info().free = 4 GB（分配器持有的不算空闲）
实际可新分配：4 GB（分配器会先复用缓存块）
```

同时，forward 过程中的激活张量是动态的，峰值无法从"当前分配量"推算。

### 2.2 公式分解

```python
# model_runner.py: allocate_kv_cache()
free, total = torch.cuda.mem_get_info()
used    = total - free              # 驱动视角的已用量
peak    = memory_stats["allocated_bytes.all.peak"]    # warmup 峰值
current = memory_stats["allocated_bytes.all.current"] # 当前分配量

# 可用 = 上限 - 固定占用 - 激活峰值预留
available = total * gpu_memory_utilization - used - (peak - current)
#            └── 用户设定的上限（默认 0.9）
#                                   └── 模型参数 + buffers + CUDA 系统占用
#                                                    └── forward 峰值激活需求
```

**关键洞察**：`peak - current` 是"warmup 结束后，激活张量的痕迹"——warmup 后激活已释放，peak > current，差值正好是需要为 forward 预留的空间。

### 2.3 每块字节数计算

```python
block_bytes = (
    2                             # K + V
    * hf_config.num_hidden_layers # 28 层（Qwen3-1.7B）
    * self.block_size             # 256 token/块
    * num_kv_heads                # 8（TP=1），4（TP=2）
    * head_dim                    # 128
    * hf_config.dtype.itemsize    # 2（bfloat16）
)
# = 2 × 28 × 256 × 8 × 128 × 2 = 1,835,008 B ≈ 1.75 MB/块
```

### 2.4 实测数字（Qwen3-1.7B，RTX 3060 Ti 8GB）

```
total  = 8,589,934,592 B（8 GB）
gpu_memory_utilization = 0.9
上限   = 8589934592 × 0.9 = 7,730,941,133 B

模型加载后：
  used    = 4,370,956,288 B（4.07 GB = 参数 4.06 GB + cos_sin_cache 21 MB + CUDA 系统）

warmup 后：
  peak    = 4,876,316,672 B（4.54 GB，含激活峰值 ~500 MB）
  current = 4,370,956,288 B（warmup 后激活释放，回到基线）

available = 7,730,941,133 - 4,370,956,288 - (4,876,316,672 - 4,370,956,288)
           = 7,730,941,133 - 4,370,956,288 - 505,360,384
           = 2,854,624,461 B（约 2.66 GB）

num_blocks = 2,854,624,461 // 1,835,008 ≈ 1556 个物理块
           = 1556 × 256 = 398,336 个 token 的 KV cache 容量
```

---

## 三、pin_memory + non_blocking 异步传输

### 3.1 为什么重要？

```
CPU（RAM）→ GPU（HBM）传输路径：
  普通内存（pageable）：先复制到 pinned buffer → 再 DMA 到 GPU
  固定内存（pinned）：  直接 DMA 到 GPU（省去中间复制）

non_blocking=True：
  DMA 传输与 GPU 计算异步进行
  但需要确保 CPU 不在传输期间修改数据
```

### 3.2 nano-vllm 中的使用

```python
# engine/input_batch.py: make_inputs / block_table.py: CpuGpuBuffer.copy_to_gpu
# 各数组写入常驻 pinned 缓冲（self.input_ids.np[:n]=...）后单次异步上传：
self.input_ids.gpu[:n].copy_(self.input_ids.cpu[:n], non_blocking=True)
#       ↑ 常驻 pinned CPU 缓冲（pin_memory=True）        ↑ 异步 H2D
```

**效果**：CPU 构建张量 + H2D 传输与 GPU 执行前一步的 kernel 可以重叠，隐藏传输延迟（PCIe 4.0 ×16 峰值 ~32 GB/s，传输 4 MB 需 ~0.1 ms）。

### 3.3 注意事项

```python
# 使用 non_blocking=True 时，代码继续运行，传输尚未完成
# 后续 CUDA kernel 会自动等待传输（隐式同步）
# 但如果需要在 CPU 侧读回结果，必须显式同步：
torch.cuda.synchronize()
```

---

## 四、性能优化全景

| 优化技术 | 实现位置 | 核心机制 | 收益 |
|---------|---------|---------|------|
| **PagedAttention** | `engine/kv_cache/` | KV cache 分页，消除碎片 | 内存利用率 ↑，并发 ↑ |
| **前缀缓存** | `engine/kv_cache/block_pool.py` | 链式哈希，跨请求复用 KV 块 | prefill 计算量 ↓，相同前缀命中 |
| **统一连续批 + Chunked Prefill** | `engine/sched/` | decode+prefill chunk 混排，避免 decode 饥饿 | TTFT ↓，TPOT 更平稳 |
| **持久化 InputBatch** | `engine/input_batch.py` + `block_table.py` | 跨步常驻行 + 增量块表 + 单次切片 H2D | 每步 CPU 构建/H2D ↓ |
| **FlashAttention（多后端）** | `layers/attention/` | varlen 统一 prefill/decode；registry+能力选择 | HBM 读写 ↓ O(n²→n)，速度 ↑ |
| **KV 写入 Triton kernel** | `layers/attention/kv_ops.py` | 向量化分散写，slot_mapping，无中间张量 | 显存分配 ↓，写入延迟 ↓ |
| **CUDA Graph** | `engine/model_runner.py` | 静态 graph replay，Python overhead 归零 | decode 小 batch 延迟 ↓ 50%+ |
| **torch.compile** | 各 layer | JIT 算子融合，消除中间张量 | elementwise / 归一化速度 ↑ |
| **Fused Add-RMSNorm** | `layers/layernorm.py` | 合并残差相加 + 归一化 | HBM bandwidth ↓ 50% |
| **Tensor Parallelism** | `layers/linear.py`, `embed_head.py` | 权重切分 + NCCL all_reduce | 多 GPU 显存 + 算力线性扩展 |
| **pin_memory + non_blocking** | `engine/block_table.py::CpuGpuBuffer` | H2D 异步传输与 GPU 计算重叠 | PCIe 传输延迟 ↓ |
| **lru_cache RoPE** | `layers/rotary_embedding.py` | 所有层共享一个 cos_sin_cache | 显存 ↓（N 层 → 1 份缓存） |
| **结构化采样层** | `layers/sample/` | greedy/top-k/top-p/penalties/logprobs；Gumbel 随机采样向量化 | 功能完整 + 大词表并行 |

---

## 五、从零实现路线图

> 这是一条**最小可用**的从零搭建路径（教学用）。当前 `phase5` 仓库在此基础上进一步对齐 vLLM V1
> 把各模块重构为子包/组件（引擎拆分、sched/、kv_cache/、executor/、InputBatch、sample/、attention
> 注册表），详见各 `arch_*/` 与 [nano_vs_vllm-架构对比](nano_vs_vllm-架构对比-20260618.md)。下文步骤已对齐
> 当前的关键事实（统一连续批、显式 AttentionMetadata、varlen 统一注意力）。

按依赖关系分阶段实现，每阶段可独立验证：

### 阶段一：基础数据结构（纯 Python，无 GPU）

```
1. config.py
   — Config dataclass，用 __post_init__ 验证约束
   — 关键字段：model_path, tp_size, max_num_seqs, max_num_batched_tokens

2. sampling_params.py
   — SamplingParams dataclass（temperature, max_tokens, ignore_eos）

3. engine/sequence.py
   — Sequence 状态机（WAITING/RUNNING/FINISHED）
   — block 计算属性（num_blocks = ceil(num_tokens / block_size)）
   — block(i) 方法返回第 i 块的 token_ids
   — pickle 优化（__getstate__/__setstate__）

4. engine/block_manager.py（不加前缀缓存）
   — Block dataclass
   — free_block_ids（deque），used_block_ids（set）
   — can_allocate（只检查空闲块数量），allocate，deallocate
   — can_append，may_append

5. engine/scheduler.py（基础 FCFS）
   — 先实现 prefill 路径，不加分块，不加抢占
   — update_from_output 只做 append_token + 终止检查
   （当前仓库升级为 sched/ 子包：SchedulerInterface + 结构化 SchedulerOutput + 可插拔队列）

验证：手动创建 Sequence，模拟分配/释放/调度流程，打印 block_table。
```

### 阶段二：神经网络层（单 GPU，小模型测试）

```
6.  utils/context.py
    — AttentionMetadata dataclass（query_start_loc / cu_seqlens_k / slot_mapping / block_table）
    — 经 model.forward(input_ids, positions, attn_md) 显式透传到 Attention 层（非全局单例）

7.  layers/layernorm.py
    — RMSNorm（先不加 fused add-norm，直接 x = x / rms * weight）

8.  layers/activation.py
    — SiluAndMul（split + silu * up）

9.  layers/rotary_embedding.py
    — precompute cos_sin_cache（max_len, head_dim, theta）
    — 手写旋转应用（rotate_half + cos * q + sin * rotate(q)）
    — @lru_cache 单例

10. layers/attention.py（naive 版本）
    — 先用 torch.scaled_dot_product_attention（不用 FlashAttention）
    — 暂时不接 KV cache，每次重新计算

11. layers/linear.py（非并行版本）
    — 只实现 ReplicatedLinear（F.linear + weight_loader 接口）

12. layers/embed_head.py（非并行版本）
    — 标准 nn.Embedding + F.linear

13. layers/sampler.py
    — 先用 torch.multinomial（后换 Gumbel-max）

14. models/qwen3.py（单 GPU 版本）
    — 组装 Transformer：Embedding + N×DecoderLayer + Norm + LMHead
    — compute_logits：统一用 query_start_loc[1:]-1 取每序列末 token（prefill/decode 同一逻辑）

验证：随机权重 forward，检查 output shape 正确（[batch, vocab_size]）。
```

### 阶段三：权重加载与完整推理

```
15. utils/loader.py
    — safetensors 读取
    — packed_modules_mapping 重映射逻辑
    — 按 weight_name 分发到 param.weight_loader

16. engine/model_runner.py（单进程版本）
    — warmup_model（构造虚假 prefill，测峰值）
    — allocate_kv_cache（按剩余显存计算 num_blocks，分配大张量）
    — prepare_inputs：统一连续批构造 input_ids/positions/cu_seqlens/slot_mapping（slot_mapping 是重点）
      （当前仓库进一步抽成持久化 InputBatch：跨步常驻行 + 增量块表）
    — run（构造 AttentionMetadata → run_model → sampler）

17. engine/llm_engine.py（单进程，无 TP）
    — generate（tokenize + 创建 Sequence + 主循环）
    — step（schedule → execute_model → update_from_output）
      （当前仓库拆为 Processor + EngineCore + OutputProcessor，并加 AsyncLLM 异步流式）

验证：加载真实权重，单条 prompt 能生成连贯文本。
      enforce_eager=True，逐 token 打印。
```

### 阶段四：工程优化（按独立性并行推进）

```
18. FlashAttention（layers/attention/）
    — 统一 flash_attn_varlen_func 覆盖 prefill 与 decode（decode = query_len 1 的退化）：
      block_table 非 None 即从分页 cache 读历史 KV，无需 with_kvcache 分支
    — 多后端：backend 三件套 + registry（AttentionBackendEnum）+ 能力选择（flash/sdpa）

19. Triton KV 写入（layers/attention/kv_ops.py）
    — store_kvcache_kernel（关键：slot=-1 的处理）；替换 Python scatter

20. Tensor Parallelism（linear.py, embed_head.py, executor/）
    — 先实现 ColumnParallelLinear / RowParallelLinear，再 QKV / Merged
    — Executor 抽象编排进程：UniProc 内联 / MultiProc（spawn + ShmTransport 广播 + ResultChannel 回传）

21. CUDA Graph（model_runner.py）
    — capture_cudagraph：warmup → 从大到小录制 → 共享 pool（用静态 AttentionMetadata）
    — run_model：纯 decode 批按 bs 选 graph，写静态张量，replay

22. Prefix Caching（kv_cache/block_pool.py + sched/scheduler.py）
    — compute_hash（链式 xxhash）；can_allocate 中的哈希探测
    — hash_blocks（update_from_output 后注册满块哈希）；脏哈希清理

23. 统一连续批 + Chunked Prefill（sched/scheduler.py）
    — 一步内先调度 RUNNING decode（每 seq 1 token），再用剩余预算给 WAITING prefill chunk，混排同批
    — 任意队首 seq 可分块（不限第一个）；update_from_output 中 is_prefill 派生为真时不 append_token

24. Fused Add-RMSNorm（layernorm.py）
    — add_rms_forward：就地加法 + 同步更新 residual
    — Qwen3DecoderLayer 中传递 residual 的模式

25. torch.compile（各 layer）
    — 在 @torch.inference_mode() 外加 @torch.compile（或函数内部）
    — 先只 compile RoPE 和 Sampler（最易）
```

---

## 六、可扩展点（重构 / 新增功能）

### 1. 支持更多模型

仿照 `models/qwen3.py`，实现 Llama / Mistral / Gemma / Phi 等：

```python
# 需要适配的主要差异：
packed_modules_mapping = { ... }   # HF 权重名 → 合并权重名的映射

class Config:
    rope_theta: float              # Llama: 10000, Qwen: 1000000
    rope_scaling: dict             # LongRoPE, YaRN 等扩展方式

# Llama 3.1 vs Qwen3 的结构差异：
# - Llama: GQA + SiLU（无 QK-Norm）
# - Qwen3: GQA + SwiGLU + QK-Norm
# - Gemma: 词表归一化（embed_tokens 有 scale）
```

### 2. Beam Search

在 Sequence 层面支持多候选（beam）：

```python
class BeamSequence(Sequence):
    beam_id: int        # beam 编号
    parent_seq_id: int  # 父 beam 的 seq_id（共享前缀 KV 块）
    score: float        # 累积对数概率

# Sampler 返回 top-N token，而非 argmax：
top_tokens = logits.topk(beam_width, dim=-1)

# BlockManager：prompt 的 KV 块由所有 beam 共享（前缀缓存自然复用）
# 每个 beam 独立维护 decode 阶段的私有块
```

### 3. 投机采样（Speculative Decoding）

```
Draft model（小模型）预测 k 个 token：
  draft_tokens = [t1, t2, t3, t4]   # 连续 decode

Target model（大模型）并行验证（一次 prefill 相当于 k+1 个 token）：
  验证方式：比较 target 在 [t1,t2,t3,t4] 上的条件分布与 draft 分布
    接受概率 = min(1, p_target / p_draft)

平均接受长度 β = Σ 接受概率（期望值，一般 β ≈ 2~4）
等效吞吐 ≈ β × target_decode_throughput
```

实现要点：
- `LLMEngine` 持有两个 `ModelRunner`（draft + target）
- draft model 也需要 KV cache，但更小
- 拒绝时：修正采样（取 target 分布中 draft 未覆盖的部分）

### 4. KV Cache Swap（换出优化）

当前抢占直接丢弃 KV cache（recompute），可改为换出到 CPU：

```python
# preempt 改为 swap_out：
def swap_out(seq):
    cpu_buffer = torch.empty_like(kv_cache[:, :, seq.block_table])
    cpu_buffer.copy_(kv_cache[:, :, seq.block_table])   # GPU → CPU
    cpu_kv_cache[seq.seq_id] = cpu_buffer
    block_manager.deallocate(seq)
    waiting.appendleft(seq)
    seq.swapped = True

# 重调度时 swap_in：
def swap_in(seq):
    block_manager.allocate(seq, 0)   # 重新分配 GPU 块
    kv_cache[:, :, seq.block_table].copy_(cpu_kv_cache.pop(seq.seq_id))  # CPU → GPU
    seq.swapped = False
```

**权衡**：swap 避免重计算（节省时间），代价是 PCIe 带宽（~16 GB/s）。对于长 prompt 的请求，swap 通常优于 recompute。

### 5. 量化支持

在 `LinearBase` 的 weight_loader 中加入量化：

```python
# W8A16：权重 int8 存储，forward 前反量化
class W8A16Linear(LinearBase):
    def __init__(self, ...):
        self.weight = nn.Parameter(torch.empty(..., dtype=torch.int8))
        self.scale = nn.Parameter(torch.empty(out_features))  # per-channel scale

    def forward(self, x):
        weight_fp16 = self.weight.float() * self.scale.unsqueeze(1)
        return F.linear(x, weight_fp16.to(x.dtype))

# W4A16（AWQ）：需要 group_size + zero_point
# FP8（H100）：torch.float8_e4m3fn，硬件原生支持
```

### 6. 在线服务接口（Streaming）—— 已实现 AsyncLLM（E 轮）

`engine/async_llm.py::AsyncLLM` 已提供异步 generator 流式接口（每 step 增量 yield `RequestOutput`），
见 `example_async.py`。剩余差距：未做 OpenAI 兼容 HTTP server（仅引擎侧异步通路）。

```python
from nanovllm import AsyncLLM, SamplingParams
async for out in async_llm.generate(prompt, SamplingParams(...), request_id):
    print(out.delta_text, end="")   # 增量文本；out.finished 标志结束
# 配合 FastAPI StreamingResponse 即可对外提供 SSE 流式（HTTP 层需自行补）
```

### 7. 多模态扩展

扩展 `VocabParallelEmbedding` 支持图像 patch embedding：

```python
class MultimodalEmbedding(nn.Module):
    def forward(self, input_ids, pixel_values=None):
        text_emb = self.text_embedding(input_ids)
        if pixel_values is not None:
            image_features = self.vision_encoder(pixel_values)  # ViT
            # 将 image token id 对应的 embedding 替换为 vision features
            image_mask = input_ids == IMAGE_TOKEN_ID
            text_emb[image_mask] = image_features.flatten(0, 1)
        return text_emb
```

2D RoPE：图像 patch 的位置编码使用 (row, col) 二维坐标而非一维序列位置。

### 8. 连续批处理精细化 —— 已实现（A 轮）

统一连续批已在 `sched/scheduler.py` 落地：一步内先填 RUNNING decode（每 seq 1 token），再用剩余
token 预算给 WAITING prefill chunk，二者混排同一 varlen 批（详见 03 §六）。decode TPOT 不再被
prefill 阻塞。剩余可精细化方向：`async_scheduler`（提前调度下一步）、swap 抢占（当前仅 recompute）。

---

## 七、调试技巧

### 关闭 CUDA Graph

```python
llm = LLM(model_path, enforce_eager=True)
# enforce_eager=True 时跳过 capture_cudagraph，走完整 PyTorch eager 路径
# 便于 print 调试、检查中间张量、使用 torch.autograd.profiler
```

### 逐步追踪调度

```python
# 在 EngineCore.step() 中打印调度信息
sched_output = scheduler.schedule()
seqs = sched_output.scheduled_seqs
print(f"{len(seqs)} seqs, total_tokens={sched_output.total_num_scheduled_tokens}, "
      f"finished={sched_output.finished_seq_ids}")
for s in seqs:
    print(f"  seq {s.seq_id}: blocks={s.block_table}, cached={s.num_cached_tokens}, "
          f"scheduled={s.num_scheduled_tokens}, is_prefill={s.is_prefill}")
```

### slot_mapping 合法性检查

```python
num_blocks = config.num_kvcache_blocks
assert (slot_mapping >= -1).all()
assert (slot_mapping < num_blocks * block_size).all()
# -1 是合法的（CUDA graph dummy）
# 其余必须在 [0, num_blocks * block_size) 范围内
```

### 前缀缓存命中率统计

```python
# 在 can_allocate 后记录
def can_allocate_with_stats(self, seq):
    n = self.can_allocate(seq)
    if n >= 0 and seq.num_blocks > 1:
        hit_ratio = n / (seq.num_blocks - 1)   # 不含最后未满块
        self.total_hit += n
        self.total_possible += seq.num_blocks - 1
    return n

# 每 N 步打印
if step % 100 == 0:
    print(f"Prefix cache hit rate: {bm.total_hit / max(1, bm.total_possible):.1%}")
```

### KV Cache 内存对齐验证

```python
# 确认 kv_cache 的内存布局与 Triton kernel 假设一致
k_cache = model_runner.kv_cache[0, 0]  # [num_blocks, block_size, num_kv_heads, head_dim]
assert k_cache.is_contiguous()
assert k_cache.stride(1) == num_kv_heads * head_dim   # block_size 维度的 stride
# Triton kernel 假设 k_cache.stride(1) == D = num_heads * head_dim
```

### GPU 内核分析

```python
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
) as prof:
    llm.generate(["test prompt"], SamplingParams())

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
# 关注：flash_attn, triton_store_kvcache, mm（matmul），all_reduce 的耗时
```

---

## 八、关键数字速查（Qwen3-1.7B，RTX 3060 Ti 8GB）

| 项目 | 数值 | 备注 |
|------|------|------|
| 模型参数量 | 1.7B | 约 4.06 GB（bfloat16） |
| num_kvcache_blocks | ~1556 | gpu_memory_utilization=0.9 时 |
| 最大并发 token | ~398K | 1556 × 256 |
| 每块 KV 大小 | 1.75 MB | 28层 × 256tok × 8heads × 128dim × 2 × bf16 |
| CUDA Graph 数量 | 16 | bs: [1,2,4,8,16,32,...,512] |
| Decode 延迟（bs=1，eager） | ~30 ms | 包含 kernel launch overhead |
| Decode 延迟（bs=1，graph） | ~15 ms | CUDA Graph replay |
| Prefill 吞吐 | ~6 tok/s | Qwen3-1.7B，enforce_eager=True |
| Decode 吞吐 | ~18 tok/s | Qwen3-1.7B，enforce_eager=True |
