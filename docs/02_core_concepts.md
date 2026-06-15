# 核心技术详解

## 1. PagedAttention（分页 KV 缓存）

### 1.1 问题背景

朴素实现为每个请求预分配 `max_seq_len × num_layers × kv_heads × head_dim` 的**连续显存**。以 Qwen3-1.7B（max_len=4096）为例，每个请求占用：

```
4096 × 28 × 8 × 128 × 2（bf16）= 234 MB
```

问题：
1. **内部碎片**：实际生成 100 token，却占了 4096 token 的空间（利用率 < 5%）
2. **外部碎片**：连续分配，无法利用显存中的"小缝隙"
3. **批并发受限**：8 GB / 234 MB ≈ 34 个并发请求（远低于理论值）

### 1.2 分页解决方案

借鉴操作系统的虚拟内存分页：

```
物理 KV cache（统一大张量）：
  shape: [2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]
                                    ↑
                           物理块池：N 个块，每块存 block_size=256 个 token

每个 Sequence 维护 block_table（逻辑块 → 物理块 ID 映射）:
  Seq A: block_table = [2, 5, 1]
    token[0:256]    → 物理 Block 2（k_cache[2, 0:256, :, :]）
    token[256:512]  → 物理 Block 5
    token[512:...]  → 物理 Block 1（当前写入中，未填满）

  Seq B: block_table = [2, 7]    ← Block 2 与 Seq A 共享（前缀缓存）
    token[0:256]    → 物理 Block 2（ref_count = 2）
    token[256:...]  → 物理 Block 7（私有）
```

**Block 引用计数**（防止多 seq 共享时的提前释放）：

```
分配：block.ref_count = 1
共享：block.ref_count += 1（前缀缓存命中）
释放：block.ref_count -= 1，为 0 才回到空闲队列
```

### 1.3 为什么 block_size = 256？

这是工程权衡：

| block_size | 碎片 | 哈希粒度 | FlashAttention tile 对齐 |
|------------|------|---------|------------------------|
| 16 | 小 | 细（命中率高） | 需多次 tile 运算 |
| 64 | 中 | 较细 | 较好 |
| **256** | **可接受** | **较粗（一块才缓存）** | **完美对齐（FA 的 tile_k=64/128）** |
| 1024 | 大 | 粗 | — |

256 × num_kv_heads × head_dim 正好是 FlashAttention `block_k` tile 的整数倍，避免 block 边界计算开销。

---

## 2. 前缀缓存（Prefix Caching）

### 2.1 核心思想

相同前缀 token 序列产生相同 KV 计算结果（**决定性，只要模型权重不变**），可跨请求复用：

```
请求 A: [system_prompt(256tok)] + [用户问题 A(100tok)]
请求 B: [system_prompt(256tok)] + [用户问题 B(80tok)]
              ↑
    hash 相同 → 共享同一物理 Block，Seq B 跳过前 256 token 的 prefill
```

### 2.2 链式哈希设计

**为什么需要链式（chained）哈希？**

假设两个请求有相同 token 序列但不同的历史（如同一词在不同上下文中出现），它们的 KV 值不同，不能共享。链式哈希将前一块的哈希拼入，确保唯一性：

```
Block 0: hash_0 = xxhash(token_ids[0:256])
Block 1: hash_1 = xxhash(hash_0 || token_ids[256:512])  ← 依赖 hash_0
Block 2: hash_2 = xxhash(hash_1 || token_ids[512:768])  ← 依赖 hash_1
```

如果两个请求的 token_ids[0:256] 相同但 token_ids[-1:256] 不同，它们的 hash_1 不同，不会错误共享 Block 1。

**实现：**

```python
@classmethod
def compute_hash(cls, token_ids, prefix=-1):
    h = xxhash.xxh64()
    if prefix != -1:
        h.update(prefix.to_bytes(8, "little"))   # 前一块哈希（链的头）
    h.update(np.array(token_ids).tobytes())       # 本块 token 内容
    return h.intdigest()                          # 64 位无符号整数
```

为什么用 xxhash 而非 hashlib.md5？xxhash64 在现代 CPU 上的吞吐 > 10 GB/s，是 MD5 的 10× 以上，且对于前缀缓存只需防碰撞、不需加密安全。

### 2.3 生命周期时序

```
时间轴：

T0: seq 加入 waiting
T1: schedule() → can_allocate(seq)
      遍历 seq 除最后块以外的满块，计算链式哈希
      查 hash_to_block_id + 校验 token_ids（防哈希碰撞）
      返回 num_cached_blocks（命中数，-1 表示内存不足）

T2: block_manager.allocate(seq, num_cached_blocks)
      命中块：直接引用（ref_count++，若在 used_block_ids）
              或重激活（若在 free_block_ids）
      其余块：从空闲队列分配新块
      seq.num_cached_tokens = num_cached_blocks * block_size
      ← prepare_prefill 中 input_ids 从 num_cached_tokens 开始

T3: prefill forward
      Attention 中：cu_seqlens_k > cu_seqlens_q 说明有历史 KV
      flash_attn_varlen_func 用 block_table 读取历史 KV 块

T4: postprocess() → block_manager.hash_blocks(seq)
      对本步新填满的块（从上次缓存边界到本步末尾）计算哈希
      注册到 hash_to_block_id 供后续 seq 命中

T5: deallocate(seq) / preempt(seq)
      ref_count--，归零则 _deallocate_block（FIFO 追加到队尾）
      hash 仍保留在 hash_to_block_id！
      ← 块进入空闲队列但 hash 未清，后续 seq 仍能命中（延迟缓存）

T6: _allocate_block()（块被真正复用时）
      清理该块的 hash 记录（从 hash_to_block_id 删除）
      防止"脏命中"：块内容将被新 seq 覆写，旧 hash 不再有效
```

### 2.4 FIFO 延迟复用的价值

空闲块不是 LRU，而是 FIFO（从队头取，从队尾还）：

```
情况：Seq A 完成，释放 Block 2 → 队尾
      短时间内 Seq B 到来，有相同前缀 → can_allocate 命中 Block 2（仍在队尾）
      Block 2 被重激活（无需重新 prefill）

若是 LRU：Block 2 可能因为其他请求而提前被踢出，命中率下降
FIFO：新释放的块延迟被复用，短期内保留前缀缓存效果
```

---

## 3. Chunked Prefill（分块预填充）

### 3.1 为什么需要分块？

朴素的"prefill 优先"策略下，极端情况：

```
Step 1: 超长 prompt（32K tokens）prefill → GPU 跑 ~2 秒
Step 2~N: 其他所有 decode 请求被饿死，TTFT（Time To First Token）飙升
```

Chunked Prefill 将长 prompt 按 `max_num_batched_tokens`（如 4096）切分，每步只处理一个 chunk，同时允许 decode 请求混入。

### 3.2 调度约束（重要细节）

```python
# scheduler.py
while waiting:
    seq = waiting[0]
    remaining = max_num_batched_tokens - already_scheduled

    # 关键：只有本轮第一个 seq 允许分块
    if remaining < num_tokens and scheduled_seqs:
        break   # 第 2+ 个 seq 必须一次完整 prefill 或等下一步

    seq.num_scheduled_tokens = min(num_tokens, remaining)
    scheduled_seqs.append(seq)
```

**为什么只允许第一个 seq 分块？**

设想同时有两个长 prompt 都在分块：
- Seq A 分块进行到 chunk 3，Seq B 分块进行到 chunk 2
- 每步的 input_ids 是两者 chunk 的拼接，`cu_seqlens_q` 正确描述各 seq 的 query 长度
- 理论上是正确的，但实现复杂（需追踪每个 seq 各自的分块进度）

nano-vllm 采用最简策略：**最多一个 seq 做分块（waiting 队列第一个），其他 seq 必须整批 prefill**。这样调度逻辑只需追踪一个分块 seq 的进度。

### 3.3 分块状态追踪

```
Seq A（2000 token prompt，max_num_batched_tokens=1024）：

Step 1: scheduled_tokens = 1024, is_prefill=True
         seq 仍在 waiting（未处理完）
         num_cached_tokens 为 0 → 处理后 = 1024

Step 2: scheduled_tokens = 976 (2000-1024), is_prefill=True
         num_cached_tokens = 1024 → 处理后 = 2000
         2000 == num_tokens → 移入 running，状态变 RUNNING

Step 3: scheduled_tokens = 1 (decode), is_prefill=False
         num_cached_tokens 持续累积
```

### 3.4 Chunked Prefill 与 Decode 混合批次

```
waiting = [SeqA(2000 tok)]，running = [SeqB(decode), SeqC(decode)]

Step 1 调度：
  is_prefill = True（因为有 waiting 序列）
  SeqA: chunk 1024 tokens（占满 token budget）
  → scheduled = [SeqA]，is_prefill=True

  问题：SeqB、SeqC 的 decode 被跳过了！

实际 nano-vllm 行为：
  prefill 时不处理 running 中的 decode 请求（is_prefill 独占一步）
  只有 waiting 队列清空后才进入 decode 步
```

> 注：真正的 prefill+decode 混批（将 decode token 加入 prefill 批次）在 nano-vllm 中未实现，这是与 vLLM 的一处差异。vLLM 的 chunked prefill 可以在同一步内混合 prefill chunk 和 decode token，进一步降低延迟。

---

## 4. 张量并行（Tensor Parallelism）

### 4.1 核心原理

矩阵乘法的可并行性：`Y = X @ W^T`，W 按不同维度切分给多个 GPU：

```
列并行（Column Parallel）：W 按行切分（输出维度切分）
  GPU 0: W_0 = W[:, 0:H/2]^T  →  Y_0 = X @ W_0  ，shape [N, H/2]
  GPU 1: W_1 = W[:, H/2:]^T  →  Y_1 = X @ W_1  ，shape [N, H/2]
  结果：cat([Y_0, Y_1], dim=-1) = Y，shape [N, H]  ← 无需通信（结果可直接拼接）

行并行（Row Parallel）：W 按列切分（输入维度切分）
  GPU 0 输入：X_0 = X[:, 0:H/2]（来自上层列并行的 Y_0）
  GPU 1 输入：X_1 = X[:, H/2:]（来自上层列并行的 Y_1）
  GPU 0: partial_0 = X_0 @ W_0^T，shape [N, H_out]（部分和）
  GPU 1: partial_1 = X_1 @ W_1^T，shape [N, H_out]（部分和）
  all_reduce(sum): partial_0 + partial_1 = Y，shape [N, H_out]  ← 需要 all_reduce
```

Transformer 中的配对：
- `QKV 投影`（列并行）+ `o_proj`（行并行 + all_reduce）
- `gate_up_proj`（合并列并行）+ `down_proj`（行并行 + all_reduce）

### 4.2 NCCL Ring-AllReduce

AllReduce 在 TP 中是最关键的通信原语：

```
Ring-AllReduce（TP=4 为例）：
  Round 1（Scatter-Reduce）：
    GPU 0 → GPU 1 → GPU 2 → GPU 3 → GPU 0（循环发送分片并累加）
    每个 GPU 发送/接收 1/4 的数据量，共 TP-1=3 轮

  Round 2（AllGather）：
    将各 GPU 已汇总的分片广播回所有 GPU
    同样 TP-1=3 轮

  总通信量：2 × (TP-1)/TP × data_size ≈ 2 × data_size（TP 大时）
  带宽利用率：随 TP 线性提升（NVLink：600 GB/s 双向）
```

每次 all_reduce（`dist.all_reduce(y_partial)` in PyTorch）会自动使用 NCCL 的 ring 实现。

### 4.3 QKV 合并并行（QKVParallelLinear）

GQA 下 Q/K/V 的 head 数不同，合并方式更复杂：

```
Qwen3-1.7B，tp=2：
  num_heads = 16（Q），num_kv_heads = 8（K/V）

每个 GPU 的 head 分配：
  Q heads: 16/2 = 8 per GPU
  K heads: 8/2 = 4 per GPU
  V heads: 8/2 = 4 per GPU

合并权重形状（per GPU）：
  [(8+4+4) × 128, 2048] = [2048, 2048]

内存布局（合并后拆分）：
  qkv[:8*128]  → Q（8 heads × 128 head_dim）
  qkv[8*128:8*128+4*128]  → K
  qkv[8*128+4*128:]  → V
```

weight_loader 负责在加载时将 HF 的独立 q/k/v_proj 正确拼装到对应区域（见 `04_implementation_guide.md`）。

### 4.4 词表并行（VocabParallelEmbedding）

词表均分后的 forward 技巧（避免 gather 带来的不必要通信）：

```python
def forward(self, input_ids):
    vocab_start, vocab_end = rank * V//tp, (rank+1) * V//tp
    # 把超出本 rank 词表范围的 token_id 置为 0（不影响 embedding，但后续要 mask）
    masked_ids = input_ids.clone()
    mask = (input_ids < vocab_start) | (input_ids >= vocab_end)
    masked_ids[mask] = 0
    # 查本 rank 的词表（local_id = global_id - vocab_start）
    output = F.embedding(masked_ids - vocab_start, self.weight)
    output[mask] = 0   # mask 掉不属于本 rank 的行（它们的 embedding 是 0）
    dist.all_reduce(output)  # 每个位置只有一个 rank 有非零值，all_reduce = select
    return output
```

all_reduce 的语义变成"每个位置取唯一非零值"，相当于分布式 gather，但避免了 gather 的 variable-length 通信。

---

## 5. CUDA Graph（解码加速）

### 5.1 问题：Python Kernel Launch Overhead

GPU 执行模型 forward 时，CPU 需要为每个算子（matmul、softmax、layernorm...）调用 `cudaLaunchKernel`，这个调用本身有 ~10 μs 的开销。

Qwen3-1.7B 的一次 decode forward 涉及约 **500+** 次 kernel launch（每层约 20 个）。decode 阶段 batch_size=1 时，kernel launch 总开销 ~5 ms，而实际 GPU 计算只需 ~2 ms。**launch overhead 占 70%+。**

### 5.2 CUDA Graph 原理

CUDA Graph 在"录制"阶段截获所有 kernel launch 命令（不真正执行），存为静态图。"Replay"时，一次 `cuGraphLaunch` 调用触发整个图的执行，Python-level 开销从 500 次变为 1 次。

**约束**：
- 张量地址固定（graph 中硬编码 GPU 指针）
- 张量形状固定（不能动态变化）

### 5.3 Warmup → Capture 流程

```python
# capture_cudagraph()
for bs in reversed([1, 2, 4, 8, 16, ..., 512]):  # 从大到小
    graph = torch.cuda.CUDAGraph()

    # ① Warmup（不录制）：预热 CUDA 缓存分配器，确保 capture 时无新分配
    set_context(False, slot_mapping[:bs], context_lens[:bs], block_tables[:bs])
    outputs[:bs] = self.model(input_ids[:bs], positions[:bs])

    # ② Capture（录制所有 kernel launch）
    with torch.cuda.graph(graph, self.graph_pool):
        outputs[:bs] = self.model(input_ids[:bs], positions[:bs])

    # ③ 第一次（最大 bs）时创建 memory pool，后续 graph 共用
    if self.graph_pool is None:
        self.graph_pool = graph.pool()

    self.graphs[bs] = graph
```

**为什么从大到小录制？**

PyTorch CUDA Graph 要求录制时不能有新的显存分配（CUDA caching allocator 会缓存，但如果 pool 是共享的，第一次录制要确保 pool 足够大）。从最大 bs 开始，pool 建立时就有足够空间，后续小 bs 的 graph 直接复用 pool 中的已有内存段。

### 5.4 Replay 详解

```python
def run_model(self, input_ids, positions, is_prefill):
    if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
        # 无法用 CUDA Graph（形状不固定 / 超过录制范围）
        return self.model.compute_logits(self.model(input_ids, positions))

    bs = input_ids.size(0)
    # 找最小的满足条件的 graph batch size
    graph_bs = next(x for x in self.graph_bs if x >= bs)   # 如 bs=3 → graph_bs=4
    graph = self.graphs[graph_bs]
    gv = self.graph_vars

    # 修改静态张量内容（graph 内核引用的是这些张量的 GPU 地址）
    gv["input_ids"][:bs] = input_ids                # [graph_bs] 中只有前 bs 个有效
    gv["positions"][:bs] = positions
    gv["slot_mapping"].fill_(-1)                     # 全填 -1（无效 slot）
    gv["slot_mapping"][:bs] = context.slot_mapping  # 前 bs 个填入真实 slot
    gv["context_lens"].zero_()
    gv["context_lens"][:bs] = context.context_lens
    gv["block_tables"][:bs, :] = ...

    graph.replay()   # 一次 GPU 调用，执行全部 500+ kernel

    return self.model.compute_logits(gv["outputs"][:bs])
```

**padding 的处理：**
- graph 录制时 bs=4，实际 bs=3，多出的 1 个"虚假"token 的 slot=-1（Triton kernel 跳过写入），context_lens=0（FlashAttention 输出 NaN/0，被后续 `[:bs]` 截掉）

### 5.5 Graph Pool（显存管理）

每个 CUDA Graph 有自己的 capture-time 内存分配。`graph.pool()` 返回该 graph 的 memory pool handle，后续 graph 通过 `torch.cuda.graph(g, pool)` 复用同一 pool。

效果：所有 batch size 的 graph 共享同一显存区域（按最大 bs 的需求分配），无碎片。

---

## 6. FlashAttention 详解

### 6.1 标准注意力的显存瓶颈

标准注意力计算：
```
S = Q @ K^T / scale          # [seq_len, seq_len]  ← 显存瓶颈
P = softmax(S)                # [seq_len, seq_len]
O = P @ V                     # [seq_len, head_dim]
```

对于 seq_len=4096，一个 head 的 S 矩阵：
```
4096 × 4096 × 2（bf16）= 32 MB
```
28 层 × 16 heads = 448 个这样的矩阵 = **14 GB**，远超 GPU 显存。

### 6.2 IO-Aware Tiling（FlashAttention 的核心创新）

FlashAttention 不存储完整的 S 矩阵，而是分 tile 计算，在 SRAM（L1 Cache）内完成 softmax + 加权：

```
SRAM 大小：A100 = 192 KB per SM，约可存 tile_q=128 行，tile_k=64 列（bf16）

算法：
  for q_tile in Q.split(128):              # Q 分块，载入 SRAM
      running_max = -inf                   # 在线 softmax 需要的统计量
      running_sum = 0
      running_output = 0
      for k_tile in K.split(64):           # K/V 分块，载入 SRAM
          S_tile = q_tile @ k_tile^T       # [128, 64]，完全在 SRAM 内
          # 在线 Softmax（Softmax 的数值稳定 online 版）
          new_max = max(running_max, S_tile.max())
          running_sum = running_sum * exp(running_max - new_max) + exp(S_tile - new_max).sum()
          running_output = running_output * exp(running_max - new_max) + exp(S_tile - new_max) @ v_tile
          running_max = new_max
      output[q_tile] = running_output / running_sum
```

**显存复杂度**：O(seq_len)（只需 running statistics），而不是 O(seq_len²)

**HBM 访问量**：HBM 读写次数 ≈ seq_len × head_dim × num_pass，而不是 seq_len²，大幅减少 HBM 带宽消耗（HBM ~2 TB/s vs SRAM ~20 TB/s）。

### 6.3 nano-vllm 中的两个接口

#### `flash_attn_varlen_func`（prefill）

```python
o = flash_attn_varlen_func(
    q,   # [total_tokens, num_heads, head_dim]  所有 seq 拼接
    k, v,
    cu_seqlens_q=[0, len_A, len_A+len_B, ...],   # 各 seq 在拼接后的起止位置
    cu_seqlens_k=[0, klen_A, klen_A+klen_B, ...], # KV 长度（含前缀缓存时 > q）
    max_seqlen_q=..., max_seqlen_k=...,
    causal=True,           # 下三角 mask（自回归）
    block_table=...,       # 前缀缓存时有值（分页 KV 地址映射），无则 None
)
```

`cu_seqlens_k > cu_seqlens_q`：前缀缓存已有历史 KV，本步 query 只有 `seqlen_q` 个但要 attend to `seqlen_k` 个 key。此时 FlashAttention 直接从 `k_cache/v_cache` 按 `block_table` 读取历史 KV。

#### `flash_attn_with_kvcache`（decode）

```python
o = flash_attn_with_kvcache(
    q.unsqueeze(1),   # [bs, 1, num_heads, head_dim]  每 seq 只有 1 个 query
    k_cache,          # [num_blocks, block_size, num_kv_heads, head_dim]
    v_cache,
    cache_seqlens=context_lens,   # [bs]  每 seq 有多少有效 KV（不含 padding）
    block_table=block_tables,     # [bs, max_blocks]  分页地址映射
    causal=True,
)
```

decode 时的 Q 只有 1 个 token，KV 来自整个历史（从 kv_cache 中按 block_table 读取）。FlashAttention 内部会自动处理分页读取逻辑。

### 6.4 KV 写入：为什么用 Triton 而非 PyTorch scatter？

```
标准做法（PyTorch）：
  k_cache.view(-1, D)[slot_mapping] = k.view(-1, D)  ← scatter
  问题：
    1. 需要 reshape → 可能分配新中间张量（cudaMalloc 很慢）
    2. PyTorch scatter 不保证内存连续性优化

Triton kernel（store_kvcache_kernel）：
  每个 CUDA block 处理 1 个 token，直接读取 k[idx] 写到 k_cache[slot*D]
  优点：
    1. 零中间张量（no_malloc）
    2. D 为 constexpr → Triton 编译时展开循环，向量化读写（SIMD）
    3. slot==-1 的 dummy token 直接跳过（CUDA graph 时必须）
```

---

## 7. 全局推理上下文（Context）

### 7.1 设计动机

Attention 层需要 `cu_seqlens`、`slot_mapping`、`block_tables` 等推理元数据，但：
- 这些是**运行时状态**，与模型结构无关
- 每步推理才会确定，不是模型的固定参数
- 如果作为 `forward` 参数传递：每层都要透传，签名臃肿，且 CUDA Graph 录制时参数必须固定

**解决**：全局单例 `_CONTEXT`，在 `prepare_prefill/decode` 时写入，在需要的层中读取，forward 结束后 reset。

### 7.2 Context 字段完整说明

| 字段 | prefill | decode | 含义 |
|------|---------|--------|------|
| `is_prefill` | True | False | 当前步类型，决定 FlashAttention 接口选择 |
| `cu_seqlens_q` | ✓（[B+1]） | — | query 累计长度，`cu_seqlens_q[i+1]-cu_seqlens_q[i]` 为第 i 个 seq 的 query 数 |
| `cu_seqlens_k` | ✓（[B+1]） | — | KV 累计长度，前缀缓存时 `cu_k[i] > cu_q[i]` |
| `max_seqlen_q` | ✓ | — | 批次最大 query 长度，FlashAttention 优化分块用 |
| `max_seqlen_k` | ✓ | — | 批次最大 KV 长度 |
| `slot_mapping` | ✓（[total_tokens]） | ✓（[bs]） | token → KV cache slot，-1 表示 dummy |
| `context_lens` | — | ✓（[bs]） | 每 seq 的有效 KV 长度（历史 + 当前） |
| `block_tables` | 有前缀时（[B, max_blocks]） | ✓（[bs, max_blocks]） | 分页地址映射，-1 填充 |

### 7.3 Context 与 CUDA Graph 的交互

```
capture_cudagraph 期间：
  set_context(is_prefill=False, slot_mapping=slot_mapping_static, ...)
  graph 录制 model.forward(input_ids_static, positions_static)
  → Attention.forward() 内 get_context() 返回静态张量（被录制进 graph）

replay 期间：
  # 直接修改静态张量的数据（不改变地址）
  graph_vars["slot_mapping"].fill_(-1)
  graph_vars["slot_mapping"][:bs] = new_slot_mapping
  → graph.replay() 时 Attention 读到的是新数据，因为 CUDA Graph 硬编码了指针
```

这是 CUDA Graph 的精妙之处：graph 内存储的是 GPU 内存地址（指针），而不是值。修改静态张量的值等于修改 graph 读取的输入数据，无需重新录制。

---

## 8. Token 采样（Gumbel-max Trick）

### 8.1 标准 Categorical 采样的问题

`torch.multinomial(probs, 1)` 内部需要计算累积分布函数（CDF）并做二分查找，本质是串行的，对于大 vocab_size（151936）批量采样效率低。

### 8.2 Gumbel-max 等价定理

**定理**：若 $G_i \sim \text{Gumbel}(0,1)$（标准 Gumbel 分布），则：
$$\arg\max_i (\log p_i + G_i) \sim \text{Categorical}(p)$$

**代码使用的等价形式**（Exponential 采样）：

标准 Gumbel 分布可由 $G = -\log(\text{Exponential}(1))$ 采样，因此：
$$\arg\max_i (\log p_i - \log U_i) = \arg\max_i (p_i / U_i), \quad U_i \sim \text{Exp}(1)$$

```python
# sampler.py
logits = logits / temperature                      # 温度缩放
probs = F.softmax(logits, dim=-1)                 # [bs, vocab_size]
noise = torch.empty_like(probs).exponential_(1)   # U ~ Exp(1)，in-place 高效
noise.clamp_min_(1e-10)                            # 防 log(0)
token_ids = (probs / noise).argmax(dim=-1)         # [bs]
```

**为什么等价于 categorical(p)？**

直觉：概率高的词，分子 $p_i$ 大；加入 $1/U_i$（随机放大），高概率词获得更大值的概率恰好等于其原始概率。

**完全向量化**：整个 `probs / noise` 是一次 elementwise 除法（充分利用 GPU 并行），`argmax` 也是高效 reduction。`@torch.compile` 将 softmax + exponential + div + argmax 融合为单个 kernel，消除中间张量。

### 8.3 Temperature Scaling 的数学意义

$$\text{softmax}(\mathbf{l}/T)_i = \frac{e^{l_i/T}}{\sum_j e^{l_j/T}}$$

- $T \to 0$：所有概率集中到最大 logit（等价 argmax，确定性生成）
- $T = 1$：原始模型分布
- $T \to \infty$：均匀分布（完全随机）

---

## 9. Fused Add-RMSNorm

### 9.1 Pre-LN Transformer 的显存带宽瓶颈

Pre-LN（Pre-Layer Normalization）Transformer 每个子层：

```
# 朴素实现（2次 HBM 读写）
residual = hidden_states + residual              # ① HBM 读 2个张量，写 1个
normed   = rms_norm(residual)                    # ② HBM 读 1个，写 1个（+读 weight）
```

每次 HBM 读写约 `bs × seq_len × hidden_size × dtype_bytes`：
```
bs=8, seq_len=512, hidden=2048, bf16：8×512×2048×2 = 16 MB
28 层 × 2 个 sub-layer × 2 次 = 112 次 HBM 读写 ≈ 1.8 GB
```

这是 HBM 带宽（2 TB/s）而非算力（312 TFLOPS）的瓶颈。

### 9.2 Fused Add-RMSNorm

合并两次 HBM 访问为一次：

```python
# layernorm.py: add_rms_forward()
def add_rms_forward(x, residual, weight, eps):
    orig_dtype = x.dtype
    x = x.float()
    residual = residual.float()

    # ① 就地残差相加（不触发新的 HBM 写回）
    x = x + residual              # x 现在是 x + residual

    # ② 同时保存 residual（下层用），在同一 pass 内完成
    residual = x.to(orig_dtype)   # 更新 residual（用于下一层的残差连接）

    # ③ RMS 归一化（就地在 x 上）
    var = x.pow(2).mean(-1, keepdim=True)
    x_normed = (x * torch.rsqrt(var + eps)).to(orig_dtype) * weight

    return x_normed, residual     # 一次读写同时完成两个任务
```

**节省**：原本 4 次 HBM 操作（读 x + 读 residual + 写 residual + 写 normed），现在 2 次（读 x + 读 residual，写 normed + 写 residual），节省 ~50% 归一化相关带宽。

### 9.3 与 @torch.compile 的协同

`torch.compile` 将 `x + residual`、`.pow(2)`、`.mean()`、`.rsqrt()`、`* weight` 识别为连续的 elementwise/reduction 算子，融合为一个 Triton kernel，进一步消除中间张量。
