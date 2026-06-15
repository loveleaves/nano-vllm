# 架构总览

## 项目概述

nano-vllm 是一个用 ~1200 行 Python 从零实现的轻量级 LLM 推理引擎，性能与 vLLM 持平甚至更快（Qwen3-0.6B 上 1434 vs 1362 tok/s）。目标是在极简代码量下还原 vLLM 的核心技术栈。

**依赖栈：**

| 依赖 | 版本 | 用途 |
|------|------|------|
| `torch` | ≥2.4 | 张量计算、分布式通信、CUDA Graph |
| `triton` | ≥3.0 | 自定义 GPU Kernel（KV cache 写入） |
| `transformers` | ≥4.51 | 模型配置（AutoConfig）/ Tokenizer |
| `flash-attn` | latest | 高效注意力计算（prefill + decode） |
| `xxhash` | latest | 前缀缓存块哈希（极速非加密哈希） |

---

## 目录结构与模块职责

```
nanovllm/
├── __init__.py           # 对外暴露 LLM, SamplingParams
├── llm.py                # LLM 入口（透传继承 LLMEngine）
├── config.py             # 全局配置 Config dataclass
├── sampling_params.py    # 采样参数 SamplingParams dataclass
│
├── engine/               # 推理引擎核心
│   ├── sequence.py       # Sequence：单个请求的完整生命周期状态
│   ├── block_manager.py  # BlockManager：KV cache 分页管理 + 前缀缓存
│   ├── scheduler.py      # Scheduler：调度 prefill/decode，管理抢占
│   ├── llm_engine.py     # LLMEngine：引擎主循环，协调各组件
│   └── model_runner.py   # ModelRunner：GPU 推理，管理 CUDA Graph
│
├── models/
│   └── qwen3.py          # Qwen3 模型（Attention / MLP / Decoder / 整体）
│
├── layers/               # 可复用神经网络层
│   ├── attention.py      # PagedAttention（Triton KV写入 + FlashAttention）
│   ├── linear.py         # 张量并行线性层（Column / Row / QKV / Merged）
│   ├── rotary_embedding.py  # RoPE 旋转位置编码
│   ├── activation.py     # SiluAndMul（SwiGLU 激活）
│   ├── layernorm.py      # RMSNorm（含 Fused Add-Norm）
│   ├── embed_head.py     # VocabParallelEmbedding + ParallelLMHead
│   └── sampler.py        # Token 采样（temperature + Gumbel-max）
│
└── utils/
    ├── context.py        # 全局推理上下文（进程内隐式传递）
    └── loader.py         # safetensors 权重加载（支持 packed 权重重映射）
```

---

## 请求生命周期

```
用户调用 LLM.generate(prompts, sampling_params)
          │
          ▼
    LLMEngine.generate()
    ├── 1. tokenize 每个 prompt（str → token_ids）
    ├── 2. 构造 Sequence 对象，加入 scheduler.waiting 队列
    └── 3. 主循环 while not is_finished():
              │
              ▼
         LLMEngine.step()
         ├── scheduler.schedule()
         │     → 决定本轮处理哪些 seq、做 prefill 还是 decode
         │     → 分配/调整 KV cache block_table
         ├── model_runner.call("run", seqs, is_prefill)
         │     → prepare_prefill/decode（构造输入张量 + 设置 Context）
         │     → run_model（eager forward 或 CUDA graph replay）
         │     → sampler（rank 0 采样）
         │     → reset_context
         │     返回 token_ids（新采样的 token）
         └── scheduler.postprocess(seqs, token_ids, is_prefill)
               → block_manager.hash_blocks（注册新填满块的哈希）
               → 更新 num_cached_tokens
               → append_token，检查终止条件
               → 完成的 seq 释放 KV 块，移出 running
```

---

## 请求状态机

```
              分配KV块，执行prefill
WAITING ──────────────────────────► RUNNING ──── 逐token decode ──► FINISHED
   ▲                                    │                                │
   │          内存不足，抢占preempt      │         命中EOS或达max_tokens   │
   └────────────────────────────────────┘                                │
                                                          释放KV块 ◄──────┘
```

**各状态持有的关键数据：**

| 状态 | block_table | token_ids | num_cached_tokens |
|------|-------------|-----------|-------------------|
| WAITING（首次） | 空 | 仅 prompt tokens | 0 |
| WAITING（被抢占） | 空（deallocate 已清） | prompt + 部分生成 token | 0（reset） |
| RUNNING（prefill 完成后） | 完整分配 | prompt tokens | == num_tokens |
| RUNNING（decode 中） | 完整分配 + 新块 | prompt + 已生成 tokens | 逐步累积 |
| FINISHED | 空（deallocate 已清） | 完整 token 序列 | — |

> 被抢占的 seq 重回 WAITING 后，下次调度时前缀缓存大概率命中（hash 仍在 hash_to_block_id），节省重计算开销。

---

## GPU 显存布局

nano-vllm 启动时按以下顺序占用 GPU 显存：

```
┌─────────────────────────────────────────────────────────────────┐
│ GPU 总显存（如 RTX 3060 Ti = 8.6 GB）                            │
│                                                                  │
│ ┌──────────────────────────────────────────────────────────┐    │
│ │ 模型参数（bfloat16）                         ~4.06 GB    │    │
│ │   ├─ embed_tokens.weight                    [vocab, H]   │    │
│ │   ├─ layers[0~27].qkv_proj.weight           [qkv, H]    │    │
│ │   ├─ layers[0~27].o_proj.weight             [H, H]      │    │
│ │   ├─ layers[0~27].gate_up_proj.weight       [2I, H]     │    │
│ │   ├─ layers[0~27].down_proj.weight          [H, I]      │    │
│ │   └─ layers[0~27].{layernorm, q/k_norm}     [H]         │    │
│ ├──────────────────────────────────────────────────────────┤    │
│ │ 模型 Buffer（float32，register_buffer）         ~21 MB   │    │
│ │   └─ RotaryEmbedding.cos_sin_cache          [max_len*2] │    │
│ ├──────────────────────────────────────────────────────────┤    │
│ │ 激活峰值（warmup 时测量，forward 过程中间张量） ~0.5 GB   │    │
│ ├──────────────────────────────────────────────────────────┤    │
│ │ KV Cache（统一大张量，按剩余显存计算块数）       ~3.5 GB  │    │
│ │   shape: [2, num_layers, num_blocks, block_size,         │    │
│ │           num_kv_heads, head_dim]                        │    │
│ │   例：[2, 28, N, 256, 8, 128]（Qwen3-1.7B，TP=1）       │    │
│ └──────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

**KV Cache 张量内存访问模式（关键设计）：**

```
kv_cache[0, layer_id]  →  k_cache: [num_blocks, block_size, num_kv_heads, head_dim]
kv_cache[1, layer_id]  →  v_cache: [num_blocks, block_size, num_kv_heads, head_dim]

访问第 block_id 块的第 offset 个 token 的 K：
  k_cache[block_id, offset, :, :]    # shape [num_kv_heads, head_dim]

等价展平索引：
  slot = block_id * block_size + offset
  k_cache_flat[slot * (num_kv_heads * head_dim) ...]  ← Triton kernel 使用
```

KV cache 在所有 Attention 层间共享同一块物理显存（大张量切片），避免碎片：

```python
# model_runner.py: allocate_kv_cache()
self.kv_cache = torch.empty(2, num_layers, num_blocks, block_size, num_kv_heads, head_dim)
for module in self.model.modules():
    if hasattr(module, "k_cache"):
        module.k_cache = self.kv_cache[0, layer_id]   # 零拷贝视图
        module.v_cache = self.kv_cache[1, layer_id]
        layer_id += 1
```

---

## Warmup 与显存估算

### 为什么要 Warmup？

CUDA 的显存分配器（caching allocator）会缓存已释放的内存，`torch.cuda.mem_get_info()` 报告的"空闲"是分配器视角的，不等于实际可用。同时，模型 forward 过程中会分配大量临时激活张量，其峰值是动态的。

Warmup 的作用是：**用一次真实的最大批次 forward，精确测量激活峰值显存**，从而准确计算 KV Cache 可用的剩余空间。

### Warmup 流程

```python
# model_runner.py: warmup_model()
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

# 构造最大批次的虚假 prefill
seq_len = min(max_num_batched_tokens, max_model_len)    # 例：4096
num_seqs = min(4096 // seq_len, max_num_seqs)           # 例：1
seqs = [Sequence([0] * seq_len)]                        # token_ids 全 0

self.run(seqs, True)   # 触发完整 forward，激活峰值被 CUDA 追踪
torch.cuda.empty_cache()   # 清理激活张量，KV cache 尚未分配
```

### 显存估算公式

```python
# model_runner.py: allocate_kv_cache()
free, total = torch.cuda.mem_get_info()
used   = total - free                   # 当前已用（模型参数 + buffers）
peak   = memory_stats["allocated_bytes.all.peak"]    # warmup 峰值（含激活）
current = memory_stats["allocated_bytes.all.current"] # 当前实际分配（无激活）

# 可用于 KV cache 的字节数：
# total × utilization：用户愿意使用的上限
# - used：已占用（模型参数等）
# - (peak - current)：forward 时激活张量的峰值需求（必须预留）
available = total * gpu_memory_utilization - used - (peak - current)

# 每个 KV block 的字节数（K + V，所有层，每块 block_size 个 token）：
block_bytes = 2 * num_layers * block_size * num_kv_heads * head_dim * dtype.itemsize

num_kvcache_blocks = available // block_bytes
```

**实测（Qwen3-1.7B，RTX 3060 Ti 8GB，gpu_memory_utilization=0.9）：**

```
total  = 8.6 GB
used   = 4.08 GB（参数 + buffers）
peak   = 4.58 GB（warmup 时激活峰值约 500 MB）
current = 4.08 GB
available = 8.6×0.9 - 4.08 - (4.58-4.08) = 7.74 - 4.08 - 0.5 = 3.16 GB

block_bytes = 2×28×256×8×128×2 = 1,835,008 B ≈ 1.75 MB
num_blocks = 3.16 GB / 1.75 MB ≈ 1838 个物理块
```

---

## Qwen3 模型树形结构

```
Qwen3ForCausalLM
│
├── model: Qwen3Model
│   ├── embed_tokens: VocabParallelEmbedding(vocab_size, hidden_size)
│   │     └── 词表按 tp_size 均分到各 GPU
│   │
│   ├── layers: N × Qwen3DecoderLayer
│   │   ├── input_layernorm: RMSNorm(hidden_size)
│   │   ├── self_attn: Qwen3Attention
│   │   │   ├── qkv_proj: QKVParallelLinear       ← 合并 Q/K/V，列并行
│   │   │   ├── q_norm: RMSNorm(head_dim)          ← QK-Norm
│   │   │   ├── k_norm: RMSNorm(head_dim)          ← QK-Norm
│   │   │   ├── rotary_emb: RotaryEmbedding        ← @lru_cache 单例，所有层共享
│   │   │   ├── attn: Attention                    ← PagedAttention
│   │   │   └── o_proj: RowParallelLinear          ← 行并行 + all_reduce
│   │   ├── post_attention_layernorm: RMSNorm(hidden_size)
│   │   └── mlp: Qwen3MLP
│   │       ├── gate_up_proj: MergedColumnParallelLinear([intermediate]*2)
│   │       ├── act_fn: SiluAndMul                 ← SwiGLU
│   │       └── down_proj: RowParallelLinear
│   │
│   └── norm: RMSNorm(hidden_size)                 ← 最终归一化
│
└── lm_head: ParallelLMHead(vocab_size, hidden_size)
      └── 可与 embed_tokens 共享权重（tie_word_embeddings）
```

**Qwen3-1.7B 关键尺寸（`config.json`）：**

| 参数 | 值 | 说明 |
|------|----|------|
| `hidden_size` | 2048 | H |
| `num_hidden_layers` | 28 | Transformer 层数 |
| `num_attention_heads` | 16 | Q heads |
| `num_key_value_heads` | 8 | KV heads（GQA，Q/KV = 2:1） |
| `head_dim` | 128 | Q/K/V 每个 head 的维度 |
| `intermediate_size` | 11008 | FFN 中间维度 |
| `vocab_size` | 151936 | 词表大小 |
| `max_position_embeddings` | 40960 | 最大位置编码 |

**Qwen3 特点：**
- **GQA**（Grouped Query Attention）：16 个 Q head 共享 8 个 KV head，KV cache 是 Q cache 的一半
- **QK-Norm**：对每个 Q/K head 做 RMSNorm（在 head 维度），稳定注意力分数梯度（无 `qkv_bias` 时启用）
- **RoPE**：旋转位置编码，`rope_theta=1000000`，支持 `rope_scaling` 扩展上下文窗口
- **SwiGLU**：`silu(gate) * up` 激活函数，比 GELU 少一次激活 FLOPs
- **Tied Embeddings**：`tie_word_embeddings=True`，`lm_head` 与 `embed_tokens` 共享权重，减少参数量

---

## 多进程架构（张量并行）

### 进程拓扑

```
主进程 (rank 0, GPU 0)
  ├── LLMEngine（Python 调度主循环）
  ├── Scheduler（CPU 端请求管理）
  └── ModelRunner(rank=0)
        ├── Qwen3ForCausalLM（GPU 0 上模型参数的一部分）
        ├── KV Cache 切片（GPU 0 上 num_kv_heads/tp 个 head）
        ├── Sampler（只有 rank 0 执行采样）
        └── SharedMemory "nanovllm"（1 MB，写端）
              ├── Event[0] ─── 通知 ──► 子进程 rank=1 (GPU 1)
              └── Event[1] ─── 通知 ──► 子进程 rank=2 (GPU 2)

子进程 rank=i (GPU i)
  └── ModelRunner(rank=i)
        ├── Qwen3ForCausalLM（GPU i 上模型参数的一部分）
        ├── KV Cache 切片（GPU i 上 num_kv_heads/tp 个 head）
        └── SharedMemory "nanovllm"（读端）
              ↑ event.wait() 阻塞等待 rank 0 写入
```

### 初始化序列

```
所有进程并发执行（LLMEngine 通过 multiprocessing.Process 启动子进程）：

1. dist.init_process_group("nccl", "tcp://localhost:2333", ...)
   — 所有 rank 阻塞在此，直到 tp_size 个进程全部连接
   — NCCL 在此时协商通信拓扑（NVLink、PCIe、IB 等）

2. torch.cuda.set_device(rank)   — 绑定 CUDA 设备

3. torch.set_default_device("cuda")  — 后续所有 torch.empty/zeros → GPU
   torch.set_default_dtype(hf_config.dtype)  — bfloat16

4. 构建 Qwen3ForCausalLM（在 GPU 上直接分配参数）

5. load_model(...)  — 从 safetensors 加载权重（TP 切分后，每 rank 只加载自己的份额）

6. warmup_model() → allocate_kv_cache() → capture_cudagraph()

7. rank 0: 创建 SharedMemory，dist.barrier()
   rank i: dist.barrier()，连接 SharedMemory，进入 loop() — 永久阻塞
```

### SharedMemory 通信协议（详细）

```
发送方（rank 0, write_shm）：
  data = pickle.dumps([method_name, arg1, arg2, ...])
  n = len(data)                           # 数据字节长度
  shm.buf[0:4] = n.to_bytes(4, "little") # 头 4 字节写长度
  shm.buf[4:n+4] = data                   # 后续写 pickle 数据
  for event in self.event:
      event.set()  # 同时通知所有 rank（OS 原语，跨进程）

接收方（rank i, read_shm）：
  self.event.wait()     # 阻塞，CPU 零消耗等待
  n = int.from_bytes(shm.buf[0:4], "little")
  method_name, *args = pickle.loads(shm.buf[4:n+4])
  self.event.clear()    # 复位 event，等待下一次通知
  return method_name, args

loop()：
  while True:
      method_name, args = read_shm()
      call(method_name, *args)    # 执行方法（run/allocate_kv_cache 等）
      if method_name == "exit": break

重要约束：
  - 1 MB SharedMemory 限制 args 的 pickle 大小
  - seqs 对象不走 SharedMemory（太大），而是 Sequence 内部有 __getstate__/__setstate__ 优化
  - 实际 run() 的 seqs 参数需能 pickle：Sequence 使用 __reduce__ 只传 token_ids 和元数据
```

### NCCL 通信时机

NCCL 通信只在模型 forward 的特定层发生，而不是每步调度时：

```
RowParallelLinear.forward():
    y_partial = x @ W_local.T    # 每 GPU 计算自己的分量
    dist.all_reduce(y_partial)   # NCCL ring-allreduce：各 GPU 贡献求和
    return y_partial             # 现在每 GPU 持有完整结果

ParallelLMHead.forward():
    logits_partial = x @ W_local.T  # 每 GPU 计算部分词表的 logit
    if rank == 0:
        dist.gather(logits_partial, all_logits)  # 收集到 rank 0
    else:
        dist.gather(logits_partial, dst=0)
    return cat(all_logits) if rank == 0 else None
```

**每层 Transformer 共触发 2 次 all_reduce：** o_proj（注意力输出）和 down_proj（FFN 输出）。

---

## LLMEngine 主循环

```python
def step(self) -> bool:
    seqs, is_prefill = self.scheduler.schedule()
    token_ids = self.model_runner.call("run", seqs, is_prefill)
    self.scheduler.postprocess(seqs, token_ids, is_prefill)
    return self.scheduler.is_finished()
```

`call("run", ...)` 的语义：
- rank 0：先 `write_shm("run", ...)` 通知子进程，再自己执行 `run(seqs, is_prefill)`
- rank i：在 `loop()` 中读取到 "run" 指令后执行 `run(seqs, is_prefill)`（GPU 同步通过 NCCL）
- 子进程 `run()` 的 `seqs` 参数通过 pickle/SharedMemory 传递，因此 Sequence 必须可序列化

---

## 模块间依赖图

```
LLM
 └── LLMEngine
       ├── Scheduler
       │     └── BlockManager          ← 只管逻辑（block_table、hash）
       │           └── (Block)
       └── ModelRunner (rank 0)
             ├── Qwen3ForCausalLM
             │     └── Qwen3DecoderLayer × N
             │           ├── Attention  ← 直接持有 kv_cache 切片
             │           │     └── store_kvcache (Triton)
             │           │     └── flash_attn_{varlen,with_kvcache}
             │           └── Qwen3MLP
             ├── Sampler
             ├── Context (全局单例)     ← prepare_*/run 之间隐式传递
             └── [SharedMemory → ModelRunner rank i × (tp-1)]
```

**关键解耦点：**
- `BlockManager` 与 GPU 完全解耦，只维护 `block_table` 的整数映射
- `Context` 解耦 `ModelRunner` 与 `Attention` 层的接口（不需要修改 forward 签名）
- `Attention` 层只知道"从 Context 取 slot_mapping/block_tables"，不知道调度策略
