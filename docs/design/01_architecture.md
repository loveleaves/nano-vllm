# 架构总览

## 项目概述

nano-vllm 是一个用精简 Python 从零实现的轻量级 LLM 推理引擎，性能与 vLLM 持平甚至更快（Qwen3-0.6B 上 1434 vs 1362 tok/s）。目标是在极小代码量下还原 vLLM 的核心技术栈。

`phase5` 分支在保持单模型(Qwen3)单节点定位的前提下，按 vLLM 0.15.1（V1 架构）逐层对齐了**分层骨架**（引擎组件拆分 + 异步通路、调度器子包、KV cache 三层、Executor 抽象 + 进程隔离、持久化 InputBatch、结构化采样层、Attention 后端注册表），engine/attention/sample 合计约 3200 行。逐项对齐说明见 [nano_vs_vllm-架构对比](nano_vs_vllm-架构对比-20260618.md) 与 `arch_*/` 各专题。

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
├── __init__.py           # 对外暴露 LLM, AsyncLLM, SamplingParams, RequestOutput
├── llm.py                # LLM 入口（透传继承 LLMEngine）
├── config.py             # 全局配置 Config（含 scheduling_policy / distributed_executor_backend）
├── sampling_params.py    # 采样参数（temperature/top_p/top_k/penalties/logprobs/stop）
│
├── engine/               # 推理引擎核心（V1 风格组件拆分）
│   ├── sequence.py       # Sequence：单个请求的完整生命周期状态
│   ├── core_types.py     # EngineCoreRequest/Output(s)、RequestOutput、FinishReason 契约
│   ├── processor.py      # Processor：tokenize → EngineCoreRequest
│   ├── core.py           # EngineCore：持 Scheduler+Executor，add_request/step/abort
│   ├── detokenizer.py    # 增量 detokenize + 停止串检测
│   ├── output_processor.py  # EngineCoreOutput → RequestOutput（含 finish reason）
│   ├── llm_engine.py     # LLMEngine：同步 facade（Processor+EngineCore+OutputProcessor）
│   ├── async_llm.py      # AsyncLLM：异步 generator 流式逐步 yield
│   ├── sched/            # 调度器子包（interface/output/request_queue/scheduler）
│   ├── kv_cache/         # KV cache 子包（block_pool/kv_cache_manager/interface(Spec)）
│   ├── executor/         # Executor 抽象（abstract+get_class / uniproc / multiproc）
│   ├── worker.py         # Worker：单 rank 执行包装（run/exit/num_kvcache_blocks）
│   ├── rpc.py            # ShmTransport（广播）+ ResultChannel（回传）
│   ├── input_batch.py    # InputBatch：持久行槽位 + 增量更新 + 行回收
│   ├── block_table.py    # BlockTable + CpuGpuBuffer：常驻块表/槽位映射
│   ├── model_runner.py   # ModelRunner：GPU 前向 + 采样，管理 CUDA Graph
│   ├── scheduler.py      # 垫片 → sched 子包（向后兼容）
│   └── block_manager.py  # 垫片 → kv_cache 子包（向后兼容）
│
├── models/qwen3.py       # Qwen3 模型（Attention / MLP / Decoder / 整体）
│
├── attention/            # 注意力子系统（顶层包，对齐 v1/attention）：backend 三件套 + registry + selector + flash/sdpa + kv_ops
├── sample/               # 结构化采样层（顶层包，对齐 v1/sample）：metadata + sampler + ops{topk_topp,penalties,logprobs}
│
├── layers/               # 可复用纯神经网络层（对齐 model_executor/layers）
│   ├── linear.py         # 张量并行线性层（Column / Row / QKV / Merged）
│   └── rotary_embedding.py / activation.py / layernorm.py / embed_head.py
│
└── utils/
    ├── context.py        # AttentionMetadata（显式经 forward 链透传，非全局单例）
    └── loader.py         # safetensors 权重加载（支持 packed 权重重映射）
```

---

## 请求生命周期

```
用户调用 LLM.generate(prompts, sampling_params)            （AsyncLLM.generate 为异步流式版本）
          │
          ▼
    LLMEngine.generate()（facade，组件拆分）
    ├── Processor.process_inputs(prompt, sp) → EngineCoreRequest（tokenize）
    ├── EngineCore.add_request(req) → 构造 Sequence，加入 scheduler.waiting
    ├── OutputProcessor.add_request(req)
    └── 主循环 while EngineCore.has_unfinished_requests():
              │
              ▼
         EngineCore.step()  →  EngineCoreOutputs（每请求增量 token + 结束标志，不含文本）
         ├── sched_output = scheduler.schedule()         # 统一连续批；结构化 SchedulerOutput
         │     → prefill chunk 与 decode 混排；分配/调整 KV block_table；产 finished_seq_ids
         ├── token_ids = executor.execute_model(seqs, sched_output.finished_seq_ids)
         │     → (UniProc 内联 / MultiProc 隔离子进程) Worker.run → ModelRunner.run：
         │       InputBatch.update（增量行 + 回收）→ make_inputs（行序展开 + 单次 H2D）
         │       → run_model（eager / CUDA graph replay）→ Sampler（rank0，结构化采样）
         │       → 行序 token 按 seq_id 映射回 seqs 顺序
         └── scheduler.update_from_output(sched_output, token_ids)
               → hash_blocks（注册新填满块）；更新 num_cached_tokens
               → append_token，检查终止；完成的 seq 释放 KV 块、移出 running、记入 finished_req_ids
         │
         ▼
    OutputProcessor.process_outputs(...)  → 增量 detokenize + 停止串 → RequestOutput（文本）
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
seqs = [Sequence([0] * seq_len)]; seqs[0].num_scheduled_tokens = seq_len   # token_ids 全 0

self.run(seqs)             # 触发完整 forward，激活峰值被 CUDA 追踪
self.input_batch.clear()   # 清空 warmup 占用的持久行，真正推理从空批开始
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

> 对齐 V1 后，"单块字节数 / 显存→块数"的计算封装进 `kv_cache/interface.py::FullAttentionSpec`
> （`page_size_bytes` / `num_blocks_for_memory`），`allocate_kv_cache` 调用它，数值与上式逐位等价。

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

## 执行器与多进程架构（Executor 抽象）

EngineCore 不直接管理进程/TP，而是依赖 **Executor 抽象**（`engine/executor/`）。`Executor.get_class(config)`
按 `distributed_executor_backend` 选择后端（None 时按 TP 自动）：

| 后端 | 触发条件 | 布局 |
|---|---|---|
| `UniProcExecutor` | TP=1 且未显式指定（默认） | 单 Worker **内联**在引擎进程，`execute_model` 本地直调，无 RPC/无 barrier |
| `MultiProcExecutor` | TP>1，或显式 `distributed_executor_backend="mp"` | **所有 rank（含 rank0）皆 spawn 子进程**；引擎进程不内联 Worker、不入 NCCL 组 |

### 进程拓扑（MultiProcExecutor，完全隔离）

```
引擎进程（纯 CPU 协调，无 Worker / 无 NCCL）
  ├── LLMEngine / EngineCore / Scheduler
  └── MultiProcExecutor
        ├── ShmTransport "nanovllm"（广播：executor → 所有 worker，N 个 Event）
        └── ResultChannel "nanovllm_result"（回传：输出 rank0 → executor）
              ▲ 取回 token_ids / num_kvcache_blocks
   spawn │   ┌───────────────────────────────────────────────┐
         ▼   ▼                                               │
  worker 子进程 rank=0..N-1（各持 GPU i 的模型切片 + KV cache 切片）
        └── Worker → ModelRunner：NCCL init → warmup → allocate → cudagraph → 收发循环
              · 收 broadcast(method,seqs,finished) → execute → (rank0) 回传 ResultChannel
              · NCCL all_reduce 在模型 forward 内做张量同步（worker 之间）
```

> 对齐前 rank0 内联在引擎进程、仅 rank1..N 为子进程；**K 轮进程隔离**后所有 rank 均隔离，
> 引擎进程不再持模型/不入 NCCL，块数经 `collective_rpc("num_kvcache_blocks")` 回传。
> 默认 TP=1 仍走 UniProc 内联（零额外开销）。详见 `arch_worker_isolation/`。

### 初始化序列（MultiProc）

```
1. executor 先建两条通道（ShmTransport + ResultChannel），子进程一启动即可打开
2. spawn rank0..N-1：各自 Worker(config, rank)
     · dist.init_process_group("nccl", "tcp://localhost:2333", world_size=N, rank)
     · set_device(rank) → 建模 → load_model（TP 切分）→ warmup → allocate_kv_cache → capture_cudagraph
     · 进入收发循环
3. executor: collective_rpc("num_kvcache_blocks") → 阻塞等 rank0 回传 → 填 config.num_kvcache_blocks
   （EngineCore 随后据此构建 Scheduler 的 KVCacheManager）
```

### RPC 通信协议（ShmTransport + ResultChannel）

```
广播（executor → workers，ShmTransport）：
  encode(method, seqs, finished) = msgpack((method, [s.__getstate__()...], finished_list))
  shm.buf[0:4]=len; shm.buf[4:]=data; for e in worker_events: e.set()
  worker: event.wait() → 读 shm → decode → execute → event.clear()

回传（输出 rank0 → executor，ResultChannel）：
  worker: result.send(token_ids)  → 写 result_shm + result_event.set()
  executor: result.recv()         → wait → 读 → clear

约束（与 vLLM 多槽 MessageQueue 的差异）：
  - msgpack 替换裸 pickle：载荷为 Sequence.__getstate__ 的轻量元组（int/list[int]/采样标量）
  - 单槽 shm：靠 execute_model 同步 + "run" 内 NCCL 集体保证时序安全
  - 进程隔离下采样发生在 rank0 子进程、吃反序列化 seq，故采样标量随 __getstate__ 传输；
    但 decode 仅传 last_token，惩罚类采样在隔离模式不可用（详见 arch_worker_isolation/design.md）
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

## EngineCore 主循环

```python
def step(self) -> EngineCoreOutputs:
    sched_output = self.scheduler.schedule()                      # 统一连续批 → 结构化输出
    if sched_output.is_empty:
        return EngineCoreOutputs()
    seqs = sched_output.scheduled_seqs
    token_ids = self.executor.execute_model(seqs, sched_output.finished_seq_ids)
    self.scheduler.update_from_output(sched_output, token_ids)
    return EngineCoreOutputs(outputs=[...每请求增量 token + finish_reason...])
```

`executor.execute_model(seqs, finished_seq_ids)` 的语义按后端不同：
- **UniProc（默认）**：本进程 `Worker.execute("run", seqs, finished)` 直调 ModelRunner.run。
- **MultiProc（隔离）**：`broadcast("run", seqs, finished)` 给所有 worker 子进程 → 各 rank 执行
  （NCCL 同步张量）→ 输出 rank0 经 ResultChannel 回传 token_ids。

`seqs` 经 `Sequence.__getstate__/__setstate__` 序列化（msgpack）；finished_seq_ids 用于各 rank 的
InputBatch 回收已结束/被抢占的行槽位。

---

## 模块间依赖图

```
LLM
 └── LLMEngine（facade）
       ├── Processor                      ← tokenize → EngineCoreRequest
       ├── EngineCore
       │     ├── Scheduler (sched/)
       │     │     └── KVCacheManager (kv_cache/) → BlockPool   ← 只管逻辑（block_table/hash/引用计数）
       │     └── Executor (executor/)      ← UniProc 内联 / MultiProc 隔离子进程
       │           └── Worker → ModelRunner
       │                 ├── Qwen3ForCausalLM
       │                 │     └── Qwen3DecoderLayer × N
       │                 │           ├── Attention  ← 绑定后端 impl，持有 kv_cache 切片
       │                 │           │     └── store_kvcache (Triton) / flash_attn_varlen / SDPA
       │                 │           └── Qwen3MLP
       │                 ├── InputBatch (持久行 + 增量块表 BlockTable)
       │                 └── Sampler (sample/)  ← rank0 结构化采样
       └── OutputProcessor                 ← 增量 detokenize + 停止串 → RequestOutput
```

**关键解耦点：**
- `KVCacheManager`/`BlockPool` 与 GPU 完全解耦，只维护 `block_table` 的整数映射与前缀缓存哈希。
- `AttentionMetadata`（`utils/context.py`）**显式经 forward 链透传**（非全局单例），解耦 ModelRunner
  与 Attention 层；后端在 `Attention.__init__` 由 `get_attn_backend(head_size, dtype, device)` 绑定。
- `Executor` 把 TP 规模/进程隔离对 EngineCore 隐藏，EngineCore 只依赖 `execute_model`。
- `InputBatch` 跨步常驻，把"每步重建输入 + 整表 H2D"降为"增量行 + 单次切片 H2D"。
