# nano-vllm 架构设计文档

## 1. 项目概述

nano-vllm 是一个轻量级、可学习的 LLM 推理引擎，以 Qwen3 模型为参考实现，涵盖 vLLM 核心算法的简洁再现。目标是在单/多 GPU 上高效运行大语言模型推理，同时保持代码的可读性和可扩展性。

### 核心设计目标

| 目标 | 说明 |
|------|------|
| 高吞吐 | PagedAttention + 连续批处理，最大化 GPU 利用率 |
| 低延迟 | CUDA Graph replay、FlashAttention、fused kernel |
| 内存效率 | KV cache 分页管理，消除碎片；前缀缓存复用 |
| 可扩展 | 张量并行（TP），支持多 GPU |
| 可学习 | 分阶段实现，每层抽象清晰，注释完整 |

---

## 2. 整体架构

```
┌─────────────────────────────────────────────────────────┐
│                      用户接口层                           │
│  LLM / LLMEngine                                         │
│  generate(prompts, sampling_params) → outputs            │
└───────────────────────┬─────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────┐
│                      调度层                               │
│  Scheduler                                               │
│  ├── waiting: deque[Sequence]  (待 prefill)              │
│  ├── running: deque[Sequence]  (decode 中)               │
│  └── BlockManager              (KV 块分配/释放/前缀缓存)   │
└───────────────────────┬─────────────────────────────────┘
                        │  seqs + is_prefill
┌───────────────────────▼─────────────────────────────────┐
│                      执行层                               │
│  ModelRunner (per GPU rank)                              │
│  ├── prepare_prefill / prepare_decode → Context          │
│  ├── run_model (eager / CUDA graph replay)               │
│  └── Sampler (Gumbel-max)                               │
└───────────────────────┬─────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────┐
│                      模型层                               │
│  Qwen3ForCausalLM                                        │
│  ├── VocabParallelEmbedding                              │
│  ├── N × Qwen3DecoderLayer                              │
│  │   ├── RMSNorm (fused add-norm)                       │
│  │   ├── Qwen3Attention                                 │
│  │   │   ├── QKVParallelLinear                          │
│  │   │   ├── RotaryEmbedding (lru_cache 单例)            │
│  │   │   └── Attention (FlashAttention + Triton KV写入) │
│  │   └── Qwen3MLP (SwiGLU)                             │
│  └── ParallelLMHead                                     │
└─────────────────────────────────────────────────────────┘
```

---

## 3. 核心模块说明

### 3.1 请求生命周期

```
add_request(prompt)
    → Sequence(token_ids, sampling_params) → waiting 队列
    → Scheduler.schedule() 分配 KV 块
    → ModelRunner.run() prefill
    → Scheduler.postprocess() → running 队列
    → ModelRunner.run() decode × N
    → Scheduler.postprocess() → FINISHED
    → 返回 completion_token_ids
```

**Sequence 状态机**：

```
WAITING ──schedule()──► RUNNING ──eos/max_tokens──► FINISHED
    ▲                      │
    └──preempt()───────────┘  (内存不足时退回等待队列)
```

### 3.2 KV Cache 分页管理

PagedAttention 将 KV cache 划分为固定大小的物理块（block_size=256 token/块）：

```
逻辑视图（seq）：  [tok_0, tok_1, ..., tok_255 | tok_256, ..., tok_511 | ...]
物理块（Block）：  block_id=7                   block_id=23              ...
block_table：     [7, 23, ...]                 ← seq.block_table
```

- `BlockManager` 维护全局空闲块池（FIFO deque）
- 分配时从空闲队列取块，释放时归还并保留 hash（前缀缓存）
- 前缀缓存：对已满块计算链式 xxhash，后续相同前缀的请求可直接复用

### 3.3 调度策略

```
每步 schedule()：
  1. Prefill 优先：waiting 非空时执行 prefill
     - Chunked Prefill：长 prompt 分块（只允许第一个 seq 分块）
     - 前缀缓存探测：can_allocate() 返回已缓存块数
  2. Decode：waiting 为空时对 running 中每个 seq 生成 1 token
     - 内存不足时抢占（preempt）running 末尾的 seq
```

### 3.4 模型执行

**Context（全局推理元数据）**：

通过全局变量隐式传递，避免逐层传参：

| 字段 | prefill | decode |
|------|---------|--------|
| `cu_seqlens_q/k` | ✓ (flash_attn_varlen) | ✗ |
| `max_seqlen_q/k` | ✓ | ✗ |
| `slot_mapping` | ✓ | ✓ |
| `context_lens` | ✗ | ✓ |
| `block_tables` | 有前缀时 ✓ | ✓ |

**FlashAttention 调用路径**：

```
prefill → flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, ...)
decode  → flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache, cache_seqlens, block_table, ...)
```

### 3.5 张量并行（TP）

权重切分方式：

| 层类型 | 切分维度 | 通信 |
|--------|----------|------|
| ColumnParallelLinear | 输出维（列） | 无（后接 RowParallel） |
| RowParallelLinear | 输入维（行） | all_reduce |
| QKVParallelLinear | Q/K/V 各自按 head 切分 | 无 |
| VocabParallelEmbedding | vocab 维 | all_reduce |

### 3.6 CUDA Graph

- decode 阶段录制 graph_bs = [1,2,4,8,16,...,512]
- 各 graph 共享同一内存池（graph_pool）
- replay 时修改静态张量（input_ids, positions, slot_mapping 等），不重新分配内存

---

## 4. 权重加载机制

HuggingFace 格式 → nano-vllm 格式的映射：

```
HF: q_proj, k_proj, v_proj  →  nano: qkv_proj（QKV 拼接）
HF: gate_proj, up_proj       →  nano: gate_up_proj（gate+up 拼接）
```

通过 `packed_modules_mapping` + `param.weight_loader` 实现：
- 每个参数在初始化时注册 `weight_loader` 函数
- loader.py 遍历 safetensors 文件，按映射分发到对应参数
- 参数自己负责切分（TP）和写入

---

## 5. 性能优化清单

| 优化 | 阶段 | 收益 |
|------|------|------|
| PagedAttention | Phase 1 | 内存利用率↑，消除碎片 |
| 前缀缓存 | Phase 4 | prefill 计算量↓ |
| Chunked Prefill | Phase 4 | 延迟更平稳 |
| FlashAttention | Phase 4 | HBM 读写 O(n²→n) |
| Triton KV 写入 | Phase 4 | 避免中间张量 |
| CUDA Graph | Phase 4 | decode 延迟↓ 50%+ |
| torch.compile | Phase 4 | elementwise 算子融合 |
| Fused Add-RMSNorm | Phase 4 | HBM bandwidth↓ 50% |
| 张量并行 | Phase 4 | 多 GPU 线性扩展 |
| pin_memory + non_blocking | Phase 3 | H2D 传输隐藏 |
| lru_cache RoPE | Phase 2 | 共享 cos_sin_cache |
| Gumbel-max 采样 | Phase 2 | 采样完全向量化 |

---

## 6. 目录结构

```
nanovllm/
├── __init__.py
├── config.py                  # 全局配置 dataclass
├── sampling_params.py         # 采样超参数 dataclass
├── llm.py                     # 用户接口（LLM = LLMEngine）
├── engine/
│   ├── sequence.py            # Sequence 状态机 + block 计算
│   ├── block_manager.py       # KV cache 分页管理 + 前缀缓存
│   ├── scheduler.py           # 调度器（FCFS + chunked prefill + 抢占）
│   ├── model_runner.py        # GPU 执行器（warmup/KV cache/CUDA graph）
│   └── llm_engine.py          # 引擎主入口（多进程 TP 协调）
├── layers/
│   ├── activation.py          # SiluAndMul (SwiGLU)
│   ├── attention.py           # PagedAttention + FlashAttention + Triton
│   ├── embed_head.py          # VocabParallelEmbedding + ParallelLMHead
│   ├── layernorm.py           # RMSNorm + Fused Add-RMSNorm
│   ├── linear.py              # TP linear 层族
│   ├── rotary_embedding.py    # RoPE + lru_cache
│   ├── sampler.py             # Gumbel-max 采样
│   └── triton_kernels.py      # Triton KV 写入 kernel
├── models/
│   └── qwen3.py               # Qwen3ForCausalLM 完整模型
└── utils/
    ├── context.py             # 推理上下文（全局传递）
    └── loader.py              # safetensors 权重加载
```

---

## 7. 实现阶段规划

| 阶段 | 内容 | 依赖 |
|------|------|------|
| Phase 1 | 基础数据结构（config, sequence, block_manager, scheduler） | 无 GPU |
| Phase 2 | 神经网络层（layernorm, activation, rope, attention-naive, linear, sampler, model） | 单 GPU，CPU 可测 |
| Phase 3 | 权重加载 + 完整推理（loader, model_runner, llm_engine） | 真实权重 |
| Phase 4a | FlashAttention + Triton KV 写入 | GPU |
| Phase 4b | Tensor Parallelism | 多 GPU |
| Phase 4c | CUDA Graph | GPU |
| Phase 4d | 前缀缓存 + Chunked Prefill | GPU |
| Phase 4e | Fused Add-RMSNorm + torch.compile | GPU |

---

## 8. 关键设计决策

### 8.1 为何 block_size = 256？

FlashAttention paged KV 接口要求 block_size 是 256 的倍数（硬件对齐）。256 也是 Triton kernel 向量化的自然粒度。

### 8.2 为何用 Gumbel-max 而非 multinomial？

`torch.multinomial` 串行采样，大词表（vocab_size=150k+）时是瓶颈。Gumbel-max 完全向量化，可被 `@torch.compile` 融合为单 kernel。

### 8.3 为何 Context 用全局变量？

模型每层 forward 签名只接受 `(input_ids, positions)`，如果把推理元数据（slot_mapping, cu_seqlens 等）逐层传递，需要修改所有层接口。全局 Context 是干净的妥协：读写有明确的生命周期（set_context → model forward → reset_context）。

### 8.4 CUDA Graph 为何只录 decode？

prefill 的输入形状（seq_len）每步不同，CUDA graph 要求静态形状，无法用于 prefill。decode 每步 batch size 在预定义集合内，适合 graph 录制。

### 8.5 为何 RowParallelLinear 只在 rank 0 加 bias？

bias 是全局值，不切分。若所有 rank 都加 bias，all_reduce 后 bias 被累加 tp_size 次。
