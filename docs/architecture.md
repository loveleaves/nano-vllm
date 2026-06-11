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

## 8. 技术模式与实现手法

本节总结代码中用到的非显而易见的工程模式，帮助阅读者理解"为何这样写"。

### 8.1 Monkey Patching（运行时模块替换）

**位置**：`nanovllm/engine/model_runner.py:113-114` 及 `_build_model()`

Phase 3 需要将无 KV cache 的 `Attention` 替换为 `AttentionWithKVCache`，采用了两层 patch 策略：

```python
# 第一层：模块属性替换（影响后续 import 该模块的代码）
import nanovllm.layers.attention as _attn_module
_attn_module.Attention = AttentionWithKVCache

# 第二层：实例替换（处理已导入旧类引用的 qwen3.py）
for module in model.modules():
    if isinstance(module, _Qwen3Attention):
        module.attn = AttentionWithKVCache(...)
```

**为何需要两层**：Python `from module import Name` 绑定的是值的引用，`qwen3.py` 在 `import Attention` 时已捕获旧类对象，模块属性替换对它无效。实例遍历才是真正生效的路径。模块属性替换则保证了未来若有新代码动态创建层时会拿到正确类。

这种手法在 vLLM 等推理框架中非常常见，用于在不修改模型定义的情况下切换计算后端（eager → FlashAttention → CUDA graph）。

---

### 8.2 全局推理上下文（Implicit Context Pattern）

**位置**：`nanovllm/utils/context.py`

```python
_CONTEXT = Context()          # 模块级全局单例

def set_context(...): ...     # ModelRunner 在 forward 前设置
def get_context() -> Context: # Attention / LMHead 在 forward 内读取
def reset_context(): ...      # forward 后清空，防止跨步污染
```

**生命周期**：`set_context → model.forward() → reset_context`，与单次推理步绑定。

**替代方案对比**：

| 方案 | 优点 | 缺点 |
|------|------|------|
| 全局变量（现方案） | 不修改层接口，模型代码干净 | 非线程安全（单进程可接受） |
| 逐层传参 | 显式、易调试 | 需修改所有层 forward 签名 |
| `threading.local` | 线程安全 | 引入复杂度，单进程无必要 |

---

### 8.3 参数动态属性（`weight_loader` on `nn.Parameter`）

**位置**：`nanovllm/layers/linear.py`、`nanovllm/layers/embed_head.py`、`nanovllm/utils/loader.py`

```python
# 初始化时把函数挂到参数对象上
self.weight = nn.Parameter(torch.empty(...))
self.weight.weight_loader = self.weight_loader   # 动态属性

# loader.py 调用时通过 getattr 查找，找不到则用默认
loader = getattr(param, "weight_loader", default_weight_loader)
loader(param, tensor, shard_id)
```

Python 允许在任意对象上附加属性，这里利用该特性让**参数自己决定如何接受权重**（直接 copy、按 shard_id 切片写入等），实现了权重加载逻辑的局部封装，而无需在 loader 里维护巨大的 if-else 分支。

---

### 8.4 `@lru_cache` 单例工厂

**位置**：`nanovllm/layers/rotary_embedding.py:71`

```python
@lru_cache(1)
def get_rope(head_size, rotary_dim, max_position, base) -> RotaryEmbedding:
    return RotaryEmbedding(head_size, rotary_dim, max_position, base)
```

Qwen3 所有解码层的 `Attention` 都调用 `get_rope()`，参数相同时返回同一个实例，共享 `cos_sin_cache` 张量（`[max_position, 1, head_dim]`），避免 N 层重复分配。`lru_cache(1)` 只缓存最近一次调用，适合参数固定的单模型场景。

---

### 8.5 自定义 Pickle 序列化（进程间通信优化）

**位置**：`nanovllm/engine/sequence.py:98-115`

```python
def __getstate__(self):
    # prefill：需要完整 token_ids（模型 forward 要读）
    # decode：只需 last_token（节省序列化开销）
    last_state = self.token_ids if self.is_prefill else self.last_token
    return (num_tokens, num_prompt_tokens, num_cached_tokens,
            num_scheduled_tokens, block_table, last_state)

def __setstate__(self, state):
    ...
    if isinstance(last_state, list):
        self.token_ids = last_state
    else:
        self.token_ids = []
        self.last_token = last_state
```

为 Phase 4 张量并行的多进程通信设计：decode 阶段每步只需传一个 token id，而非完整序列，显著降低进程间 pickle 开销。

---

### 8.6 类变量作共享状态

**位置**：`nanovllm/engine/sequence.py:34-35`

```python
class Sequence:
    block_size: int = 256          # 全局块大小，LLMEngine 统一修改
    counter = count()              # itertools.count()，全局自增 ID
```

`LLMEngine.__init__` 通过 `Sequence.block_size = config.kvcache_block_size` 一次性配置所有实例共享的块大小，避免在每个 `Sequence` 构造时传参。`itertools.count()` 作类变量，天然保证跨实例的 ID 唯一且有序。

---

### 8.7 Fused Add-RMSNorm 双路 `forward` 分发

**位置**：`nanovllm/layers/layernorm.py:44-51`

```python
def forward(self, x, residual=None):
    if residual is None:
        return self.rms_forward(x)           # 首层：标准 RMSNorm
    return self.add_rms_forward(x, residual) # 后续层：fused 残差+归一化
```

`Qwen3DecoderLayer` 将残差张量显式传入 `layernorm`，合并了"残差相加"和"RMS 归一化"两次内存读写为一次，节省约 50% HBM 带宽。首层 `residual=None` 走标准路径，接口统一。

---

### 8.8 Gumbel-max 向量化采样

**位置**：`nanovllm/layers/sampler.py`

```python
probs = torch.softmax(logits / temperature, dim=-1)
noise = torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)
return probs.div_(noise).argmax(dim=-1)
```

等价于 `argmax(log(probs) + Gumbel(0,1))`（因为 `Gumbel = -log(Exponential(1))`）。相比 `torch.multinomial`（大词表下串行），该实现完全向量化，可被 `torch.compile` 融合为单 kernel。

---

### 8.9 `pin_memory` + `non_blocking` 异步 H2D 传输

**位置**：`nanovllm/engine/model_runner.py:296-325`

```python
input_ids = torch.tensor(..., pin_memory=True).cuda(non_blocking=True)
positions  = torch.tensor(..., pin_memory=True).cuda(non_blocking=True)
```

锁页内存（pinned memory）允许 DMA 引擎直接访问，配合 `non_blocking=True` 使 CPU-GPU 数据传输与 CPU 计算并行，隐藏 H2D 延迟。在 prefill/decode 的数据准备阶段广泛使用。

---

### 8.10 `dataclass` + `__post_init__` 输入验证

**位置**：`nanovllm/config.py`、`nanovllm/sampling_params.py`

```python
@dataclass
class Config:
    model: str
    kvcache_block_size: int = 256
    ...
    def __post_init__(self):
        assert self.kvcache_block_size % 256 == 0
        self.hf_config = AutoConfig.from_pretrained(self.model)  # 延迟初始化
```

`__post_init__` 在 dataclass 生成的 `__init__` 末尾自动调用，用于约束校验和延迟依赖初始化（避免在纯 Python 测试中触发 `transformers` 导入）。

---

### 8.11 LM Head Prefill 优化（只算最后 token 的 logits）

**位置**：`nanovllm/layers/embed_head.py:43-46`

```python
if context.is_prefill and context.cu_seqlens_q is not None:
    last_indices = context.cu_seqlens_q[1:] - 1   # 每个序列的最后 token 位置
    x = x[last_indices].contiguous()
```

prefill 时 N 个 token 的 hidden states 只有最后一个需要映射到词表（用于采样下一个 token），提前 slice 将矩阵乘法从 `[total_tokens, vocab_size]` 缩减为 `[num_seqs, vocab_size]`，节省大量计算。

---

### 8.12 `packed_modules_mapping` 声明式权重重映射

**位置**：`nanovllm/models/qwen3.py:159-165`

```python
packed_modules_mapping = {
    "q_proj":    ("qkv_proj", "q"),
    "k_proj":    ("qkv_proj", "k"),
    "v_proj":    ("qkv_proj", "v"),
    "gate_proj": ("gate_up_proj", 0),
    "up_proj":   ("gate_up_proj", 1),
}
```

将 HuggingFace 分离权重到 nano-vllm 合并参数的映射集中声明在模型类上，`loader.py` 读取后通用处理，无需为不同模型结构写不同加载逻辑。

---

### 8.13 显存 Warmup + 动态 KV Cache 分配

**位置**：`nanovllm/engine/model_runner.py:167-208`

```python
# 1. warmup：跑一次最大 batch prefill，测量峰值显存
torch.cuda.reset_peak_memory_stats()
self._run_prefill_eager(seqs)
peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]

# 2. 用剩余显存（按 gpu_memory_utilization 比例）分配 KV cache
num_blocks = int(total * gpu_memory_utilization - used - peak + current) // block_bytes
self.kv_cache = torch.empty(2, num_layers, num_blocks, block_size, kv_heads, head_dim)

# 3. 将 kv_cache 各层切片绑定到对应 AttentionWithKVCache 实例
module.k_cache = self.kv_cache[0, layer_id]
```

通过实际运行而非静态估算来测量模型显存占用，然后将剩余显存全部分配给 KV cache，最大化 GPU 利用率。KV cache 是一个连续大张量，各层通过切片（视图）共享底层存储。

---

## 9. 依赖库技术说明

### 核心运行时依赖

| 库 | 版本要求 | 用途 |
|----|---------|------|
| **torch** | ≥2.1 | 张量计算、自动微分（推理时禁用）、CUDA 内核、`torch.compile` |
| **transformers** | — | `AutoConfig`（读取模型配置）、`AutoTokenizer`（文本编解码） |
| **safetensors** | — | 安全高效的模型权重格式，`safe_open` 支持懒加载单个张量 |
| **xxhash** | — | 前缀缓存的链式哈希（Phase 4），极高吞吐量的非加密哈希 |
| **tqdm** | — | `generate()` 的进度条与吞吐量实时显示 |
| **numpy** | — | 间接依赖（torch/transformers 使用） |

### GPU 加速可选依赖（Phase 4）

| 库 | 用途 |
|----|------|
| **flash-attn** | `flash_attn_varlen_func`（prefill varlen）、`flash_attn_with_kvcache`（decode paged）；将注意力计算的 HBM 读写从 O(n²) 降至 O(n) |
| **triton** | 自定义 CUDA kernel：KV cache scatter 写入（`triton_kernels.py`），替换 Python for 循环，完全 GPU 并行 |

### torch 内部关键 API

| API | 位置 | 作用 |
|-----|------|------|
| `F.scaled_dot_product_attention` | `attention.py`、`model_runner.py` | Phase 3 注意力（自动选择 Flash/Math/Memory-efficient 后端） |
| `torch.inference_mode()` | `model_runner.py:328` | 禁用 autograd，减少内存和计算开销 |
| `torch.cuda.mem_get_info()` | `model_runner.py:184` | 查询 GPU 可用/总显存 |
| `torch.cuda.memory_stats()` | `model_runner.py:185-187` | 精细的显存统计（peak/current allocated） |
| `torch.set_default_device` | `model_runner.py:136` | 模型初始化时将默认设备设为 cuda，避免逐层指定 |
| `register_buffer(..., persistent=False)` | `rotary_embedding.py:56` | 注册非持久化缓冲区（不随 `state_dict` 保存，不计入权重） |
| `tensor.exponential_(1)` | `sampler.py:30` | 原地生成指数分布噪声（Gumbel-max trick） |
| `nn.Parameter.data.copy_` | `linear.py` 等 | 就地复制权重，不触发 autograd |

---

## 10. 关键设计决策

### 10.1 为何 block_size = 256？

FlashAttention paged KV 接口要求 block_size 是 256 的倍数（硬件对齐）。256 也是 Triton kernel 向量化的自然粒度。

### 10.2 为何用 Gumbel-max 而非 multinomial？

`torch.multinomial` 串行采样，大词表（vocab_size=150k+）时是瓶颈。Gumbel-max 完全向量化，可被 `@torch.compile` 融合为单 kernel。

### 10.3 为何 Context 用全局变量？

模型每层 forward 签名只接受 `(input_ids, positions)`，如果把推理元数据（slot_mapping, cu_seqlens 等）逐层传递，需要修改所有层接口。全局 Context 是干净的妥协：读写有明确的生命周期（set_context → model forward → reset_context）。

### 10.4 CUDA Graph 为何只录 decode？

prefill 的输入形状（seq_len）每步不同，CUDA graph 要求静态形状，无法用于 prefill。decode 每步 batch size 在预定义集合内，适合 graph 录制。

### 10.5 为何 RowParallelLinear 只在 rank 0 加 bias？

bias 是全局值，不切分。若所有 rank 都加 bias，all_reduce 后 bias 被累加 tp_size 次。
