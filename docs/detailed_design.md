# nano-vllm 详细设计文档

> 本文档随各阶段实现进度持续更新，记录 nano-vllm 各核心模块的设计决策、数据结构约束与接口规范。

---

## 1. 项目概述

nano-vllm 是对 vLLM 核心推理算法的轻量级重实现，以 Qwen3 为参考模型，通过清晰的代码展示现代 LLM 推理引擎的核心技术。

### 1.1 设计原则

| 原则 | 说明 |
|------|------|
| 正确性优先 | 每个核心算法有单元测试覆盖，CI 全量通过 |
| 显式性能 | 关键路径明确标注，优化选择有文档依据 |
| 分层解耦 | 接口层/调度层/执行层/模型层职责独立 |
| 渐进实现 | 四阶段从纯 Python 到完整工程优化 |

### 1.2 四阶段实现路线

| 阶段 | 内容 | 依赖 |
|------|------|------|
| Phase 1 | 基础数据结构（Config/Sequence/BlockManager/Scheduler） | 纯 Python |
| Phase 2 | 神经网络层（RMSNorm/Attention/Linear/Sampler/Qwen3） | PyTorch CPU |
| Phase 3 | 权重加载 + 单进程推理流水线 | safetensors, GPU |
| Phase 4 | 工程优化（前缀缓存/Chunked Prefill/TP/CUDA Graph） | CUDA, NCCL |

---

## 2. 系统架构

```
┌──────────────────────────────────────────────────────┐
│  LLM API（接口层）                                     │
│    LLM.generate(prompts, sampling_params)             │
├──────────────────────────────────────────────────────┤
│  LLMEngine（引擎层）                                   │
│    add_request / step / generate                      │
├──────────────────────────────────────────────────────┤
│  Scheduler（调度层）                                   │
│    FCFS + prefill 优先 + 内存约束                      │
├──────────────────────────────────────────────────────┤
│  ModelRunner（执行层）                                 │
│    prepare_prefill/decode → model forward → sample    │
├──────────────────────────────────────────────────────┤
│  Qwen3ForCausalLM（模型层）                            │
│    Embedding → Decoder Layers × N → LMHead           │
└──────────────────────────────────────────────────────┘
```

---

## 3. Phase 1：基础数据结构

### 3.1 Config（推理配置）

**设计目标**：集中管理推理引擎所有超参数，作为只读配置在各子系统间传递。

**字段规范**：

| 字段 | 类型 | 默认值 | 约束 | 说明 |
|------|------|--------|------|------|
| model | str | - | 目录存在 | HF 模型权重路径 |
| max_num_batched_tokens | int | 16384 | >0 | 单步最大 token 预算 |
| max_num_seqs | int | 512 | >0 | 最大并发序列数 |
| max_model_len | int | 4096 | >0 | 最大序列长度 |
| gpu_memory_utilization | float | 0.9 | (0,1] | GPU 显存使用比例 |
| tensor_parallel_size | int | 1 | [1,8] | TP 并行度 |
| enforce_eager | bool | False | - | 禁用 CUDA Graph |
| kvcache_block_size | int | 256 | 256的倍数 | KV Cache 块大小 |
| num_kvcache_blocks | int | -1 | 运行时填充 | KV Cache 总块数 |

**block_size=256 的依据**：
- 平均内部碎片 128 token，对 2K-32K context 可接受
- 与 FlashAttention 分块对齐（64/128/256）
- 哈希缓存粒度合理：256 token ≈ system prompt 的 ½ 块

**初始化流程**：
1. 参数验证（断言约束）
2. `AutoConfig.from_pretrained(model)`（懒加载，失败时静默跳过）
3. `max_model_len = min(max_model_len, hf_config.max_position_embeddings)`

### 3.2 SamplingParams（采样参数）

**设计目标**：每次生成请求的独立采样配置，与 Sequence 绑定，不可变。

| 字段 | 类型 | 默认值 | 约束 |
|------|------|--------|------|
| temperature | float | 1.0 | >0（排除 greedy） |
| max_tokens | int | 64 | ≥1 |
| ignore_eos | bool | False | - |

### 3.3 Sequence（请求状态机）

**设计目标**：表示一次完整推理请求的生命周期状态，是调度器与执行层的核心操作对象。

**状态转换**：
```
WAITING ──allocate──→ RUNNING ──eos/max_tokens──→ FINISHED
    ↑                     │
    └──────preempt─────────┘  （Phase 4 抢占）
```

**关键字段**：

| 字段 | 类型 | 说明 |
|------|------|------|
| seq_id | int | 全局唯一 ID（类级计数器自增） |
| token_ids | list[int] | 完整 token 序列（prompt + 生成） |
| block_table | list[int] | 逻辑块 → 物理块 ID 映射 |
| num_cached_tokens | int | 已写入 KV Cache 的 token 数 |
| num_scheduled_tokens | int | 当前步待处理 token 数（scheduler 设置） |
| is_prefill | bool | 当前是否在 prefill 阶段 |

**属性计算**：

```python
num_blocks = ceil(num_tokens / block_size)
last_block_num_tokens = num_tokens - (num_blocks - 1) * block_size
```

**Pickle 优化**（Phase 4 多进程 IPC）：
- prefill：序列化完整 `token_ids`（模型 forward 需要）
- decode：只序列化 `last_token`（大幅减少传输量）

### 3.4 Block（KV Cache 物理块）

**字段**：

| 字段 | 说明 |
|------|------|
| block_id | 物理块编号（0 ~ num_blocks-1） |
| ref_count | 引用计数（前缀共享时 >1） |
| hash | 链式哈希值（Phase 4，-1 表示未哈希） |
| token_ids | 存储的 token 序列（哈希碰撞校验） |

### 3.5 BlockManager（KV Cache 分配器）

**数据结构**：
```
free_block_ids : deque[int]   # FIFO 空闲池（左取右还）
used_block_ids : set[int]     # 当前引用块集合
blocks         : list[Block]  # 所有块（下标 = block_id）
```

**分配策略**：FIFO（先进先出）
- 最近释放的块排在队列末尾，延迟复用，为前缀缓存保留更长有效期（Phase 4）

**接口规范**：

| 接口 | Phase 1 返回 | Phase 4 返回 |
|------|------------|------------|
| `can_allocate(seq)` | 0（可分配）或 -1（OOM） | num_cached_blocks（缓存命中数） |
| `allocate(seq, n=0)` | 填充 block_table | 前 n 块复用缓存块 |
| `deallocate(seq)` | 释放所有块，清空 block_table | 同，但保留 hash |
| `can_append(seq)` | bool：decode 时是否有空闲块 | 同 |
| `may_append(seq)` | 按需分配新块（块满时） | 同 |

### 3.6 Scheduler（调度器）

**调度策略（Phase 1）**：FCFS + prefill 优先

**调度循环**：
```
schedule() → (seqs, is_prefill)
  ├─ Prefill（优先）：从 waiting 贪心调度
  │   ├─ 检查 token 预算（max_num_batched_tokens）
  │   ├─ 检查内存（can_allocate）
  │   └─ 移入 running
  └─ Decode：从 running 调度所有 seq
      ├─ 检查空闲块（can_append）
      └─ 按需追加块（may_append）

postprocess(seqs, token_ids, is_prefill)
  ├─ append_token(token_id)
  ├─ 检查 EOS 或 max_tokens
  └─ 若完成：FINISHED + deallocate
```

**死锁防护**：waiting 非空且 running 为空且内存不足 → 返回 `([], True)`，避免无限阻塞。

---

## 4. 接口设计

### 4.1 公共 API

```python
class LLM:
    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[dict]:
        # 返回：[{"text": str, "token_ids": list[int]}, ...]
```

### 4.2 内部接口

**Scheduler**：
```python
def schedule() -> tuple[list[Sequence], bool]
def postprocess(seqs, token_ids, is_prefill) -> None
def add(seq: Sequence) -> None
def is_finished() -> bool
```

**BlockManager**：
```python
def can_allocate(seq) -> int        # -1 表示 OOM
def allocate(seq, num_cached=0) -> None
def deallocate(seq) -> None
def can_append(seq) -> bool
def may_append(seq) -> None
```

---

---

## 6. Phase 2：神经网络层

### 6.1 Context（推理上下文）

**设计目标**：将单步推理的元数据（序列长度、slot 映射、KV 表）以全局变量隐式传递，避免每层手动传参。

```python
@dataclass
class Context:
    is_prefill: bool              # 当前步是 prefill 还是 decode
    cu_seqlens_q: Tensor | None   # [num_seqs+1]，prefill 时 Q 的累计长度
    cu_seqlens_k: Tensor | None   # [num_seqs+1]，prefill 时 K 的累计长度（含 prefix cache）
    max_seqlen_q: int             # 本批次最大 Q 序列长度
    max_seqlen_k: int             # 本批次最大 K 序列长度
    slot_mapping: Tensor | None   # token → KV cache 物理 slot 映射
    context_lens: Tensor | None   # [num_seqs]，decode 时每个 seq 的 KV 总长度
    block_tables: Tensor | None   # [num_seqs, max_blocks]，物理块表
```

**线程安全性**：当前实现使用进程级全局变量，单 GPU 进程内线程安全；多进程 TP 场景下每个进程各自维护独立 Context。

### 6.2 RMSNorm（均方根归一化）

**公式**：$\text{RMSNorm}(x) = \frac{x}{\sqrt{\text{mean}(x^2) + \epsilon}} \cdot w$

**关键实现**：
- 在 `float32` 精度下计算，避免 `float16` 下数值溢出
- `(x * w.float()).to(orig_dtype)` 保证输出 dtype 与输入一致
- `add_rms_forward(x, residual)`：融合残差加法 + 归一化，节省一次 HBM 读写（Phase 4 `@torch.compile` 进一步加速）

**`__call__` dispatch 规则**：
```python
def forward(self, x, residual=None):
    if residual is None:
        return self.rms_forward(x)          # 首层
    return self.add_rms_forward(x, residual) # 中间层（融合路径）
```

### 6.3 RotaryEmbedding（旋转位置编码）

**实现要点**：
- `cos_sin_cache [max_pos, 1, head_dim]`：预计算全部位置的旋转系数，inference 时按位置索引
- `get_rope(head_dim, rotary_dim, max_position, base)` 使用 `@lru_cache`：所有层共享同一实例
- 旋转在 `float32` 精度下进行，保持数值精度

### 6.4 线性层族（Linear Layer Family）

**TP 分片方案**：

| 类 | 分片维度 | weight_loader | 说明 |
|---|---|---|---|
| `ReplicatedLinear` | 无分片 | 直接 copy | Norm/Bias 等不分片参数 |
| `ColumnParallelLinear` | 行（output dim） | 按 rank 取输出切片 | Q/K/V/Gate/Up |
| `RowParallelLinear` | 列（input dim） | 按 rank 取输入切片 + all_reduce | O/Down |
| `MergedColumnParallelLinear` | 行（合并多子矩阵） | shard_id(int) 指定子矩阵 | gate_up_proj |
| `QKVParallelLinear` | 行（Q+K+V 拼接） | shard_id("q"/"k"/"v") | qkv_proj |

**weight_loader 设计**：
- 每个参数上注册 `weight_loader` 函数属性，供 `loader.py` 调用
- TP=1 时等价于普通线性层，TP>1 时自动切片

### 6.5 Attention（注意力机制）

**Phase 2（SDPA）**：使用 `torch.nn.functional.scaled_dot_product_attention` 实现 prefill，causal mask 通过 `is_causal=True` 自动应用。

**Phase 3（带 KV Cache）**：
- `Attention` 持有 `k_cache`/`v_cache` 张量引用（由 ModelRunner 注入）
- Prefill：将 KV 写入 cache，再做 prefill attention
- Decode：从 cache 读取历史 KV，做单步 decode attention

**Phase 4（FlashAttention + Triton）**：
- Prefill：`flash_attn_varlen_func`（变长序列）
- Decode：`flash_attn_with_kvcache`（分页 KV cache 直接索引）
- KV 写入：Triton `store_kvcache_kernel`（向量化 scatter，支持非连续 slot mapping）

### 6.6 Sampler（采样器）

**Gumbel-max 技巧**（等价于按概率采样）：
```python
# logits → softmax → Gumbel-max
probs = softmax(logits / temperature)
sample = argmax(probs / Exponential(1))   # 等价于 Gumbel 扰动后 argmax
```

**Phase 4 `@torch.compile`**：将整个采样计算图编译为单个 kernel，消除 Python 开销。

### 6.7 Qwen3 模型架构

```
Qwen3ForCausalLM
  └─ Qwen3Model
      ├─ embed_tokens: VocabParallelEmbedding
      ├─ layers: [Qwen3DecoderLayer × N]
      │   ├─ input_layernorm: RMSNorm
      │   ├─ self_attn: Qwen3Attention
      │   │   ├─ qkv_proj: QKVParallelLinear
      │   │   ├─ o_proj: RowParallelLinear
      │   │   ├─ q_norm / k_norm: RMSNorm（QK-Norm）
      │   │   └─ attn: Attention（含 KV Cache）
      │   ├─ post_attention_layernorm: RMSNorm
      │   └─ mlp: Qwen3MLP
      │       ├─ gate_up_proj: MergedColumnParallelLinear
      │       ├─ act_fn: SiluAndMul
      │       └─ down_proj: RowParallelLinear
      └─ norm: RMSNorm
  └─ lm_head: ParallelLMHead
```

**packed_modules_mapping**（HF 权重名 → nano-vllm 参数名）：
```python
{
    "q_proj":    ("qkv_proj", "q"),
    "k_proj":    ("qkv_proj", "k"),
    "v_proj":    ("qkv_proj", "v"),
    "gate_proj": ("gate_up_proj", 0),
    "up_proj":   ("gate_up_proj", 1),
}
```

---

## 7. Phase 3：权重加载与推理流水线

### 7.1 权重加载（load_model）

**加载流程**：
```
for each .safetensors file（排序后顺序读取）:
  for each weight_name in file:
    if weight_name 匹配 packed_modules_mapping:
      → 替换名称, 获取 shard_id
      → param.weight_loader(param, tensor, shard_id)
    else:
      → param = model.get_parameter(weight_name)
      → param.weight_loader(param, tensor)
```

**设计亮点**：
- `safetensors` 格式：lazy mmap 读取，不一次性加载所有权重到 CPU 内存
- `weight_loader` 函数属性：参数级别的自定义加载逻辑，TP 切片对上层透明

### 7.2 ModelRunner（执行器）

**职责**：
1. 初始化 NCCL 进程组（Phase 4），设置 CUDA 设备
2. 加载模型权重
3. warmup → 测量峰值显存 → 计算 KV cache 容量
4. 分配 KV cache 张量，注入各 Attention 层
5. 捕获 CUDA graph（Phase 4）
6. 提供 `run(seqs, is_prefill)` 接口

**KV cache 内存估算**：
```python
# warmup 后测量峰值
peak = memory_stats["allocated_bytes.all.peak"]
current = memory_stats["allocated_bytes.all.current"]
# 剩余显存可用于 KV cache
available = total * gpu_memory_utilization - used - peak + current
num_blocks = available // block_bytes
```

### 7.3 推理流水线

**Prefill 路径**：
```
prepare_prefill(seqs) → input_ids, positions
  设置 Context(is_prefill=True, cu_seqlens_q, cu_seqlens_k, slot_mapping, ...)
model(input_ids, positions) → hidden_states
compute_logits(hidden_states) → logits（每 seq 取最后 token）
sampler(logits, temperatures) → token_ids
```

**Decode 路径**：
```
prepare_decode(seqs) → input_ids, positions
  设置 Context(is_prefill=False, slot_mapping, context_lens, block_tables)
model(input_ids, positions) → hidden_states
compute_logits(hidden_states) → logits（全部 token）
sampler(logits, temperatures) → token_ids
```

---

## 8. Phase 4：工程优化

### 8.1 前缀缓存（Prefix Caching）

**算法**：链式 xxhash
```python
hash(block_i) = xxhash(tokens_in_block_i + prev_block_hash)
```

**命中检测**（`can_allocate`）：
- 遍历除最后块外的满块，计算链式哈希
- 查 `hash_to_block_id`，验证 `token_ids`（防碰撞）
- 连续命中才增加 `num_cached_blocks`，一旦未命中即停止

**引用计数管理**：
- 多 seq 共享同一缓存块时 `ref_count > 1`
- 释放时不立即删除哈希，延迟到物理块被复用时清理
- `_allocate_block` 复用已有哈希块时，主动删除旧哈希避免脏命中

### 8.2 Chunked Prefill

**策略**：
- 只允许 waiting 队列的第一个 seq 分块（避免其他 seq 饥饿）
- `num_scheduled_tokens = min(num_remaining_tokens, budget_remaining)`
- 分块中途不追加新 token（`postprocess` 中通过 `num_cached_tokens < num_tokens` 判断）

### 8.3 张量并行（Tensor Parallelism）

**进程通信**：
```
rank 0（主进程）─── NCCL all_reduce ───→ rank 1..N
        ↓                    ↑
   write_shm(pickle)   read_shm()
        └──── Event.set() ──→ Event.wait()
```

**权重分片**（TP=N 时）：
- ColumnParallel: `weight[out/N*rank : out/N*(rank+1), :]`
- RowParallel: `weight[:, in/N*rank : in/N*(rank+1)]`

### 8.4 CUDA Graph

**录制策略**：
- batch sizes = [1, 2, 4, 8, 16, 32, ..., 512]
- 从大到小录制，第一次创建 memory pool，后续共享（减少碎片）
- 静态张量：prefill 时数据变化 → 录制前更新静态 buffer，replay 时直接使用

**适用条件**：
- 仅用于 decode 阶段（形状固定）
- batch size ≤ 512（超过则 eager）
- `enforce_eager=True` 时禁用

### 8.5 FlashAttention 与 Triton

**Prefill**：`flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, ...)`
- 输入为打平的变长序列，自动处理多 seq batch

**Decode**：`flash_attn_with_kvcache(q, k_cache, v_cache, block_table=..., cache_seqlens=...)`
- 直接以分页 KV cache 作为输入，避免 gather

**Triton KV 写入**：
```python
@triton.jit
def store_kvcache_kernel(q, k, v, k_cache, v_cache, slot_mapping, ...):
    # 向量化 scatter：slot_mapping → 物理 slot
    # 绕过 Python 的 for-loop，HBM 带宽利用率更高
```

---

## 9. 测试策略

### 9.1 测试分层

| 层次 | pytest 标记 | 环境 | 目标 |
|------|------------|------|------|
| 单元测试 | `@pytest.mark.unit` | CPU，CI 自动 | 算法正确性、边界条件 |
| 集成测试 | `@pytest.mark.gpu` | GPU + 权重 | 端到端推理质量 |

### 9.2 测试文件组织（按功能分类）

| 文件 | 覆盖组件 | 阶段 |
|------|---------|------|
| `test_config.py` | Config, SamplingParams | Phase 1 |
| `test_sequence.py` | Sequence 状态机 | Phase 1 |
| `test_block_manager.py` | BlockManager（基础 + 前缀缓存） | Phase 1, 4 |
| `test_scheduler.py` | Scheduler（FCFS + 分块 prefill + 抢占） | Phase 1, 4 |
| `test_normalization.py` | RMSNorm（基础 + 融合加法）| Phase 2, 4 |
| `test_activation.py` | SiluAndMul | Phase 2 |
| `test_rotary_embedding.py` | RotaryEmbedding, apply_rotary_emb | Phase 2 |
| `test_linear.py` | Linear 层族（基础 + TP 权重加载） | Phase 2, 4 |
| `test_embed_head.py` | VocabEmbedding, LMHead（基础 + 并行） | Phase 2, 4 |
| `test_sampler.py` | Sampler（Gumbel-max + torch.compile）| Phase 2, 4 |
| `test_attention.py` | Attention（SDPA + KV Cache）| Phase 2 |
| `test_context.py` | Context 工具函数 | Phase 2 |
| `test_qwen3.py` | Qwen3ForCausalLM 结构 | Phase 2 |
| `test_model_loader.py` | 权重加载（packed + default）| Phase 3 |

### 9.3 运行命令

```bash
# CI（仅单元测试，无需 GPU）
pytest tests/ -m unit -v --timeout=60

# 本地全量（含 GPU 集成测试）
NANO_VLLM_MODEL=/path/to/qwen3 pytest tests/ -m "unit or gpu" -v
```
