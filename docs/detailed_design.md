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

## 5. 测试策略

### 5.1 测试分层

| 层次 | pytest 标记 | 环境 | 目标 |
|------|------------|------|------|
| 单元测试 | `@pytest.mark.unit` | CPU，CI 自动 | 算法正确性、边界条件 |
| 集成测试 | `@pytest.mark.gpu` | GPU + 权重 | 端到端推理质量 |

### 5.2 测试文件组织（按功能分类）

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

### 5.3 运行命令

```bash
# CI（仅单元测试，无需 GPU）
pytest tests/ -m unit -v --timeout=60

# 本地全量（含 GPU 集成测试）
NANO_VLLM_MODEL=/path/to/qwen3 pytest tests/ -m "unit or gpu" -v
```
