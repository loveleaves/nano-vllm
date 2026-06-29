# Sleep Mode 在 nano-vllm 中的实现分析

## 一、vLLM Sleep Mode 核心理解

### 1.1 设计动机

RLHF 训练流水线中，推理与训练无法同时占用 GPU：
```
传统方案（慢）：停止推理进程 → 训练 → 重启推理（数十秒开销）
Sleep Mode：  推理 sleep → 训练（复用同一进程）→ wake_up → 推理
```

关键收益：**进程不退出**，避免模型重加载、CUDA 编译、CUDA Graph 重录制的冷启动开销。

### 1.2 vLLM 的物理基础：CUDA VMM

vLLM 使用 CUDA 虚拟内存管理 API（`cuMemCreate` / `cuMemMap` / `cuMemUnmap`），将虚拟地址与物理内存页的绑定关系解耦：

```
标准 cudaMalloc：
  [预留虚拟地址] + [分配物理页] + [建立映射]  ← 三步合一，不可分离

CUDA VMM：
  cuMemAddressReserve  → 只预留虚拟地址（无物理内存）
  cuMemCreate          → 只分配物理内存句柄
  cuMemMap             → 建立映射（地址 → 物理页）
  cuMemUnmap           → 解除映射（地址保留，物理页释放）
```

这样 sleep 时可以**释放物理内存但保持虚拟地址不变**，CUDA Graph 中硬编码的地址仍然有效，wake_up 时只需重新映射物理页，无需重录 Graph。

### 1.3 vLLM 的五层架构与三级睡眠

```
用户 API: LLM.sleep(level) / LLM.wake_up(tags)
    ↓
EngineCore: 暂停调度器，drain in-flight requests
    ↓
Executor: collective_rpc 广播命令到所有 Worker
    ↓
Worker: 调用 model.sleep/wake_up
    ↓
CuMemAllocator: 按 tag 选择性 unmap/remap 物理内存
    ↓
C扩展: cuMemUnmap / cuMemMap（CUDA Driver API）
```

| 等级 | 权重 | KV Cache | 适用场景 |
|------|------|----------|---------|
| 0 | 保留 GPU | 保留 GPU | 仅暂停调度 |
| 1 | 备份到 CPU pinned memory | 丢弃 | RLHF 权重不变 |
| 2 | 丢弃（无 CPU 备份） | 丢弃 | RLHF 权重将被更新 |

**分步唤醒（RLHF 关键）：**
```python
llm.sleep(level=2)                  # 释放所有 GPU 显存
llm.wake_up(tags=["weights"])       # 仅恢复权重的虚拟地址→物理映射
trainer.load_new_weights(model)     # 训练框架原地写入新权重
llm.wake_up(tags=["kv_cache"])      # 恢复 KV Cache
llm.generate(...)                   # 继续推理（无冷启动）
```

---

## 二、nano-vllm 架构与 vLLM 的关键差异

### 2.1 显存布局（实测 Qwen3-1.7B，RTX 3060 Ti 8GB）

```
GPU 总显存: 8.6 GB
├── 模型参数:   4.06 GB  (bfloat16, named_parameters)
├── 模型 Buffer:  21 MB  (cos_sin_cache, bfloat16→float32)
├── KV Cache:   ~3.5 GB  (torch.empty, 由 allocate_kv_cache 按剩余显存分配)
└── 激活值:     ~0.5 GB  (forward 过程的中间 tensor)
```

### 2.2 nano-vllm 无 CUDA VMM——根本差异

nano-vllm 使用 `torch.set_default_device("cuda")` + 标准 `torch.empty()`，所有 tensor 走 PyTorch 默认分配器（cudaMalloc），**没有虚拟地址与物理内存的分离**。

这意味着：
- **优点**：代码简单，无需 C 扩展
- **限制**：sleep 后内存地址改变，CUDA Graph 必须**重新录制**

### 2.3 nano-vllm 各层职责对应

| vLLM 层 | nano-vllm 对应 | 差异 |
|---------|----------------|------|
| C 扩展 + CuMemAllocator | 不需要（标准 PyTorch 内存） | 用 `.cuda()/.cpu()` 替代 VMM |
| Worker | `ModelRunner` | 直接操作，无 Executor 中间层 |
| Executor | 无（SharedMemory 广播） | 功能内嵌于 `call()` 机制 |
| EngineCore | `LLMEngine.step()` + `Scheduler` | 需要增加 sleeping 状态 |
| 用户 API | `LLM.generate()` | 需要增加 `sleep()`/`wake_up()` |

### 2.4 nano-vllm 特有的技术约束

#### 约束 1：RotaryEmbedding 的 `lru_cache` 单例

```python
# rotary_embedding.py
@lru_cache(1)
def get_rope(head_size, rotary_dim, max_position, base):
    return RotaryEmbedding(...)  # cos_sin_cache 是 GPU buffer
```

- `RotaryEmbedding` 的 `cos_sin_cache` 是 `register_buffer`（非 parameter），不在 `named_parameters()` 中
- `lru_cache` 保证所有层共享同一实例，sleep 时需要将此 buffer 移回 CPU，wake_up 时重新放到 GPU

#### 约束 2：CUDA Graph 与 `graph_vars` 静态张量

```python
# model_runner.py
self.graph_vars = dict(
    input_ids=input_ids,       # 静态 GPU 张量
    outputs=outputs,           # CUDA Graph 录制时使用的地址
    ...
)
```

- sleep 后这些静态张量需要释放（否则占用显存）
- wake_up 后必须重新 `capture_cudagraph()`，重录所有 batch size 的 Graph

#### 约束 3：多进程协调（tensor_parallel_size > 1）

```python
# model_runner.py: call()
if self.world_size > 1 and self.rank == 0:
    self.write_shm(method_name, *args)   # 广播给 rank i
method = getattr(self, method_name)
return method(*args)
```

`sleep` 和 `wake_up` 方法只需注册到 `ModelRunner` 上，`call()` 机制会自动广播到所有 rank，**无需额外修改多进程逻辑**。

#### 约束 4：Scheduler 有 in-flight 请求

sleep 前必须处理完 running 队列（或强制抢占回 waiting），否则释放 KV cache 后这些 seq 的 block_table 指向已释放的显存。

---

## 三、nano-vllm Sleep Mode 实现方案

### 3.1 三级睡眠语义（与 vLLM 对齐）

| 等级 | 操作 | 释放显存 | 恢复代价 |
|------|------|---------|---------|
| 0 | 仅暂停调度，不移动任何显存 | 0 | 极低 |
| 1 | 权重→CPU pinned memory；删除 KV cache；删除 Graph | ~7.5 GB | 中（H2D copy + Graph 录制） |
| 2 | 直接丢弃权重（无 CPU 备份）；删除 KV cache；删除 Graph | ~7.5 GB | 高（需重新从磁盘加载权重） |

> nano-vllm 因无 CUDA VMM，level 2 wake_up 必须重新 `load_model()`，比 vLLM 代价更高。

### 3.2 分步唤醒语义

```python
# RLHF 典型用法（对应 vLLM 的 tags 参数）
llm.sleep(level=1)                      # 权重备份到 CPU，释放 GPU
trainer.update_model_weights(cpu_buf)   # 训练框架在 CPU 更新权重
llm.wake_up(tags=["weights"])           # 只恢复权重到 GPU（不分配 KV cache）
trainer.maybe_verify(llm.model)         # 可选：验证权重
llm.wake_up(tags=["kv_cache"])          # 分配 KV cache + 重录 CUDA Graph
llm.generate(...)                       # 恢复推理
```

### 3.3 新增数据流

```
sleep(level=1) 调用路径：
  LLMEngine.sleep(level)
    → scheduler: preempt 所有 running seq → waiting
    → model_runner.call("sleep", level)
        rank 0: write_shm("sleep", level) → 通知 worker 进程
        所有 rank: ModelRunner.sleep(level)
          → 保存 weights_backup（CPU pinned）
          → 保存 buffers_backup（CPU）
          → del kv_cache + 清空 Attention.k_cache/v_cache
          → del graphs, graph_vars（若非 enforce_eager）
          → torch.cuda.empty_cache()

wake_up(tags) 调用路径：
  LLMEngine.wake_up(tags)
    → model_runner.call("wake_up", tags)
        所有 rank: ModelRunner.wake_up(tags)
          if "weights" in tags:
            → 恢复 params 到 GPU（CPU→GPU copy）
            → 恢复 buffers（cos_sin_cache 等）
            → 清空 weights_backup（GC 释放 CPU 内存）
          if "kv_cache" in tags（或 tags=None）:
            → allocate_kv_cache()（重新计算并分配）
            → capture_cudagraph()（重录 CUDA Graph）
```

### 3.4 关键实现细节

#### (1) 模型参数的 CPU 备份与恢复

```python
def sleep(self, level=1):
    if level == 0:
        return

    # 备份参数到 CPU pinned memory（level=1）或直接丢弃（level=2）
    self.weights_backup = {}
    for name, param in self.model.named_parameters():
        if level == 1:
            # pin_memory() 加速后续 H2D 传输
            self.weights_backup[name] = param.data.cpu().pin_memory()
        param.data = torch.empty(0, device="cuda")   # 释放 GPU 显存

    # 备份 GPU buffers（cos_sin_cache 等 register_buffer）
    self.buffers_backup = {}
    for name, buf in self.model.named_buffers():
        self.buffers_backup[name] = buf.cpu()        # 体积小，无需 pinned
        # 将 buffer 指向空 tensor（释放 GPU 显存）
        # 需要通过路径找到对应 module 并 register_buffer
        *parts, attr = name.split(".")
        module = self.model
        for p in parts:
            module = getattr(module, p)
        module.register_buffer(attr, torch.empty(0), persistent=False)

    # 释放 KV cache
    del self.kv_cache
    self.kv_cache = None
    for module in self.model.modules():
        if hasattr(module, "k_cache"):
            module.k_cache = torch.tensor([])
            module.v_cache = torch.tensor([])

    # 释放 CUDA Graph
    if not self.enforce_eager and hasattr(self, "graphs"):
        del self.graphs, self.graph_pool, self.graph_vars

    torch.cuda.empty_cache()


def wake_up(self, tags: list[str] | None = None):
    do_weights  = tags is None or "weights" in tags
    do_kvcache  = tags is None or "kv_cache" in tags

    if do_weights:
        hf_config = self.config.hf_config
        # 恢复参数到 GPU
        for name, param in self.model.named_parameters():
            if name in self.weights_backup:
                param.data = self.weights_backup[name].to(
                    device="cuda", non_blocking=True
                )
        self.weights_backup.clear()

        # 恢复 buffers
        for name, cpu_buf in self.buffers_backup.items():
            *parts, attr = name.split(".")
            module = self.model
            for p in parts:
                module = getattr(module, p)
            module.register_buffer(
                attr, cpu_buf.to(device="cuda"), persistent=False
            )
        self.buffers_backup.clear()
        torch.cuda.synchronize()

    if do_kvcache:
        # 重新计算可用显存并分配 KV cache
        self.allocate_kv_cache()
        # 重录 CUDA Graph（因为 graph_vars 地址已变）
        if not self.enforce_eager:
            self.capture_cudagraph()
```

#### (2) Scheduler 状态清理

sleep 前必须将所有 running seq 强制抢占回 waiting，确保 KV cache 引用计数归零后才能安全释放：

```python
# scheduler.py 新增方法
def preempt_all(self):
    """sleep 前将所有 running seq 撤回 waiting，释放 KV cache。"""
    while self.running:
        self.preempt(self.running.pop())
```

#### (3) LLMEngine 的 sleep/wake_up 接口

```python
# llm_engine.py 新增方法
def sleep(self, level: int = 1):
    """
    level 0: 仅暂停，不释放显存
    level 1: 权重卸载到 CPU pinned memory，KV cache 释放
    level 2: 丢弃所有 GPU 显存（wake_up 需重新加载权重）
    """
    assert level in (0, 1, 2)
    self._sleep_level = level
    self._sleeping = True

    # 将 in-flight 请求撤回 waiting（保留请求，不丢弃）
    if level > 0:
        self.scheduler.preempt_all()

    # 通知所有 rank 执行 sleep
    self.model_runner.call("sleep", level)


def wake_up(self, tags: list[str] | None = None):
    """
    tags=None:          完整唤醒（权重 + KV cache + CUDA Graph）
    tags=["weights"]:   仅恢复权重（不分配 KV cache）
    tags=["kv_cache"]:  仅恢复 KV cache（权重已恢复）
    """
    self.model_runner.call("wake_up", tags)

    # 只有完全唤醒后才能恢复调度
    if tags is None or set(tags) >= {"weights", "kv_cache"}:
        self._sleeping = False
```

#### (4) generate() 的 sleeping 检查

```python
def generate(self, prompts, sampling_params, use_tqdm=True):
    if getattr(self, "_sleeping", False):
        raise RuntimeError("LLM is sleeping. Call wake_up() first.")
    # ... 原有逻辑
```

---

## 四、与 vLLM 实现的差异对比

| 特性 | vLLM | nano-vllm 方案 |
|------|------|----------------|
| 物理机制 | CUDA VMM（虚拟地址保留） | CPU offload（地址变化） |
| CUDA Graph | sleep 后**无需重录**（地址不变） | wake_up 后**必须重录** |
| C 扩展 | 必须（cuMemMap/Unmap） | 不需要 |
| tag 系统 | weights / kv_cache / default | weights / kv_cache |
| 分步唤醒 | 支持（partial wake） | 支持 |
| Level 2 恢复权重 | 训练框架直接写入同一地址 | 需重新 load_model() 或从外部传入 |
| Buffers 处理 | clone 到 CPU，copy_ 恢复 | 同左（处理 cos_sin_cache 等） |
| 多进程 | collective_rpc 广播 | `call()` + SharedMemory 自动广播 |
| 实现复杂度 | 高（5 层 + C 扩展） | 低（2 层，纯 Python） |

---

## 五、实现步骤（从易到难）

```
Step 1. Scheduler.preempt_all()
        — 最简单，在 scheduler.py 加一个 while 循环

Step 2. LLMEngine._sleeping 状态 + sleep()/wake_up() 接口
        — llm_engine.py 增加状态标志和两个公开方法

Step 3. ModelRunner.sleep(level=1) 核心逻辑
        — 参数备份到 CPU、KV cache 释放、Graph 删除

Step 4. ModelRunner.wake_up(tags=None) 核心逻辑
        — 参数从 CPU 恢复、重新 allocate_kv_cache、重录 Graph

Step 5. Buffers 处理（cos_sin_cache 等 register_buffer）
        — 需要按模块路径 set，略复杂但量少（21 MB）

Step 6. Level 2 支持（完整丢弃后重加载权重）
        — wake_up(["weights"]) 时调用 load_model()

Step 7. 多进程验证（tensor_parallel_size > 1）
        — sleep/wake_up 通过 call() 已自动广播，验证 NCCL 进程组在 sleep 期间不崩溃
```

---

## 六、边界情况与注意事项

### 6.1 sleep 期间的请求处理

- `_sleeping=True` 时 `generate()` 应抛异常（不排队新请求）
- waiting 队列中的请求在 wake_up 后可以继续处理（block_table 已被 preempt_all 清空，会重新分配）

### 6.2 CUDA Graph 重录制的开销

重录 CUDA Graph（`capture_cudagraph`）需要：
1. 热身一次 forward（估算显存峰值）—— 已由 `allocate_kv_cache` 前的状态保证
2. 对每个 batch size 录制一次 graph —— 约 1-3 秒

这比 vLLM 的"地址不变、无需重录"代价高，但对于分钟级的 RLHF 训练轮次，可以接受。

### 6.3 RotaryEmbedding 的 `lru_cache` 问题

`get_rope` 使用 `@lru_cache(1)` 缓存实例。wake_up 恢复 buffer 后，缓存的实例持有的 `cos_sin_cache` 将重新指向 GPU tensor，无需清除缓存。但若 `lru_cache` 持有的强引用导致 CPU buffer 无法 GC，需在 sleep 后手动清除缓存：

```python
from nanovllm.layers.rotary_embedding import get_rope
get_rope.cache_clear()  # sleep 后清除，wake_up 时重新创建
```

### 6.4 多 GPU 时 NCCL 进程组

sleep/wake_up 期间不执行 NCCL 通信，NCCL 进程组保持存在（不销毁），恢复后可直接使用。只需确保 sleep 期间不触发任何 `dist.all_reduce`。

### 6.5 与 vLLM level 2 的语义差异

vLLM level 2 的 RLHF 用法：
```
sleep(2) → 训练框架在原虚拟地址写新权重 → wake_up(["kv_cache"]) → 推理
```

nano-vllm 没有 VMM，level 2 需要：
```
sleep(2) → 训练框架在独立内存训练 → wake_up 时传入新权重 → load 到 GPU → 推理
```

可以通过增加 `wake_up(weights_dict)` 接口支持：直接从外部传入新的 state_dict，无需从磁盘重读。
