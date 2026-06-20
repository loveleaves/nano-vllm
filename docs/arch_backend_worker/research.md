# nano-vllm 架构对齐第三轮（C 多后端抽象 + D Worker 解耦）— 技术调研报告

> 承接 `docs/arch_alignment` 的 A+B 成果。参照 vLLM 0.15.1 (V1)，本地检出 `/home/cb/work/vllm/vllm`。

## 摘要

V1 的 attention 分层（C）是三层抽象：

1. **`CommonAttentionMetadata`**（`v1/attention/backend.py`）：后端无关的"通用元数据"，由 model_runner 每步构造一次（`query_start_loc`/`seq_lens`/`slot_mapping`/`block_table`/`max_query_len`/`max_seq_len`…）。**nano 现有的 `AttentionMetadata` 几乎就是它**。
2. **`AttentionBackend`(ABC)**：静态工厂——`get_name()`/`get_impl_cls()`/`get_builder_cls()`/`get_kv_cache_shape()` + 一组能力查询（`supports_dtype`/`supports_head_size`…）。
3. **`AttentionMetadataBuilder.build(common)→M`** 把通用元数据转成后端专属元数据；**`AttentionImpl.forward(layer,q,k,v,kv_cache,attn_metadata)→o`** 执行 kernel。
4. **`get_attn_backend()`**（`v1/attention/selector.py`）：按硬件/dtype/可用性解析出一个 `AttentionBackend` 子类（用类名字符串 + `resolve_obj_by_qualname` 反射，带 lru_cache）。

V1 的执行分层（D）：`EngineCore`（调度，独立进程）→ `Executor` → `Worker`（`v1/worker/gpu_worker.py`，每 GPU 一个，管分布式初始化 + `execute_model(scheduler_output)`）→ `GPUModelRunner`（纯 GPU 执行）。**关键：Worker 负责进程/通信/分布式，ModelRunner 只管前向**——这正是 nano 当前 `ModelRunner` 混在一起的两件事。

结论：C 直接复用 nano 已显式化的 `AttentionMetadata`（升格为 `CommonAttentionMetadata` 角色），改动集中在 `layers/attention/`；D 把 `ModelRunner` 里的 `loop/read_shm/write_shm/call` 抽到独立 `Worker`+`Transport`，`ModelRunner` 瘦身为纯执行器。

---

## 参照实现对比

### 参照 1：`v1/attention/backend.py` — 后端抽象三件套

```python
class AttentionBackend(ABC):
    @staticmethod
    @abstractmethod
    def get_name() -> str: ...
    @staticmethod
    @abstractmethod
    def get_impl_cls() -> type["AttentionImpl"]: ...
    @staticmethod
    @abstractmethod
    def get_builder_cls(): ...          # -> type[AttentionMetadataBuilder]
    @staticmethod
    @abstractmethod
    def get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size): ...
    # + supports_dtype / supports_head_size / supports_block_size ... 能力查询

class AttentionMetadataBuilder(ABC, Generic[M]):
    @abstractmethod
    def build(self, common_prefix_len, common_attn_metadata: CommonAttentionMetadata) -> M: ...

class AttentionImpl(ABC, Generic[T]):
    num_heads: int; head_size: int; scale: float
    @abstractmethod
    def __init__(self, num_heads, head_size, scale, num_kv_heads=None, ...): ...
    @abstractmethod
    def forward(self, layer, query, key, value, kv_cache, attn_metadata, output=None) -> Tensor: ...
```

**`CommonAttentionMetadata` 字段**（与 nano 对照）：

| V1 字段 | nano 现有 `AttentionMetadata` | 说明 |
|---|---|---|
| `query_start_loc` | `query_start_loc` | ✅ 同名同义（cu_seqlens_q）|
| `seq_lens` | `cu_seqlens_k`（累计形式）| ⚠️ 语义等价，nano 因 flash 2.8.3 用累计 |
| `max_query_len` / `max_seq_len` | 同 | ✅ |
| `slot_mapping` | `slot_mapping` | ✅ |
| `block_table_tensor` | `block_table` | ✅ |
| `num_actual_tokens`/`num_reqs` | 无（可由张量推导）| 可选补充 |

**优点**：后端可插拔、能力可查询、元数据构造与 kernel 分离、可单测。
**对 nano 的启示**：nano 只需 FlashAttn + SDPA 两个后端，能力查询可极简（只保留 `get_name/get_impl_cls/get_builder_cls`）。builder 对这两个后端几乎是恒等转换（字段已齐），但保留接口为未来后端铺路。

### 参照 2：`v1/attention/selector.py` — 后端选择

```python
@lru_cache
def _cached_get_attn_backend(head_size, dtype, kv_cache_dtype, block_size, ...) -> type[AttentionBackend]:
    attention_cls = current_platform.get_attn_backend_cls(selected_backend, head_size, dtype, ...)
    return resolve_obj_by_qualname(attention_cls)   # 类名字符串 → 类
```
按 platform/env 选后端类名字符串，反射加载，lru_cache 缓存。
**对 nano 的启示**：nano 无 platform 抽象，selector 简化为：`HAS_FLASH_ATTN and torch.cuda.is_available()` → `FlashAttentionBackend`，否则 `TorchSDPABackend`。可加环境变量 `NANOVLLM_ATTN_BACKEND` 强制覆盖（便于测试两后端一致性）。

### 参照 3：`v1/worker/gpu_worker.py` — Worker / ModelRunner 分层

```python
class Worker(WorkerBase):
    def __init__(self, vllm_config, local_rank, rank, ...): ...   # 分布式 rank/device
    def init_device(self): ...                  # set_device + init_process_group
    def load_model(self): self.model_runner.load_model()
    def determine_available_memory(self) -> int: ...    # 显存探测（nano 的 allocate_kv_cache 前半）
    def compile_or_warm_up_model(self): ...     # warmup + capture graph
    def execute_model(self, scheduler_output) -> ModelRunnerOutput:
        return self.model_runner.execute_model(scheduler_output)
```
Worker = 分布式/生命周期/显存；ModelRunner = 纯前向。多进程编排在 `Executor`（MultiprocExecutor）层，用 `MessageQueue`（共享内存环形缓冲）+ msgspec 广播 RPC。

**对 nano 的启示**：nano 当前 `ModelRunner` 同时承担：①NCCL init ②模型加载 ③warmup/显存/graph ④**多进程 loop/shm/call RPC** ⑤前向执行。D 的目标是把 ④ 抽到独立单元：
- `Worker`（薄封装）：持有 `ModelRunner`，负责 rank>0 的 `loop()` 驱动与 `Transport` 交互。
- `Transport`（`ShmTransport`）：封装 `SharedMemory`+`Event`+序列化，提供 `broadcast(method,args)` / `recv()`。
- `ModelRunner`：去掉 `loop/read_shm/write_shm/call/shm/event`，只留 `__init__/load/warmup/allocate_kv_cache/capture_cudagraph/run/prepare_inputs`。

### 参照 4：RPC 序列化（裸 pickle → 结构化）

V1 用 `msgspec` 编码 `SchedulerOutput`（带自定义 encoder hook）。nano 当前：`pickle.dumps([method_name, *args])` 写入 SharedMemory，args 含 `list[Sequence]`，靠 `Sequence.__getstate__` 已把每个 seq 压成轻量元组 `(num_tokens, num_prompt_tokens, num_cached_tokens, num_scheduled_tokens, block_table, last_state)`。

**关键发现**：该元组全是 int / list[int]，**msgspec 原生可编码**。故 D-2 可行路径：RPC 直接传"方法名 + 每 seq 的 state 元组列表"，用 `msgspec.msgpack` 编解码，rank>0 用 `Sequence.__setstate__` 重建。比 pickle 更快、更结构化、无任意代码执行风险。

---

## 当前代码库分析

涉及模块与改动面：

| 文件 | 现状 | 本轮改动 |
|---|---|---|
| `layers/attention.py` | 单文件含 store_kvcache + Attention(forward 直调 flash/SDPA) | 拆为 `layers/attention/` 包：`backend.py`(抽象) / `flash_attn.py` / `torch_sdpa.py` / `selector.py` / `layer.py`(Attention 持有 impl) |
| `utils/context.py` | `AttentionMetadata` 数据类 | 升格为 `CommonAttentionMetadata` 角色（保持字段；可加 `num_reqs`）；builder 消费它 |
| `engine/model_runner.py` | 含 ①~⑤ 全部职责 | 抽出 ④ RPC；`prepare_inputs` 末尾改为 `builder.build(common)`；前向 `Attention` 不变 |
| `engine/worker.py`（新）| — | `Worker` 封装 rank>0 loop + Transport |
| `engine/rpc.py`（新）| — | `ShmTransport`（SharedMemory+Event+msgspec） |
| `engine/llm_engine.py` | 直接 new ModelRunner + 调 call | 通过 Worker/Transport 驱动 |
| `engine/sequence.py` | `__getstate__`/`__setstate__` | 复用现有 state 元组做 msgspec 载荷（可能加 `to_rpc()/from_rpc()`）|

**有利条件**：
- `AttentionMetadata` 已显式、已是"通用元数据"形态 → C 的 builder 输入现成。
- `Sequence.__getstate__` 已产出 msgspec 友好元组 → D-2 现成。
- store_kvcache 已是独立函数 → 移入 backend impl 即可。
- TP=1 路径完全不走 RPC（`world_size==1` 时 `call` 直接本地调用）→ C 可独立验证，D 仅影响 TP>1。

**风险点**：
1. **store_kvcache 归属**：移入 impl 后需保证 FlashAttn/SDPA 两 impl 都正确写 cache（SDPA 用 naive，Flash 用 triton）。缓解：抽 `write_kv` 为 backend 方法，共享底层 kernel。
2. **CUDA graph 捕获**：graph 捕获 `self.model(...)` 内部走 `Attention.forward→impl.forward`。impl 调用须 graph-capturable（FlashAttn varlen 已验证可捕获）。selector 在 `__init__` 阶段定后端，graph 期间不切换。缓解：backend 在 `Attention.__init__` 时绑定，forward 期不 dispatch。
3. **msgspec 对 Sequence 重建**：需保证 rank>0 重建的 Sequence 字段足够 prepare_inputs 使用（token_ids/last_token/block_table/num_cached/num_scheduled）。缓解：复用现有 `__setstate__`，先单测往返一致。
4. **Worker 抽取破坏 TP**：dist.barrier/进程生命周期时序敏感。缓解：保持现有时序，仅平移代码；TP=2 E2E 验证（若双卡不可用则降级为 mock transport 单测）。
5. **selector 两后端数值一致性**：SDPA 与 FlashAttn 在 CUDA 上结果应在容差内一致。缓解：GPU 上 `NANOVLLM_ATTN_BACKEND=sdpa` vs `flash` 跑同 prompt 比对（容差，bf16）。

---

## 结论与选型建议

**C（采纳，先做）**：
- 把 `layers/attention.py` 拆成 `layers/attention/` 包。
- `AttentionBackend`(ABC)：`get_name/get_impl_cls/get_builder_cls/get_kv_cache_shape`。
- `AttentionImpl`：`__init__(num_heads,head_dim,scale,num_kv_heads)` + `forward(q,k,v,k_cache,v_cache,attn_md)`。
- `AttentionMetadataBuilder.build(common)→backend_md`（两后端近恒等，保留接口）。
- `get_attn_backend()`：环境 dispatch + `NANOVLLM_ATTN_BACKEND` 覆盖。
- `Attention`(layer) 在 `__init__` 绑定 backend impl + builder，`forward` 委派 impl。

**D（采纳，后做，务实边界）**：
- 抽 `engine/rpc.py::ShmTransport`（封装 SharedMemory+Event，序列化用 `msgspec.msgpack` 编码 `[method, [seq_state,...]]`）。
- 抽 `engine/worker.py::Worker`（持有 ModelRunner，rank>0 跑 `loop()`，rank0 通过 Transport broadcast）。
- `ModelRunner` 删除所有 RPC/shm 字段与方法，成为纯执行器。
- **保留同步 `step()` 驱动**，不引入 async。

**不采纳**：platform 抽象、能力查询全集、ZMQ、msgspec 自定义 encoder hook（用现成 state 元组即可）、真异步。

**验证策略**：C 用 TP=1 E2E（flash vs sdpa 比对 + 与第二轮 baseline 逐 token 一致）；D 用 Sequence msgspec 往返单测 + ShmTransport mock 单测 +（条件允许）TP=2 E2E。

下一阶段（Phase 3）产出 `design.md`：给出 `AttentionBackend/Impl/Builder/Transport/Worker` 的接口签名骨架、包结构、数据流图与状态机。
