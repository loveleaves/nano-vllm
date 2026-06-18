# nano-vllm 架构对齐第三轮（C + D）— 详细设计文档

> 基于 `research.md`。C = 多后端 `AttentionBackend` 抽象；D = `Worker`/`Transport` 解耦。

## Motivation

A+B 后 nano 范式已对齐 V1，但两处抽象缺失：
- **C**：`Attention.forward` 写死 `flash_attn_varlen_func` + SDPA 两分支，kernel 与元数据构造耦合在 `model_runner`，无法按硬件切换、不可独立单测后端。
- **D**：`ModelRunner` 同时是 GPU 执行器与多进程 RPC 端点（`loop/read_shm/write_shm/call`+`SharedMemory`+裸 `pickle`），职责混杂、传输无结构。

目标：引入可插拔 `AttentionBackend`（C）+ 解耦 `Worker`/`ShmTransport`（D），保持 Qwen3 端到端结果不变、测试全绿。

---

## Architecture

### 包结构（C）

```
nanovllm/layers/attention/            # 由单文件 attention.py 拆分而来
├── __init__.py        # 导出 Attention, get_attn_backend, AttentionBackend...
├── backend.py         # AttentionBackend(ABC) / AttentionImpl(ABC) / AttentionMetadataBuilder(ABC)
├── common.py          # CommonAttentionMetadata（= 现 AttentionMetadata 升格）
├── flash_attn.py      # FlashAttentionBackend / Impl / Builder（迁移现 varlen 路径）
├── torch_sdpa.py      # TorchSDPABackend / Impl / Builder（迁移现 _sdpa_unified）
├── kv_ops.py          # store_kvcache（triton + naive，原 attention.py 顶部）
├── selector.py        # get_attn_backend()：环境 dispatch + 覆盖
└── layer.py           # Attention(nn.Module)：持有 impl+builder，forward 委派
```

> `utils/context.py` 的 `AttentionMetadata` 迁移/别名为 `common.py::CommonAttentionMetadata`（保留 `AttentionMetadata` 别名向后兼容现有 import）。

### 数据流（C，单步）

```
model_runner.prepare_inputs(seqs)
  └─ 构造 CommonAttentionMetadata（query_start_loc/cu_seqlens_k/slot_mapping/block_table/max_*）
     （注：builder 由各 Attention 层在 forward 时调用，或 runner 预构造一次共享）

model.forward(input_ids, positions, common_md)
  └─ 每层 Attention.forward(q,k,v, common_md):
        md = self.builder.build(common_md)        # 后端专属元数据（两后端近恒等）
        return self.impl.forward(q,k,v, self.k_cache,self.v_cache, md)
              ├─ FlashAttnImpl: store_kvcache + flash_attn_varlen_func(block_table,cu_seqlens_k)
              └─ TorchSDPAImpl: store_kvcache + 逐序列 SDPA（右下对齐 causal）
```

> 优化：`builder.build` 每层结果相同（与层无关），可在 model_runner 构造一次注入；但为贴合 V1 分层、且开销极小（仅引用搬运），设计上每层调用 builder，builder 对 nano 两后端是恒等/轻量。

### 执行分层（D）

```
LLMEngine
  ├─ rank0: Worker(rank=0) ── ShmTransport.broadcast("run", seqs) ──┐
  │            └─ ModelRunner.run(seqs)（本地执行 + 采样）           │
  └─ rank1..N: 子进程 Worker(rank=i)                                │
               └─ loop(): ShmTransport.recv() ──► ModelRunner.run ◄─┘
                          （NCCL all_reduce 在 ModelRunner 内部）
```

`ModelRunner` 不再持有 `shm/event`，不再有 `loop/call/read_shm/write_shm`；这些移入 `Worker` + `ShmTransport`。

---

## Interfaces

### C-1 抽象基类（`attention/backend.py`）

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
    def get_builder_cls() -> type["AttentionMetadataBuilder"]: ...
    @staticmethod
    def get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_dim) -> tuple:
        return (2, num_blocks, block_size, num_kv_heads, head_dim)   # nano 默认布局

class AttentionMetadataBuilder(ABC):
    def __init__(self, **kw): ...
    @abstractmethod
    def build(self, common: CommonAttentionMetadata): ...    # → 后端专属 md（可为 common 本身）

class AttentionImpl(ABC):
    @abstractmethod
    def __init__(self, num_heads, head_dim, scale, num_kv_heads): ...
    @abstractmethod
    def forward(self, q, k, v, k_cache, v_cache, attn_md) -> torch.Tensor: ...
```

### C-2 两后端（`flash_attn.py` / `torch_sdpa.py`）

```python
class FlashAttentionBackend(AttentionBackend):
    @staticmethod
    def get_name(): return "flash_attn"
    @staticmethod
    def get_impl_cls(): return FlashAttentionImpl
    @staticmethod
    def get_builder_cls(): return FlashAttentionMetadataBuilder

class FlashAttentionImpl(AttentionImpl):
    def forward(self, q, k, v, k_cache, v_cache, md):
        if k_cache.numel() and md.slot_mapping is not None:
            store_kvcache(k, v, k_cache, v_cache, md.slot_mapping)
        if md.block_table is not None:
            k_fa, v_fa, bt = k_cache, v_cache, md.block_table
        else:
            k_fa, v_fa, bt = k, v, None
        return flash_attn_varlen_func(q, k_fa, v_fa,
            cu_seqlens_q=md.query_start_loc, max_seqlen_q=md.max_query_len,
            cu_seqlens_k=md.cu_seqlens_k,    max_seqlen_k=md.max_seq_len,
            softmax_scale=self.scale, causal=True, block_table=bt)

class TorchSDPAImpl(AttentionImpl):
    def forward(self, q, k, v, k_cache, v_cache, md):
        if k_cache.numel() and md.slot_mapping is not None:
            store_kvcache(k, v, k_cache, v_cache, md.slot_mapping)
        return self._sdpa_unified(q, k, v, k_cache, v_cache, md)   # 迁移现 _sdpa_unified
```

> 两后端的 builder 对当前 `CommonAttentionMetadata` 是恒等返回（`build(common)=common`）。保留类型以便未来后端（如需 paged-token 重排、cascade）扩展。

### C-3 selector（`selector.py`）

```python
def get_attn_backend(is_cuda: bool | None = None) -> type[AttentionBackend]:
    """环境 dispatch；NANOVLLM_ATTN_BACKEND={flash_attn,torch_sdpa} 可强制覆盖。"""
    forced = os.getenv("NANOVLLM_ATTN_BACKEND")
    if forced == "torch_sdpa": return TorchSDPABackend
    if forced == "flash_attn": return FlashAttentionBackend
    if is_cuda is None: is_cuda = torch.cuda.is_available()
    if is_cuda and HAS_FLASH_ATTN: return FlashAttentionBackend
    return TorchSDPABackend
```

### C-4 `Attention` 层（`layer.py`）

```python
class Attention(nn.Module):
    def __init__(self, num_heads, head_dim, scale, num_kv_heads):
        super().__init__()
        backend = get_attn_backend()
        self.impl = backend.get_impl_cls()(num_heads, head_dim, scale, num_kv_heads)
        self.builder = backend.get_builder_cls()()
        self.k_cache = self.v_cache = torch.tensor([])
    def forward(self, q, k, v, common_md):
        md = self.builder.build(common_md)
        return self.impl.forward(q, k, v, self.k_cache, self.v_cache, md)
```

> `model_runner.allocate_kv_cache` 继续按 `isinstance(module, Attention)` 找层、写 `module.k_cache/v_cache`（不变）。

### D-1 传输层（`engine/rpc.py`）

```python
class ShmTransport:
    """rank0↔rank>0 的结构化 RPC（SharedMemory + Event + msgspec）。"""
    def __init__(self, world_size, rank, events, create: bool): ...
    def broadcast(self, method: str, seqs: list[Sequence] | None):   # rank0
        payload = (method, [s.__getstate__() for s in seqs] if seqs else None)
        buf = msgspec.msgpack.encode(payload)
        # 写长度+buf 到 shm，set 所有 event
    def recv(self) -> tuple[str, list[Sequence] | None]:             # rank>0
        # event.wait → 读 shm → msgspec.decode → 重建 Sequence(__setstate__)
    def close(self): ...
```

> 载荷用 `Sequence.__getstate__` 的轻量元组（全 int/list[int]，msgspec 原生支持）。`exit` 等无参方法 seqs=None。

### D-2 Worker（`engine/worker.py`）

```python
class Worker:
    """封装单 rank 的生命周期与 RPC 驱动；持有纯执行器 ModelRunner。"""
    def __init__(self, config, rank, transport: ShmTransport | None):
        self.model_runner = ModelRunner(config, rank)   # 纯执行器
        self.transport = transport
        self.rank = rank
    def loop(self):                  # rank>0
        while True:
            method, seqs = self.transport.recv()
            self.execute(method, seqs)
            if method == "exit": break
    def execute(self, method, seqs):
        if method == "run":  return self.model_runner.run(seqs)
        if method == "exit": return self.model_runner.exit()
    def call(self, method, seqs=None):   # rank0：先广播再本地执行
        if self.transport: self.transport.broadcast(method, seqs)
        return self.execute(method, seqs)
```

### D-3 `ModelRunner` 瘦身

删除：`shm`/`event`/`loop`/`read_shm`/`write_shm`/`call` 及 `__init__` 里建 shm/barrier/loop 的分支。保留：`__init__`(dist init+load+warmup+allocate+capture)、`run`、`prepare_inputs`、`exit`(只留资源释放)。`exit` 的 shm unlink 移到 Transport。

### D-4 `LLMEngine` 适配

```python
# __init__：建 events + 子进程 Worker；rank0 建 Worker(transport)
# step：token_ids = self.worker.call("run", seqs)
# exit：self.worker.call("exit"); join 子进程
```

---

## State Machine

无新增请求状态机。新增 **Worker 生命周期**（rank>0）：
```
spawn → ModelRunner.__init__（NCCL/load/warmup/graph）→ barrier
      → loop(): [recv → execute]* 直到 method=="exit" → 释放 → 退出
```
rank0 Worker 无 loop，由 `LLMEngine.step` 同步驱动 `call`。

---

## Risks

| 风险 | 缓解 |
|---|---|
| store_kvcache 移入 impl 后两后端写 cache 不一致 | `kv_ops.store_kvcache` 共享；两 impl 都先写后算；单测比对 |
| CUDA graph 捕获 impl.forward 失败 | backend 在 `Attention.__init__` 绑定（forward 期不 dispatch）；FlashAttn varlen 已验证可捕获 |
| builder 每层调用增开销 | nano 两后端 build 为恒等（返回入参），无张量拷贝 |
| msgspec 无法编码 state 元组某字段 | 元组全 int/list[int]/单 int；先写往返单测；last_state 为 int 或 list[int] 均 OK |
| Worker 抽取打乱 TP 时序（barrier/shm 建序）| 保持现时序平移；TP=1 不走 transport，先验 TP=1；TP=2 条件验证 |
| flash vs sdpa 数值差异 | GPU 上容差比对（bf16 ~1e-2）；E2E 仍以 flash 为基线对齐第二轮 |
| 现有 `from nanovllm.layers.attention import Attention` 失效 | `attention/__init__.py` 重导出 `Attention`，import 路径不变 |

## Test Plan

- **Unit（C）**：
  - `test_attention_backend.py`：`get_attn_backend()` dispatch（cuda/无 flash/env 覆盖）；backend 三件套返回类型。
  - `test_attention.py`（改）：经 `Attention` 层走 SDPA impl（CPU），保留现有 7 用例 + builder 恒等性。
  - 后端一致性（GPU，标记 `gpu`）：flash vs sdpa 同输入容差一致。
- **Unit（D）**：
  - `test_rpc.py`：`ShmTransport` mock/单进程往返（encode→decode→Sequence 重建字段一致）。
  - `test_sequence.py`（沿用）：`__getstate__/__setstate__` 往返。
- **Integration**：`model_runner.prepare_inputs` 仍产出正确 `CommonAttentionMetadata`；`Worker.execute("run", seqs)` 等价旧 `call`。
- **E2E**：
  - TP=1 eager+graph：Qwen3 输出与第二轮 baseline 逐 token 一致（验收 5）。
  - `NANOVLLM_ATTN_BACKEND=torch_sdpa` GPU 跑通且与 flash 容差一致。
  - TP=2（若双卡可用）：输出与 TP=1 一致；否则记为受限。
- 全量单测全绿（验收 4）。

---

## 设计评审自答（Phase 4）

1. **合理性/漏洞**：分层完全对齐 V1（Backend/Impl/Builder/Common + Worker/ModelRunner）。唯一裁剪是 builder 恒等——合理，因两后端共享元数据；接口保留不留债。漏洞点 store_kvcache 归属已用共享 kv_ops 覆盖。
2. **扩展性**：新增后端 = 加一个 `xxx.py` 实现三件套 + selector 一行；新增传输 = 实现 `Transport` 协议。改动局部，符合开闭原则。
3. **兼容性**：对外 `LLM/generate` 不变；`from nanovllm.layers.attention import Attention` 经 `__init__` 重导出保持可用；`AttentionMetadata` 保留别名。TP 协议语义不变（仅序列化方式 pickle→msgspec）。
4. **性能**：C 仅多一次恒等 `build` 调用（无拷贝）+ 一层方法委派，可忽略；graph 路径不变。D 序列化 pickle→msgspec 更快更小；同步驱动无新增 overhead。关键路径（decode）零退化。

---

**门控：请确认设计通过。** 通过后进入 Phase 5 拆任务 + Phase 6 编码（沿用"允许临时破坏、最终收口"，TP=1 全绿 + 后端一致为收口线）。
