# Worker/Executor 层对齐 — V1 现状调研

> 对照基准：本机 `/home/cb/work/vllm/vllm` @ tag `v0.15.1`（V1 架构）。

## 背景：什么是 Executor，为什么要抽象

**问题**：张量并行（TP）下，一个模型被切到多张 GPU、每张由一个 Worker 进程驱动。"怎么把一步
推理下发到 1 个还是 N 个 Worker、怎么收集结果"这套编排，不应散落在 EngineCore 或 ModelRunner 里。

**核心思想——Executor 抽象**：在 EngineCore 与 Worker 之间加一层 `Executor`，对上暴露统一接口
（`execute_model` / `collective_rpc` / `shutdown`），对下隐藏并行细节。两种实现：
- **UniProcExecutor**（TP=1）：单 Worker 跑在本进程，直接函数调用，无 RPC、无 barrier。
- **MultiProcExecutor**（TP>1）：spawn 各 rank 子进程，经共享内存广播指令 + 收集结果。

EngineCore 只管 `executor.execute_model(seqs)`，不关心背后是 1 个还是 N 个进程。Worker 瘦身为
纯单 rank 执行单元，进程编排（NCCL init / barrier / shm 生命周期）上移到 Executor。

**作用 / 收益**：单卡/多卡走同一上层代码；并行后端可替换（MultiProc / 未来 Ray）；是进程隔离（K）
与 EngineCore 进程化（P）的承接层。

## V1 执行器 / Worker 组件

| 文件 | 职责 | nano 对应 |
|---|---|---|
| `v1/executor/abstract.py` | `Executor(ABC)`：`get_class` 按 backend 分派；`collective_rpc`(abstract) / `execute_model` / `determine_available_memory` / `shutdown` | `executor/abstract.py`（最小子集） |
| `v1/executor/uniproc_executor.py` | `UniProcExecutor`：单进程单 Worker，collective_rpc 本地直调 | `executor/uniproc_executor.py` |
| `v1/executor/multiproc_executor.py` | `MultiprocExecutor`：spawn 多 worker 进程 + ZMQ collective_rpc | `executor/multiproc_executor.py`（ShmTransport 广播 + rank0 本地执行） |
| `v1/executor/ray_distributed_executor.py` | Ray 多节点 | ❌ 不引入 |
| `v1/worker/gpu_worker.py` | `Worker`：单 rank，包 ModelRunner，`execute_model` / `determine_available_memory` / `load_model` ... | `engine/worker.py`（execute(run/exit)） |
| `v1/worker/gpu_model_runner.py` | GPU 前向 + 采样 | `engine/model_runner.py`（已有） |

## 关键观察

1. **Executor 是 EngineCore 与 Worker 之间的编排层**：EngineCore 只调 `executor.execute_model`，
   不感知单进程还是多进程 TP。后端（UniProc/Multiproc/Ray）由 `get_class` 按配置选择。
2. **collective_rpc 是统一原语**：`execute_model` / `determine_available_memory` / `shutdown`
   都是 `collective_rpc(method)` 的薄包装——"对所有 rank 下发同一方法并收集结果"。
3. **Worker 是纯单 rank**：V1 的 Worker 只管本 rank 执行，不含"广播给其他 rank"的逻辑——
   那是 Executor 的职责。nano 对齐前把广播/barrier/shm 协调揉在 `Worker.call/_exit/__init__` 里，
   这是本轮主要搬运点。
4. **rank0 落点差异（保留 nano 现状）**：V1 multiproc 把所有 rank（含 rank0）都放进子进程，
   引擎进程不持模型。nano 保持 rank0 在引擎进程内本地执行 + 广播给 rank1..N，改动更小、
   单卡零通信开销；本轮只把这套编排从 Worker 上移到 MultiProcExecutor，不重排 rank0 落点。
