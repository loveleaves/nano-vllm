# Worker/Executor 层对齐 — V1 现状调研

> 对照基准：本机 `/home/cb/work/vllm/vllm` @ tag `v0.15.1`（V1 架构）。

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
