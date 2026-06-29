# Worker/Executor 进程隔离对齐 — V1 现状调研

> 对照基准：本机 `/home/cb/work/vllm/vllm` @ tag `v0.15.1`（V1 架构）。
> 承接 H 轮（Executor 抽象）遗留项：H 明确把"rank0 也进子进程"列为**未做**，本轮补齐。

## 背景：为什么 rank0 也要进子进程

**问题**：TP 多进程下，H 轮让 rank1..N-1 进子进程，但 **rank0 仍内联在引擎进程**里。这导致引擎
进程也要初始化 CUDA / 进 NCCL 组，与"前端保持 CUDA-free、各 rank 对称"的理想相悖，也使
fork/spawn、信号处理、异常传播变复杂。

**核心思想——所有 rank 对称隔离**：让 **rank0 也进独立子进程**，引擎进程不内联任何 Worker、不进
NCCL 组，只通过两条共享内存通道与 Worker 群通信：
- `ShmTransport`：引擎 → 各 Worker 的**广播**（下发"跑一步"指令 + 序列化的输入）。
- `ResultChannel`：输出 rank（rank0）→ 引擎的**回传**（采样出的 token / 块数等）。

**核心思想——共享内存替代 ZMQ**：vLLM 用 ZMQ；nano 用 stdlib `SharedMemory + Event` + msgspec
序列化达到同等"进程隔离 + 结构化消息"，更轻、无额外依赖。

**作用 / 收益**：各 rank 对称、引擎进程 CUDA-free；`distributed_executor_backend="mp"` 让单卡也能
跑进程隔离（便于在单 GPU 上测试隔离机制，无需多卡）。

## V1 `MultiprocExecutor` 关键结构

| 元素 | 职责 |
|---|---|
| `WorkerProc`（rank 0..N-1 全是子进程） | 每个 rank 一个独立进程，executor 进程**不持模型、不在 NCCL 组内** |
| `rpc_broadcast_mq`（MessageQueue） | executor → 所有 worker 广播 `(method, args, kwargs, output_rank)` |
| `worker_response_mq`（每 worker 一个） | worker → executor 回传结果（含状态 SUCCESS/FAILURE） |
| `output_rank` | 只从单个 rank（PP 末 rank；纯 TP 时 0）收集 `execute_model` 结果 |
| `collective_rpc(method, unique_reply_rank=...)` | enqueue 广播 → 从（指定/所有）response_mq dequeue 收集 |

## 关键观察

1. **executor 是纯协调进程**：不内联任何 rank，不 import CUDA/NCCL；通过消息队列下发指令、
   收集结果。所有计算（含 rank0）都在隔离的 worker 子进程里。
2. **双向通道**：广播（1→N）+ 每 worker 回传（N→1）。`execute_model` 只取 `output_rank`
   的结果（其余 rank 输出无意义）。
3. **KV 块数经 RPC 回传**：worker 子进程在 `determine_available_memory` / 初始化阶段算出
   可用 KV 容量，executor 通过 collective_rpc 取回，再据此让前端建调度器——因为块数算在
   worker 进程里，引擎进程必须显式拉取。
4. **UniProcExecutor 仍内联**：TP=1 非多进程模式下 worker 与 executor 同进程（V1 亦然）；
   多进程仅用于隔离/并行。

## nano 对齐前状态（H 轮）

`MultiProcExecutor` 把 rank0 **内联**在引擎进程：rank0 本地 `self.worker.execute(...)` 直接拿
token，仅 rank1..N-1 是子进程；executor 进程加入 NCCL 组、跑 rank0 的模型。`num_kvcache_blocks`
因 rank0 在本进程，allocate 后直接读 `config`。**广播单向**（ShmTransport），无回传通道。

## nano 取舍（不引入）

MessageQueue（多槽 + 确认）→ nano 用单槽 SharedMemory（详见 design.md 的单槽时序约束）；
FailureCallback / 健康监控 / SIGTERM 编排 / PP·DP·EP 的多 output_rank 聚合。
