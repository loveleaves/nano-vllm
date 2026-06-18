# nano-vllm Worker/Executor 层对齐 V1 — 详细设计文档

> 基于 `research.md`。目标：在 EngineCore 与 Worker 之间引入 V1 风格的 **Executor 抽象**
> （UniProc / MultiProc，`get_class` 按 TP 分派），并把 Worker 瘦身为**纯单 rank 执行器**——
> 进程编排（广播 / barrier / shm 生命周期）从 Worker 上移到 Executor。

## Motivation

phase5 的 C/D 已把 RPC（ShmTransport）从 ModelRunner 拆出、Worker/ModelRunner 分层。但
`Worker` 仍混着两类职责：单 rank 执行（execute）与跨 rank 编排（`call` 广播、`__init__`/`_exit`
里的 barrier+shm 协调、rank>0 的 `loop`）。EngineCore 也直接 spawn 进程、持有 worker、调
`worker.call`。与 V1 的差距：缺 Executor 这一层，TP 规模耦合进 EngineCore。

## 范围决策（与 V1 的取舍）

| V1 特性 | 是否对齐 | 说明 |
|---|---|---|
| `Executor` ABC + `get_class` 工厂 | ✅ | TP=1→UniProc，TP>1→MultiProc |
| `collective_rpc` / `execute_model` / `shutdown` | ✅ | nano 子集 |
| `UniProcExecutor` / `MultiProcExecutor` | ✅ | 后者用 ShmTransport 广播 + rank0 本地执行 |
| Worker 瘦身为纯单 rank（execute） | ✅ | 编排上移 |
| Ray / 多节点 / 多硬件（CPU/TPU/XPU）Worker | ❌ | 无对应环境 |
| rank0 也进子进程（V1 multiproc 布局） | ❌ | 保留 nano 现状：rank0 在引擎进程内 |
| InputBatch 持久化增量更新 / LoRA·ubatch mixin | ❌ | 属 ModelRunner 层，另议 |

## Architecture

### 包结构

```
nanovllm/engine/executor/
├── __init__.py             # 导出 Executor / UniProcExecutor / MultiProcExecutor
├── abstract.py             # Executor(ABC)：get_class 工厂 + collective_rpc(abstract) + execute_model + shutdown
├── uniproc_executor.py     # UniProcExecutor：单进程单 Worker，本地直调
└── multiproc_executor.py   # MultiProcExecutor：spawn rank1..N + ShmTransport 广播；子进程入口 _worker_proc_main
nanovllm/engine/worker.py   # Worker 瘦身：仅 model_runner + execute(run/exit)
```

### 调用链

```
EngineCore.__init__:  executor = Executor.get_class(config)(config)   # 建各 rank Worker（rank0 warmup 填 num_kvcache_blocks）
EngineCore.step:      token_ids = executor.execute_model(seqs)
EngineCore.exit:      executor.shutdown()

UniProc (TP=1):   execute_model → worker.execute("run", seqs)
MultiProc(TP>1):  execute_model → transport.broadcast("run", seqs) → worker.execute("run", seqs)  # rank0
                  rank1..N 子进程：_worker_proc_main 循环 transport.recv() → worker.execute()
```

### 进程编排（从 Worker 迁入 MultiProcExecutor，顺序逐位对齐原实现）

```
启动 _init_executor:
  spawn rank1..N(_worker_proc_main)  → 各自 Worker(NCCL init) → dist.barrier()【等 shm】→ 开 shm → 收发循环
  rank0: Worker(NCCL init) → ShmTransport(create) → dist.barrier()【通知 shm 就绪】

退出 shutdown（对齐原 _exit：close → barrier →(rank0)unlink → destroy pg）:
  rank0: broadcast("exit") → transport.close() → barrier → unlink() → worker.execute("exit") → join 子进程
  rank>0: recv "exit" 跳出循环 → transport.close() → barrier → worker.execute("exit")
```

### 关键设计点

- **行为等价**：NCCL init / barrier / shm create-open / close-barrier-unlink-destroy 的相对顺序与
  对齐前 `Worker.__init__`/`_exit` 完全一致，只是落点从 Worker 移到 Executor + 子进程入口函数。
- **"exit" 语义拆分**：对齐前 `execute("exit")` 一手包办 close+barrier+unlink+destroy；现拆为
  Worker.execute("exit") 只做 `model_runner.exit()`（销毁进程组），传输的 close/barrier/unlink 由
  Executor/子进程入口在外层按序完成——避免在 barrier 前就销毁进程组。
- **Scheduler 构造时机不变**：`executor_class(config)` 内 rank0 Worker→ModelRunner warmup 仍先填好
  `config.num_kvcache_blocks`，EngineCore 之后据此建 Scheduler。
- **EngineCore 解耦 TP**：不再直接 spawn/持 worker；`self.executor` 单一依赖，`step` 调 `execute_model`。
