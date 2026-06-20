# EngineCore 进程化（进程拓扑）对齐 — V1 现状调研

> 对照基准：本机 `/home/cb/work/vllm/vllm` @ tag `v0.15.1`。
> 起点：nano-vllm 此前**组件拆分但同进程**——LLMEngine/AsyncLLM 直接持 EngineCore，
> generate 循环同进程驱动 `EngineCore.step()`；进程隔离只到 Worker 层（K 轮）。这是
> 全局对比标注的"最根本区别：进程拓扑"。本轮（P）补 EngineCore 独立进程。

## V1 进程拓扑

```
[前端进程] LLMEngine/AsyncLLM + Processor(tokenize) + OutputProcessor(detokenize)
   │  ZMQ (msgpack)  ← v1/engine/core_client.py
   ▼
[EngineCore 进程] EngineCoreProc.run_busy_loop()：Scheduler + Executor
   │  collective_rpc
   ▼
[Worker 进程 × N] rank0..rankN-1
```

目的：让 GPU 调度循环不被 tokenize / detokenize / HTTP 阻塞，前端与内核独立伸缩。

## V1 关键组件

| 组件 | 文件 | 职责 |
|---|---|---|
| `EngineCoreClient`(ABC) | `v1/engine/core_client.py` | 前端句柄：`get_output` / `add_request` / `abort_requests` + async 变体 |
| `InprocClient` | 同上 | EngineCore 同进程，`get_output()` 即 `step()`（offline/调试） |
| `MPClient` / `SyncMPClient` / `AsyncMPClient` | 同上 | EngineCore 独进程，ZMQ 收发；后台输出线程把 output socket 排进队列，`get_output` 读队列 |
| `EngineCoreProc` | `v1/engine/core.py` | EngineCore 子类：input/output 两后台线程做 socket↔queue，`run_busy_loop` 跑核心循环 |
| `run_engine_core` | 同上 | 子进程入口：握手 → 建 EngineCoreProc → busy loop |

## V1 busy loop 要点

- 两个后台线程（process_input_sockets / process_output_sockets）把 ZMQ IO 与 GPU 重叠
  （socket IO 释放 GIL）。
- busy loop：从 input_queue 取请求（空闲阻塞）→ add/abort → step → 产出投 output_queue。
- 前端不再驱动 step；通过 `get_output()` 异步收产出。请求状态（is_finished）由前端
  OutputProcessor 跟踪。
- 启动握手：DEALER/ROUTER socket 交换地址 + READY，确保两端就绪。

## 与 nano 的差距（本轮范围）

| V1 特性 | 是否对齐 | 说明 |
|---|---|---|
| EngineCoreClient 抽象 + Inproc/MP 双实现 | ✅ | `engine/core_client.py` |
| EngineCore 独立子进程 + busy-loop | ✅ | `EngineCoreProc.busy_loop` + `run_engine_core_proc` |
| 前端只 add_request / 收 outputs，不驱动 step | ✅ | `get_output()` / `get_output_async()` |
| has_unfinished 前端本地跟踪 | ✅ | add 计入、finished/abort 移除 |
| 传输 | ⚠️ | **mp.Queue 替代 ZMQ**（与 Worker 层用 SharedMemory 替代 ZMQ 一致） |
| 后台 IO 线程重叠 socket/GPU | ❌ | mp.Queue 自带 feeder 线程；busy-loop 直接收发 |
| 启动握手（DEALER/ROUTER） | ⚠️ | 简化为 READY 哨兵（核心 init 完成后投递） |
| DP coordinator / 多 EngineCore / 弹性 | ❌ | 单核心，超出精简范围 |
