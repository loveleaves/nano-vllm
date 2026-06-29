# nano-vllm EngineCore 进程化（进程拓扑）对齐 V1 — 详细设计

> 基于 `research.md`。目标：把"组件拆分但同进程"升级为 V1 的三级进程拓扑——
> 前端（tokenize/detokenize/HTTP）与 EngineCore（Scheduler+Executor）分进程，
> EngineCore 子进程自跑 busy-loop，前端经队列收发。**用 stdlib multiprocessing.Queue
> 替代 ZMQ**（与 Worker 层用 SharedMemory 替代 ZMQ 的取舍一致）。

## 范围决策（与 V1 的取舍）

| V1 特性 | 是否对齐 | 说明 |
|---|---|---|
| EngineCoreClient 抽象 + Inproc/MP 双实现 + 工厂 | ✅ | `make_client(config)` 按 `multiproc_engine_core` 选 |
| EngineCore 独立子进程 busy-loop | ✅ | spawn + `run_engine_core_proc` |
| 前端只 add/收 outputs，has_unfinished 本地跟踪 | ✅ | — |
| 传输 ZMQ | ❌→mp.Queue | pickle 友好的 EngineCoreRequest/Outputs 直接过队列 |
| 后台 socket IO 线程、DEALER/ROUTER 握手 | ❌→简化 | mp.Queue 自带 feeder；用 READY 哨兵 |
| DP coordinator / 多核心 / 弹性扩缩 | ❌ | 单核心 |

## Architecture

```
engine/core_client.py
├── EngineCoreClient(ABC)        # get_output / get_output_async / add_request / abort_requests
│   └── make_client(config)      #   → InprocClient | MPClient（按 config.multiproc_engine_core）
├── InprocClient                 # 持 EngineCore；get_output()=step()（默认，行为零变化）
├── EngineCoreProc               # busy_loop(core,in_q,out_q) + _handle_input（传输无关，可进程内单测）
├── run_engine_core_proc(...)    # 子进程入口：建核心→READY→busy_loop→退出回收
└── MPClient                     # spawn 子进程 + mp.Queue 收发 + 本地 has_unfinished 跟踪

engine/llm_engine.py   self.engine_core = EngineCoreClient.make_client(config); step()→get_output()
engine/async_llm.py    同上；handler 用 get_output_async()
```

### 进程拓扑（MPClient 路径）

```
[前端进程] LLMEngine/AsyncLLM + Processor + OutputProcessor + MPClient
   │  input_queue  (ADD/ABORT/EXIT, pickle)
   │  output_queue (READY/OUTPUTS/ERROR, pickle)
   ▼
[EngineCore 进程] run_engine_core_proc → EngineCoreProc.busy_loop：Scheduler + Executor
   │  (UniProc 内联 rank0 / MultiProc：ShmTransport + ResultChannel)
   ▼
[Worker 进程 × N]
```

### 消息协议

| 方向 | 消息 | 载荷 |
|---|---|---|
| 前端→核心 | `(ADD, req)` / `(ABORT, ids)` / `(EXIT, None)` | EngineCoreRequest / list[str] / — |
| 核心→前端 | `(READY, None)` | 核心 init 完成 |
| 核心→前端 | `(OUTPUTS, (outs, stats))` | EngineCoreOutputs + SchedulerStats |
| 核心→前端 | `(ERROR, traceback)` | init 或循环异常（避免前端永久阻塞） |

### busy-loop 算法

```
while True:
  if not core.has_unfinished_requests():
      msg = input_queue.get()          # 空闲阻塞，不空转
      if msg==EXIT: return
      handle(msg)
  drain input_queue (非阻塞)：ADD/ABORT，EXIT→return
  if core.has_unfinished_requests():
      outputs = core.step()
      if outputs.outputs:               # 空步（chunked prefill 中途/空调度）不投递，前端续等
          output_queue.put((OUTPUTS, (outputs, core.get_stats())))
```

### 关键设计点

- **零回归**：默认 `multiproc_engine_core=False` → InprocClient，`get_output()` 即旧
  `step()`，逐字节等价；现有 examples/bench/tests 全不变。
- **has_unfinished 本地跟踪**：MPClient `add_request` 把 request_id 计入 `_unfinished`，
  `get_output` 见 finished 输出 / `abort_requests` 时移除——无需向子进程往返查询。
  停止串结束走前端 OutputProcessor → `abort_requests` 路径，正确清账。
- **tokenize 留前端**：跨进程只传 EngineCoreRequest（token ids）/ EngineCoreOutputs
  （token ids + finished + logprobs），纯 int/dataclass，pickle 友好；Config（含
  PretrainedConfig）spawn 下可 pickle（已验证）。
- **前端不碰 CUDA**：NCCL/建模/warmup 全在 EngineCore 子进程，前端保持 CUDA-free。
- **错误传播**：init 或 busy-loop 任何异常 → `(ERROR, traceback)`，前端 `get_output`/
  启动握手抛 RuntimeError，不在队列上死等。
- **生命周期**：`MPClient.exit()` 投 EXIT → busy-loop 退出 → `core.exit()` 回收
  Executor/Worker → join(timeout) 兜底 terminate。

### 配置

`Config.multiproc_engine_core: bool = False`。与既有 `distributed_executor_backend`
（Worker 层 uni/mp）正交：可组合 `multiproc_engine_core=True` + TP>1 mp（核心子进程再
spawn worker 子进程，嵌套 spawn）。spawn 需调用方 `if __name__=="__main__":` 守卫。
