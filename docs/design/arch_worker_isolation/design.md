# nano-vllm Worker/Executor 进程隔离对齐 V1 — 详细设计

> 基于 `research.md`。目标：补齐 H 轮遗留的"rank0 也进子进程"——让 `MultiProcExecutor`
> 把**所有 rank（含 rank0）放进隔离子进程**，executor 进程不持 Worker、不入 NCCL 组，
> 只经共享内存的**广播 + 回传**两条通道与各 worker 通信。

## 范围决策（与 V1 的取舍）

| V1 特性 | 是否对齐 | 说明 |
|---|---|---|
| 所有 rank（含 rank0）为隔离子进程 | ✅ | executor 纯协调进程，不内联模型、不入 NCCL |
| 广播通道 executor→workers | ✅ | 复用 ShmTransport（创建方改为 executor） |
| 回传通道 output_rank→executor | ✅ | 新增 ResultChannel（rank0 唯一采样者） |
| KV 块数经 RPC 回传引擎 | ✅ | `collective_rpc("num_kvcache_blocks")` |
| `distributed_executor_backend` 选择后端 | ✅ | None→按 TP 自动；显式 "mp" 让 TP=1 也隔离（单卡可测） |
| MessageQueue 多槽 / FailureCallback / 健康监控 / PP·DP | ❌ | nano 单槽 shm + 单 output_rank |

## 通道与进程布局

```
                ┌──────────── executor 进程（纯 CPU 协调，无 Worker/NCCL）─────────┐
                │  ShmTransport(create=True, events=[e0..e_{N-1}])  广播 1→N        │
                │  ResultChannel(create=True, result_event)         回传 N→1(仅r0) │
                └───────┬───────────────────────────────────────────▲─────────────┘
            broadcast    │ set e_i / 写 shm                  rank0 send│ 写 result_shm
                         ▼                                            │
      ┌──────────────────────────────  worker 子进程 rank r  ─────────────────────────┐
      │ Worker(config, r)  → ModelRunner: NCCL init + warmup + allocate + cudagraph    │
      │ ShmTransport(r, e_r, create=False)；rank0 另开 ResultChannel(create=False)      │
      │ loop: recv(method,seqs,finished) → worker.execute(...) → (rank0) result.send()  │
      └───────────────────────────────────────────────────────────────────────────────┘
```

## 启动 / 退出编排

```
启动 _init_executor:
  executor 先建两条通道（子进程一启动即可打开）
  → spawn rank0..N-1（各自 Worker.__init__：NCCL init→warmup→allocate_kv_cache→cudagraph）
  → collective_rpc("num_kvcache_blocks")：广播 → 阻塞等 rank0 回传 → 填 self.config.num_kvcache_blocks
     （EngineCore 随后据此建 Scheduler；引擎侧 config 与 executor.config 同一对象，赋值即可见）

单步 execute_model:
  collective_rpc("run", seqs, finished) → broadcast → result.recv()（仅 rank0 的 token_ids）

退出 shutdown:
  broadcast("exit") → 各 rank 跳出循环 → worker.execute("exit")(destroy_process_group，集体)
  → join → executor close+unlink 两通道
```

## 关键设计点

- **executor 不再加入 NCCL 组**：rank0 移入子进程后，N 个 worker 在自己的进程组里完成
  `init_process_group(world_size=N)` 与 all_reduce；executor 全程不 import CUDA，职责更纯。
- **KV 块数显式回传**：块数算在 rank0 worker 进程，引擎进程必须 RPC 拉取（对齐前 rank0
  内联可直接读 config）。`Worker.execute("num_kvcache_blocks")` 返回 `model_runner.config.num_kvcache_blocks`。
- **collective_rpc 返回 `[output_rank 结果]`**：与 UniProcExecutor 的列表契约一致，
  `execute_model` 统一取 `[0]`。
- **单槽 shm 的时序安全**：广播/回传各用单槽 SharedMemory（非 V1 的多槽 MessageQueue）。
  安全依赖：① `execute_model` 同步——broadcast 后阻塞等 rank0 回传，故广播被串行化；
  ② "run" 的前向内含 NCCL all_reduce，rank0 回传结果时 rank>0 必已读过本次广播（集体通信
  迫使所有 rank 进入 execute）。故下一次广播不会在 rank>0 读取前覆盖 shm。无 NCCL 的
  "num_kvcache_blocks"/"exit" 因后续操作间隔足够大同样安全。**与 H 同级的多卡未真机验证限制仍在。**
- **默认行为不变**：`distributed_executor_backend=None` 且 TP=1 → UniProcExecutor 内联，
  example.py / bench.py 零改动、无新进程开销。隔离仅在 TP>1（自动）或显式 `"mp"` 时启用。

## 隔离暴露的跨层耦合（验证中发现并修复）

把 rank0 移入子进程后，**rank0 也吃反序列化后的 Sequence**（对齐前 rank0 内联、持真对象）。
两处连带修正：

1. **采样标量必须序列化**：`prepare_sample` 在 rank0 子进程里读 `seq.temperature/top_p/top_k/
   *_penalty/logprobs`，故这些标量加入 `Sequence.__getstate__/__setstate__`（J 轮曾假设
   "rank0 持真对象、采样配置不入 getstate"，本轮被隔离推翻）。
2. **spawn 启动约束**：用户入口须置于 `if __name__ == "__main__":`（Python spawn 通用要求，
   非本轮新增——对齐前的 TP>1 多进程亦然）。

### 边界（隔离模式不支持惩罚采样）

decode 序列化只传 `last_token`（不传完整 token 历史以省带宽），而 presence/frequency/
repetition 惩罚需要 prompt∪output 的完整 token。故**进程隔离（mp）下惩罚类采样不可用**——
需惩罚时用默认内联 UniProc（temperature/top-k/top-p/greedy/logprobs 在隔离模式均正常）。

### 已知限制（健壮性）

若某 worker 子进程在 `__init__` 阶段异常退出，executor 会在 `result.recv()` 上**永久阻塞**
（无 V1 的 FailureCallback / 健康监控）。验证期间即由此现象定位到上述序列化缺陷。生产化需
补进程存活探测，本轮未做。

## 改动清单

```
config.py                         # + distributed_executor_backend: str|None（None/"uni"/"mp"）
engine/rpc.py                     # + ResultChannel（回传）；ShmTransport 创建方语义改为 executor
engine/worker.py                  # + execute("num_kvcache_blocks")
engine/executor/abstract.py       # get_class 按 backend 分派（None 时按 TP）
engine/executor/multiproc_executor.py  # 全 rank 子进程 + 双通道 + 块数回传（重写）
engine/executor/uniproc_executor.py    # 不变（TP=1 默认内联）
```
