# 调度器对齐 — V1 现状调研

> 对照基准：本机 `/home/cb/work/vllm/vllm` @ tag `v0.15.1`（V1 架构）。

## 背景：调度器做什么

**职责**：每一步推理前，调度器从"等待队列 + 运行队列"里挑出本步要喂给 GPU 的序列与各自的
token 数，受三重预算约束：最大并发序列数、单步最大 token 数、KV 显存（块）是否够分配。它是
吞吐与延迟的总开关。

**核心思想——统一连续批（continuous batching）**：不再分"先 prefill 一批、再 decode 一批"的阶段，
而是每步把 decode（每序列 1 个新 token）与 prefill（新请求的 prompt，可分块）**混排进同一个变长
（varlen）批**。一个序列生成完即时让出名额、新请求即时补入——GPU 利用率持续打满，而非等整批
对齐。decode 显存不足时按策略**抢占**（换出/重算）。

**作用 / 收益**：高吞吐、低排队延迟、平稳的显存占用；可插拔排队策略（FCFS / 优先级）。

**子包结构（本轮对齐 V1）**：`interface`(ABC) + `output`(结构化 SchedulerOutput) +
`request_queue`(FCFS/Priority 队列) + `scheduler`(算法主体：decode 优先 → prefill 填剩余预算 →
预算不足则抢占)。

## V1 调度器子包（`vllm/v1/core/sched/`）

| 文件 | 职责 | nano 对应 |
|---|---|---|
| `interface.py` | `SchedulerInterface(ABC)`：schedule / update_from_output / add_request / finish_requests / has_unfinished_requests / get_grammar_bitmask / update_draft_token_ids / make_stats / shutdown ... | `interface.py`（最小子集） |
| `output.py` | `SchedulerOutput`（+ `NewRequestData` / `CachedRequestData`）：结构化调度结果，含 new/cached 请求拆分、spec/encoder/kv-connector 元数据 | `output.py`（核心字段子集，不拆 new/cached） |
| `request_queue.py` | `RequestQueue(ABC)` + `FCFSRequestQueue` + `PriorityRequestQueue` + `SchedulingPolicy` + `create_request_queue` | `request_queue.py`（同名同构） |
| `scheduler.py` | `Scheduler(SchedulerInterface)`：连续批 + chunked prefill + 前缀缓存 + 抢占(recompute/swap) + spec/structured/encoder/kv-connector 调度 | `scheduler.py`（去掉 nano 没有的功能项） |
| `async_scheduler.py` | 异步调度（提前出下一步） | ❌ 不在本轮（nano EngineCore 同步） |
| `utils.py` | 调度辅助 | 内联 |

## 关键观察

1. **调度决策结构化**：V1 `schedule()` 返回 `SchedulerOutput` 而非裸元组，把"调度了什么"
   固化成可传输、可断言的对象——这是对齐的核心。nano 原先返回 `(seqs, num_scheduled dict)`，
   升级为 `SchedulerOutput`。
2. **排队策略可插拔**：waiting 队列经 `RequestQueue` 抽象，FCFS / PRIORITY 由 `create_request_queue`
   工厂选择，与调度算法解耦。Priority 按 `(priority, arrival)` 出队（值小先行）。
3. **接口与实现分离**：`SchedulerInterface` 让 EngineCore 只依赖契约。
4. **schedule / update_from_output 两段式**：schedule 决策，执行器跑完后 update_from_output
   回写状态——nano 原 `postprocess` 即此语义，本轮按 V1 命名对齐。
5. **不强塞 nano 没有的功能**：spec decode / structured output / encoder cache / kv connector /
   swap 抢占 / stats / grammar bitmask 均不纳入（无对应功能，纳入只是空壳）。
