# nano-vllm 调度器对齐 V1 — 详细设计文档

> 基于 `research.md`。目标：把单文件 `engine/scheduler.py` 升级为 V1 风格的
> `engine/sched/` 子包——**SchedulerInterface 抽象 + 结构化 SchedulerOutput +
> 可插拔 RequestQueue（FCFS/Priority）**，并把 `postprocess` 对齐为 `update_from_output`。

## Motivation

对齐前调度器：单文件、`schedule()` 返回裸元组 `(seqs, num_scheduled dict)`、排队写死
FCFS deque、回写方法叫 `postprocess`。与 V1 的差距集中在"形"：缺接口抽象、缺结构化输出、
排队策略不可插拔、命名不统一。本轮补齐这些，调度算法本身（连续批/chunked/前缀缓存/抢占）
保持不变。

## 范围决策（与 V1 的取舍）

| V1 特性 | 是否对齐 | 说明 |
|---|---|---|
| `engine/sched/` 子包结构 | ✅ | 镜像 `v1/core/sched/`（沿用 attention 拆包先例） |
| `SchedulerInterface` ABC | ✅ | 最小子集 |
| 结构化 `SchedulerOutput` | ✅ | 替换裸元组 |
| `RequestQueue`：FCFS + Priority | ✅ | `create_request_queue` 工厂；Config 加 `scheduling_policy` |
| `postprocess` → `update_from_output` | ✅ | 命名对齐 |
| NewRequestData/CachedRequestData 拆分 | ❌ | nano 执行器直接吃 Sequence 对象，无需拆 |
| async_scheduler / swap 抢占 / spec / structured / encoder / kv-connector / stats | ❌ | nano 无对应功能 |

## Architecture

### 包结构

```
nanovllm/engine/sched/
├── __init__.py        # 导出 Scheduler / SchedulerInterface / SchedulerOutput / SchedulingPolicy / RequestQueue / create_request_queue
├── interface.py       # SchedulerInterface(ABC)
├── output.py          # SchedulerOutput（scheduled_seqs / num_scheduled_tokens / total / preempted_seq_ids）
├── request_queue.py   # RequestQueue(ABC) + FCFS + Priority + SchedulingPolicy + create_request_queue
└── scheduler.py       # Scheduler(SchedulerInterface)：算法主体（从旧 scheduler.py 迁入）
nanovllm/engine/scheduler.py   # 向后兼容垫片：re-export Scheduler/SchedulerOutput/SchedulingPolicy
```

### 数据流（单步）

```
EngineCore.step():
  out = scheduler.schedule()                 ─► SchedulerOutput
        ├─ 1) RUNNING decode：每 seq 1 token；块不足 → preempt 末尾（记入 out.preempted_seq_ids）
        └─ 2) WAITING prefill chunk：waiting.peek_request()（策略决定顺序），按预算切块
  token_ids = worker.call("run", out.scheduled_seqs)
  scheduler.update_from_output(out, token_ids)   # 原 postprocess：hash_blocks/追加 token/终止
```

### 关键设计点

- **SchedulerOutput**：`scheduled_seqs` + `num_scheduled_tokens{seq_id→n}` +
  `total_num_scheduled_tokens` + `preempted_seq_ids`。EngineCore 据此算吞吐
  （任一 n>1 视为含 prefill）。
- **RequestQueue 真值陷阱**：容器协议（`__contains__/__len__/__iter__/bool`）**不**声明为
  abstractmethod——否则抽象占位会在 MRO 中遮蔽 `FCFSRequestQueue`(deque) 基于 `__len__`
  的隐式真值，使 `bool(q)` 返回 None。故 ABC 只抽象 add/pop/peek/prepend/remove。
- **Priority 语义**：`(seq.priority, seq.seq_id)` 最小堆，值越小越先、同级按到达（seq_id）。
  `priority` 经 SamplingParams 之外的独立通道（Sequence.priority / EngineCoreRequest.priority /
  Processor.process_inputs(priority=) / LLMEngine.add_request(priority=)）透传，默认 0。
- **preempt** 用 `waiting.prepend_request`（FCFS 回插队首；Priority 按优先级归位）。
- **向后兼容**：保留 `engine/scheduler.py` 垫片与 `Scheduler.add`（= `add_request`）别名，
  旧 import / 调用不破。`Config.scheduling_policy` 默认 `"fcfs"`，行为与对齐前一致。
