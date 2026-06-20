# 调度器对齐 — 测试设计

全部 CPU 纯逻辑测试（不触 GPU）。运行：
`pytest tests/test_scheduler.py tests/test_request_queue.py`

## 测试矩阵

| 文件 | 覆盖点 |
|---|---|
| `test_scheduler.py` | schedule() 返回 `SchedulerOutput`（类型/total/is_empty/preempted）；FCFS 基础调度、prefill→decode、EOS/max_tokens 终止、ignore_eos、跨 seq 预算、max_num_seqs、显存满返回空、完整生命周期、并发、释放块、**abort 释放**；chunked prefill 两轮/仅首 seq 分块/留 waiting；抢占（含 `preempted_seq_ids`）/回插队首/死锁保护；连续批 prefill+decode 混排；**Priority 策略按值出队** |
| `test_request_queue.py` | 工厂类型；FCFS 顺序/prepend/contains/remove；Priority 最小值优先/同级按到达/contains·remove·iter |

## 适配要点（API 迁移）

- `seqs, ns = sched.schedule()` → `out = sched.schedule()`，读 `out.scheduled_seqs` /
  `out.num_scheduled_tokens`（测试用 `_sched()` 辅助一次性解包）。
- `sched.postprocess(seqs, toks, ns)` → `sched.update_from_output(out, toks)`。
- `sched.add` 别名保留；`sched.waiting` 现为 `RequestQueue`，断言改用
  `waiting.peek_request()` / `seq in waiting` / `len(waiting)`（FCFS 仍是 deque，索引可用）。

## 回归

- 全量套件：**191 passed, 4 skipped**（skip 为需权重的 loader 用例），较对齐前 +9 用例。
- EngineCore（test_engine_core）经 `update_from_output` / `SchedulerOutput` 改造后行为不变，全绿。
- `Config.scheduling_policy` 默认 `"fcfs"`，端到端行为与对齐前一致；`example.py`/`bench.py` 不改。
