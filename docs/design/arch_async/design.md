# 异步调度对齐 V1 — 详细设计

> 基于 `research.md`。门控 `Config.async_scheduling`，仅 UniProc，与 swap/mp 互斥。

## 流水（深度 1，单进程 CUDA 异步）

`EngineCore._step_async()` 每次调用的顺序（关键）：

```
1. sched = scheduler.schedule()                 # CPU：本步调度（用上一步推进后的长度）
2. if not empty: executor.execute_model_async(sched.scheduled_seqs, sched.finished)
     └─ ModelRunner：make_inputs → 用 inflight 采样张量前向回填 decode 输入 → forward
        → sample → 采样张量存 pending（**不** .tolist()，不同步）
     ← 立即返回（GPU 异步执行；步 1~2 的 CPU 工作与上一步 GPU 计算重叠）
3. if inflight: tok_by_id, lp_by_id = executor.resolve_inflight()   # D2H 同步取回上一步
     produced = scheduler.resolve_output(inflight, token_ids)       # 回填占位 + EOS 判定
     outputs = build(produced)
4. if launched: scheduler.advance_after_schedule(sched)   # 推进记账（在 resolve 之后！）
                executor.promote_async()                  # pending → inflight（下一步前向源）
                self._inflight = sched
   else:        scheduler.finished_req_ids |= sched.finished_seq_ids   # 退回未投递的回收 id
                self._inflight = None
```

**为何 resolve（步3）在 advance（步4）之前**：advance 的 `hash_blocks` 需读已喂入 token；
若先 advance 下一步占位再 resolve，会让下一步 advance 的哈希读到未回填占位。先 resolve 回填、
再 advance，且已结束序列被 advance 跳过。

## 各层改动

### Config
- `async_scheduling: bool = False`；`__post_init__` 断言 TP==1、backend∈{None,uni}、num_swap_blocks==0。

### Sequence（占位 token）
- `num_pending`（尾部未回填占位数，同步恒 0）。
- `append_placeholder()`（追加值 0、num_pending+1）/ `resolve_placeholder(tok)`（覆写最早占位）
  / `truncate_pending()`（丢弃全部占位，用于抢占/结束清理）。

### SchedulerOutput
- `produced_token_seq_ids: set[int]` — 本步产出 token 的 seq（decode + prefill 收尾）。

### Scheduler
- `schedule()` 收集 `produced_seq_ids`。
- `advance_after_schedule(output)`：镜像 update_from_output 的长度推进，但**追加占位**而非真实
  token、不做 EOS 判定；跳过已结束序列。
- `resolve_output(output, token_ids) -> list[Sequence]`：仅回填 `status==RUNNING 且 num_pending>0`
  的序列（跳过 partial prefill / 已结束多调度一步 / 被抢占中止），EOS/max 判定 + 结束清理，
  返回产出序列。
- `preempt`：异步下先 `truncate_pending()` 再 deallocate，使 recompute 从最后已回填 token 干净重算。

### ModelRunner（采样 token 留 GPU + 跨步前向）
- 两槽 `_ai`（inflight：上一步采样张量 + ordered + index{seq_id→row} + logprobs）/ `_ap`（pending）。
- `execute_model_async(seqs, finished)`：make_inputs 后，对"喂生成 token"的 decode 行
  （`q==1 且 num_cached>=num_prompt`）用 `input_ids[flat_pos] = _ai.sampled[prev_row]` 在 GPU 就地
  回填（无 D2H）；forward+sample → 采样张量存 `_ap`（不 tolist）。
- `resolve_inflight()`：`_ai.sampled.tolist()` → (tok_by_id, lp_by_id)。
- `promote_async()`：`_ai = _ap`。

### Executor
- 抽象基类 `execute_model_async/resolve_inflight/promote_async` 默认 `raise NotImplementedError`。
- `UniProcExecutor` 转发 `worker.model_runner.*`。

### EngineCore
- `async_scheduling` + `_inflight`；`step()` 按标志分派 `_step_sync`/`_step_async`。
- `has_unfinished_requests` 计入 `_inflight is not None`（排空在飞步）。

## 正确性论证

- **逐 token 一致（单序列）**：前向喂给每步的 token 与同步完全相同（同一采样张量），无批伴随
  → 无 FP 扰动 → greedy 输出逐 token 等于同步（GPU 验证证实）。
- **多序列**：结束序列被多调度一步（其 batch 成员多一员），可能令同批其他序列在 FP 平局处翻转
  （与 chunked prefill 同性质，非 bug）；实测 batch-8 graph 模式仍与同步逐 token 一致。
- **抢占协同**：被抢占序列 truncate 占位 + 干净重算；其在飞结果在 resolve_output 按"非运行态"丢弃。
- **跨 generate 行回收**：空调度排空步不下发 model，故把未投递的 finished_seq_ids 退回累积器，
  留待下一次 launch（含下个 generate）回收 InputBatch 行（否则陈旧行触发 make_inputs 断言）。
