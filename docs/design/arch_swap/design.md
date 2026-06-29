# swap 抢占对齐 V1 — 详细设计

> 基于 `research.md`。新增 swap 抢占路径，门控于 `Config.num_swap_blocks > 0`，仅 UniProc 支持。

## 数据流

```
EngineCore.step()
  ├─ sched_output = scheduler.schedule()
  │     phase 0  换回：swapped 队列 FIFO，can_swap_in → block_manager.swap_in(seq)
  │               → swap_in_list += [(gpu_block_id, swap_slot)]，seq 回 running
  │     phase 1  decode：can_append 失败 → preempt(victim, swap_out_list)
  │               preempt: swap_enabled & can_swap_out → block_manager.swap_out(seq)
  │               → swap_out_list += [(gpu_block_id, swap_slot)]，seq 入 swapped
  │               否则 recompute（deallocate + 回 waiting）
  │     SchedulerOutput.blocks_to_swap_in/out = swap_in_list / swap_out_list
  ├─ executor.execute_swap(blocks_to_swap_in, blocks_to_swap_out)   # 在 execute_model 之前
  │     UniProc → worker.model_runner.execute_swap：先 swap_out(D2H) 再 swap_in(H2D)
  │     MultiProc → 有搬运则 raise NotImplementedError
  └─ executor.execute_model(seqs, finished_seq_ids)
```

## 各层改动

### Config（config.py）
- `num_swap_blocks: int = 0` — CPU swap 区块数；>0 开启 swap，仅 TP=1 内联支持。

### SchedulerOutput（sched/output.py）
- `blocks_to_swap_out: list[tuple[int, int]]` / `blocks_to_swap_in: list[tuple[int, int]]`
  ——元素 `(gpu_block_id, swap_slot)`，按逻辑块序。

### KVCacheManager（kv_cache/kv_cache_manager.py）
新增 swap 区状态与原语：
- `free_swap_slots: deque[int]`（初始 `range(num_swap_blocks)`），`swapped_slots: dict[seq_id, list[slot]]`。
- `can_swap_out(seq)` = `len(free_swap_slots) >= seq.num_blocks`。
- `swap_out(seq)` → 分配 slots、释放 GPU 块（逆序 deref）、清空 block_table（**保留**
  num_cached_tokens）、返回 `[(gpu_block_id, swap_slot)]`。
- `can_swap_in(seq)` = `block_pool.get_num_free_blocks() >= seq.num_blocks`。
- `swap_in(seq)` → 重分配 GPU 块、归还 slots、恢复 block_table、返回 `[(gpu_block_id, swap_slot)]`。

### Scheduler（sched/scheduler.py）
- `__init__(... num_swap_blocks=0)`：`swap_enabled`、`self.swapped: deque[Sequence]`。
- `schedule()` phase 0：换回循环（FIFO，受 `max_num_seqs` 与 `can_swap_in` 约束）。
- `preempt(seq, swap_out_list=None)`：swap_enabled 且 can_swap_out → swap_out 入 swapped；
  否则 recompute。
- `get_num_unfinished_requests` / `is_finished` 计入 swapped。
- `abort`：swapped 中的 seq 归还 swap 槽位（无 GPU 块可释放）。

### ModelRunner（engine/model_runner.py）
- `allocate_kv_cache`：`num_swap_blocks>0` 时分配 `cpu_kv_cache`（CPU pinned，形状
  `[2, L, num_swap_blocks, block_size, num_kv_heads, head_dim]`）。
- `swap_out(blocks)`：gather GPU 块 → `.to("cpu")` 落地 → 写 cpu_kv_cache 对应槽（同步拷贝）。
- `swap_in(blocks)`：gather cpu 槽 → `.to("cuda")` → 写 kv_cache 对应块。
- `execute_swap(in, out)`：先 swap_out（读旧 KV）再 swap_in。
- `exit()`：清 `cpu_kv_cache`。

> 注意：拷贝用**两步**（先 `.to(device)` 落地成中间张量，再 index 赋值），避免高级索引
> gather 的异步副本与 scatter 写入竞争。swap 仅在抢占时发生，同步开销可忽略。

### Executor
- 抽象基类 `execute_swap`：有搬运则 `raise NotImplementedError`（默认不支持）。
- `UniProcExecutor.execute_swap`：转发 `worker.model_runner.execute_swap`。
- `MultiProcExecutor`：继承基类默认 → 抛错（RPC 载荷未扩展）。

### EngineCore（engine/core.py）
- Scheduler 构造传 `num_swap_blocks=config.num_swap_blocks`。
- `step()`：`blocks_to_swap_in/out` 非空时，在 execute_model **之前** `executor.execute_swap`。

## 正确性论证

- **换出读旧 KV**：execute_swap 先于 execute_model，swap_out 读到的是被抢占序列未被
  本步覆写的 KV。
- **块不相交**：swap_in（phase 0）分配的块 vs swap_out（phase 1）释放的块，前者先于后者，
  同一步不交叉；execute_swap 内 swap_out 先于 swap_in，读/写块也不交叉。
- **InputBatch 一致**：换出序列的 seq_id 随 preempted_seq_ids → finished_seq_ids 下发，
  行槽位被回收；换回时 `req_id_to_index` 无此 seq → `add_request` 以完整 block_table 新建行，
  `make_inputs` 用保留的 num_cached_tokens 起算 position，续算 decode。
- **等价性**：swap 与 recompute 对 greedy 输出**逐 token 一致**（KV 字节级还原），见 testing.md。
