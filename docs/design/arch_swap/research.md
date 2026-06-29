# swap 抢占对齐 V1 — 调研

> 目标：把 nano 唯一的抢占策略（recompute：撤回 → 重算 prefill）补上 V1 的第二条路径
> ——**swap（KV 块换出到 CPU pinned 内存，恢复时换回 GPU、续算 decode）**，作为可选项
> （`num_swap_blocks > 0` 开启）。

## 背景：什么是抢占，swap vs recompute

**问题**：连续批运行中，正在 decode 的序列不断增长、要不断分配新 KV 块。当显存（块）耗尽、又有
新序列要进来或老序列要续算时，必须**抢占**——临时让出某些序列占用的 KV 块，等显存宽裕再恢复。

**两条恢复路径**：
- **recompute（重算）**：直接丢弃被抢占序列的 KV，恢复时把它当新请求**重新 prefill**。实现简单、
  零额外显存，但浪费已算过的前向（prompt 长时代价大）。nano 原有的唯一策略。
- **swap（换出）**：把被抢占序列的 KV 块从 GPU **拷到 CPU pinned 内存**（换出 D2H），恢复时再拷回
  GPU（换入 H2D）、从断点**续算 decode**。省去重算，但需 CPU 暂存区与 H2D/D2H 拷贝。

**核心思想**：用 CPU 内存当 GPU KV 的"交换区"（类比操作系统的 swap 分区），以带宽换算力。

**作用 / 收益**：长 prompt、高并发抢占频繁时，swap 比 recompute 省大量重复 prefill。
**边界**：本实现仅 UniProc，门控 `num_swap_blocks>0`（默认 0 → 仍走 recompute，零回归）。

## V1 的抢占设计

vLLM V1 `v1/core/sched/scheduler.py` 在 decode 步显存不足时抢占 running 序列。两种回收方式：

| 策略 | 动作 | 恢复 | 代价 |
|---|---|---|---|
| **recompute** | 释放 KV 块，序列回 waiting 重新 prefill | 重算整段 prompt+已生成 | 重复前向计算 |
| **swap** | KV 块 D2H 拷到 CPU swap 区，序列挂起 | H2D 拷回 GPU，续算 decode | PCIe 带宽，无重复计算 |

V1 通过 `SchedulerOutput.blocks_to_swap_in / blocks_to_swap_out`（`list[tuple[int,int]]`，
GPU 物理块 ↔ CPU swap 槽）把搬运决策下发执行器；GPU worker 在执行模型前完成块拷贝。
CPU swap 区大小由 `--swap-space`（GB）决定，换算成 CPU 块数。

## nano 对齐前现状

- 仅 recompute：`Scheduler.preempt()` = `deallocate` + 推回 waiting 队首。
- 无 CPU KV 镜像，无 swap 槽位管理。
- `SchedulerOutput` 无 swap 字段。

## 范围决策（与 V1 的取舍）

| V1 特性 | 是否对齐 | 说明 |
|---|---|---|
| recompute 抢占 | ✅（既有） | 默认路径，`num_swap_blocks=0` 时唯一路径 |
| swap 抢占（CPU pinned 镜像 + 块搬运） | ✅ | 本轮新增，`num_swap_blocks>0` 开启 |
| `blocks_to_swap_in/out` 下发执行器 | ✅ | SchedulerOutput 新增字段 |
| swap 与 prefix-cache 块共享的精细恢复 | ❌ | 换出按整条 block_table 私有处理，换回重分配（正确但不复用前缀块） |
| TP>1 / MultiProc 的 swap | ❌ | RPC 载荷未扩展；仅 UniProc（TP=1 内联）支持，MultiProc 抛 NotImplementedError |
| `--swap-space` GB 配置换算 | ❌ | 直接用 `num_swap_blocks`（块数）配置 |

## 决策依据

1. **默认零回归**：`num_swap_blocks` 默认 0 → 走原 recompute，现有路径与测试 100% 不变。
2. **换出时机**：`swap_out` 的 D2H 读取的是"被换出序列的旧 KV"，必须在 `execute_model`
   （会调 `store_kvcache` 覆写块）**之前**完成。故 EngineCore 在 execute_model 前先
   `execute_swap`。
3. **同步换入/换出不冲突**：schedule() 内 swap_in（phase 0，从空闲块分配）与 swap_out
   （phase 1 抢占，释放块）操作的 GPU 块不相交——swap_in 在 swap_out 释放之前已分配，
   故同一步无 ABA 冲突。
4. **保留 num_cached_tokens**：swap_out 清空 block_table 但**不动** num_cached_tokens，
   故换回后 `is_prefill` 仍为 False，直接续算 decode（区别于 recompute 的归零重算）。
