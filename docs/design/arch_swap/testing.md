# swap 抢占对齐 V1 — 测试

## CPU 单元测试（tests/test_swap.py，8 例，`-m unit`）

### KVCacheManager swap 原语（TestKVCacheManagerSwap）
- `test_swap_out_allocates_slots_and_preserves_cached_tokens` — swap_out 占用槽位、释放
  GPU 块、清空 block_table，但**保留** num_cached_tokens。
- `test_swap_in_restores_block_table_and_returns_slots` — swap_in 归还同批槽位、重分配 GPU 块。
- `test_can_swap_out_in_thresholds` — 槽位/空闲块阈值判定。

### Scheduler swap 决策（TestSchedulerSwap）
- `test_preempt_routes_to_swap_when_enabled` — swap_enabled 时抢占进 swapped 而非 waiting，
  收集 swap_out 映射，保留缓存。
- `test_swap_in_resume_on_next_schedule` — 下一步 schedule() phase 0 换回，seq 回 running
  并立即被调度 decode，blocks_to_swap_in 非空。
- `test_falls_back_to_recompute_when_swap_full` — swap 区满 → 回退 recompute（回 waiting 队首，
  block_table 清空，无 swap 映射）。
- `test_unfinished_count_includes_swapped` — swapped 计入未完成计数 / is_finished。
- `test_abort_swapped_returns_slots` — abort 换出序列归还 swap 槽位。

全套：`267 passed`（含本轮 swap/metrics/sampler 新增），无回归。

## GPU 验证（scripts/gpu_validate_swap.py，需显卡 + Qwen3-1.7B）

1. **张量往返**：填已知值 → `swap_out`(D2H) → 抹零 GPU 块 → `swap_in`(H2D) → `torch.equal`
   校验逐元素还原 → **PASS**。
2. **端到端等价**：monkeypatch `allocate_kv_cache` 钳制 `num_kvcache_blocks=6`，3 条并发
   prompt（必触发抢占），对比 `num_swap_blocks=0`（recompute）与 `=32`（swap）的 greedy 输出
   → 3/3 prompt **逐 token 一致** → **PASS**。

```
=== 1) swap_out/swap_in 张量往返 ===  还原一致: True  PASS
=== 2) swap vs recompute greedy 等价（强制抢占）===
   prompt[0..2] len=64 一致=True  PASS
ALL GPU SWAP VALIDATIONS PASSED
```

## 局限
- MultiProc / TP>1 的 swap 未实现（execute_swap 抛 NotImplementedError）；GPU 验证仅 UniProc。
- swap 换回不复用前缀缓存块（按整条 block_table 私有还原），功能正确但内存非最优。
