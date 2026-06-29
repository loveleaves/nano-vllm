# 异步调度对齐 V1 — 测试

## CPU 单元测试（tests/test_async_scheduling.py，9 例，`-m unit`）

### Sequence 占位 token（TestSequencePlaceholder）
- `test_append_and_resolve` — append_placeholder/resolve_placeholder 正确推进与回填。
- `test_truncate_pending` — 丢弃占位、恢复 last_token。

### EngineCore 异步 vs 同步**等价**（TestAsyncEquivalence）
用假执行器（确定 token + 两槽模拟"采样张量留 GPU"）驱动同步与异步两条流水到完成，逐请求
比对产出 token 流与结束原因：
- `test_length_finish_single` — 达 max_tokens → LENGTH。
- `test_eos_finish_single` — 命中 EOS → STOP（首 token 即结束）。
- `test_multiple_requests` — 3 请求不同 max_tokens 并发，各自 token 流一致。
- `test_chunked_prefill` — 长 prompt 小预算多个 partial prefill chunk（不产 token）再 decode。

### 流水机制（TestAsyncMechanics）
- `test_inflight_drains_after_schedule_empty` — 首步只下发无回收（无产出）；在飞步经后续步排空。

全套：`274 passed`（含本轮 async 9 例），无回归。

## GPU 验证（scripts/gpu_validate_async.py，需显卡 + Qwen3-1.7B）

1. **单序列 async vs sync 逐 token 一致**：无批伴随 → 无多调度步 FP 扰动 → 64 token 完全相同 → PASS。
2. **多序列 async 确定性 + 连贯**：两次 async 结果一致，输出文本连贯 → PASS。
3. **吞吐对比**：

| 模式 | sync | async | 备注 |
|---|---|---|---|
| enforce_eager | 7.8s | 8.2s（0.94x） | eager 已含 CPU 开销，前向回填开销难被掩盖 |
| CUDA graph（batch-8, 128 tok） | 1.67s | 1.61s（**1.03x**）**输出逐 token == sync** | graph 模式 CPU 调度被掩盖，略增益 |

> 单卡小模型重叠收益有限，async 的价值随 CPU-overhead-bound 工作负载（大 batch / graph 模式 /
> 短 decode）放大。重在**正确（逐 token 等价）+ 不退化**。graph 模式 batch-8 含抢占/多调度仍与
> 同步逐 token 一致，验证了占位/前向/抢占协同/跨 generate 行回收的正确性。

## 局限
- 仅 UniProc / TP=1；与 swap 抢占、mp 进程隔离互斥（Config 断言）。
- 惩罚类采样在 async 下读到的 output token 历史含未回填占位（0），略不精确（penalties 罕用）。
- 流水深度固定为 1（无 PP 多步占位）。
