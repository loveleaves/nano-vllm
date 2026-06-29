# InputBatch 增量更新对齐 — 测试设计

## 测试矩阵

| 文件 | 覆盖点 | 依赖 |
|---|---|---|
| `test_block_table.py` | CpuGpuBuffer numpy 视图/copy_to_gpu；add_row/append_row/move_row/num_blocks_per_row；compute_slot_mapping（单/多请求）；commit/clear | CPU（device="cpu"） |
| `test_input_batch.py` | add_request 致密行；remove_request + condense 行回收（含块表搬移）；update 新增/逐出/decode 仅追加新块；make_inputs 行序排序 + cu_seqlens/slot_mapping/无缓存 warmup 路径 + decode-only last_token；批集合≠调度集合断言 | CPU |
| `test_rpc.py`（改） | encode/decode 三元组（method, seqs, finished）；finished_seq_ids 往返 | CPU |
| `test_sequence.py`（改） | `__getstate__/__setstate__` 还原 seq_id | CPU |
| `test_engine_core.py`（改） | `FakeExecutor.execute_model(seqs, finished_seq_ids=None)` 新签名 | CPU |
| `test_qwen3.py` | 端到端：LLM→EngineCore→Executor→Worker→ModelRunner.InputBatch，实跑 GPU 推理（prefill+decode+CUDA graph 全链） | GPU |

## 关键不变量验证

- **批集合 == 调度集合**：`make_inputs` 内 `assert num_reqs == len(scheduled_seqs)`，
  `test_make_inputs_asserts_batch_equals_scheduled` 显式触发。
- **slot_mapping 数值**：block_size=4、块表 [10,11] → slots [40..45]，CPU 上精确比对。
- **decode 增量**：prefill 1 块 → decode 跨块后 `num_blocks_per_row` 仅 +1、块表 [10,11]。
- **行回收**：移除中间行 + condense 后末尾行滑入空洞，块表数据随之搬移，req_id_to_index 更新。

## 回归

- 全量套件：**225 passed, 4 skipped**（+21：test_block_table 8 + test_input_batch 12 +
  test_rpc 1；test_sequence/engine_core 为既有用例改签名）。
- GPU `test_qwen3` 绿 + `example.py` 两 prompt 生成连贯 → 增量 InputBatch 路径与对齐前
  数值等价；prefill/decode/CUDA graph/前缀缓存全链无回归。

## 限制

- 行序==调度序的简化依赖"连续批每步调度整个活跃集合"。若未来引入"running 子集调度"
  （如严格 token 预算下部分 running 不解码），`make_inputs` 的断言会显式报错而非静默错算，
  届时需改为按行序 gather 调度子集。
- TP>1（MultiProcExecutor）的 finished_seq_ids 广播逻辑随既有多进程链路，单卡未真机验证。
