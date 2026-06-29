# nano-vllm InputBatch 增量更新对齐 V1 — 详细设计

> 基于 `research.md`。目标：把 ModelRunner 每步"从零构造批输入 + 整表 H2D"重构为
> **跨步常驻的 InputBatch**——持久行槽位（seq_id→row）、块表增量追加、行回收（condense），
> 模型按行序前向/采样、输出按 req_id 对齐回调度顺序。

## 范围决策（与 V1 的取舍）

| V1 特性 | 是否对齐 | 说明 |
|---|---|---|
| `InputBatch` 持久行槽位 + `add/remove/condense` | ✅ | seq_id→row，行回收 |
| `BlockTable` 常驻块表 + `append_row`/`move_row`/`compute_slot_mapping` | ✅ | decode 仅追加新块 |
| `CpuGpuBuffer`（pinned + numpy 视图 + 单次异步 H2D） | ✅ | input_ids/positions/cu_seqlens/slot_mapping/block_table |
| `finished_req_ids` 经调度输出下发 runner | ✅ | `SchedulerOutput.finished_seq_ids`，随 RPC 广播到各 rank |
| 输出按 req_id 对齐 | ✅ | 行序 token → seq_id 映射回调度顺序 |
| LoRA / spec / 多模态 / prompt-embeds / 多 KV 组 / hybrid / CP·DCP | ❌ | nano 不具备 |

## 关键设计点

### 1) 行序即模型批序（nano 的简化，等价且更省）

V1 InputBatch 行号对请求**终身稳定**，模型按行序（0..num_reqs-1）前向，输出再按 req_id
映射回逻辑顺序。nano 沿用此结构，并利用一条不变量收紧实现：

> **连续批每步调度整个活跃集合**——故"逐出上一步结束/本步被抢占的行 + 补入新请求的行"
> 之后，**批集合恰等于本步调度集合**（`make_inputs` 内有断言保证）。

因此模型直接吃 `block_table.gpu[:num_reqs]`（行序），无需按调度子集 gather；`cu_seqlens`/
`slot_mapping`/`input_ids` 也按行序展开。采样得到行序 token 后，用 `req_id_to_index` 映射回
入参 `seqs` 顺序返回，使上层 `update_from_output` 仍可与 `scheduled_seqs` 直接 zip。

### 2) 块表增量

- 新请求 / 抢占后重入：`add_request` → `add_row(seq.block_table)` 整行写。
- decode / chunked-prefill 续算：`update` 比较 `len(seq.block_table)` 与 `num_blocks_per_row[row]`，
  只 `append_row` 本步新分配的块（绝大多数步 0~1 个 int）。
- `condense` 用 `move_row` 把末尾活跃行搬入低位空洞——稳定存活的请求保持原行号，
  其增量追加不受影响。

### 3) finished_seq_ids 通路

```
Scheduler.finished_req_ids（累积器）
  ├─ update_from_output：seq 结束 → add(seq_id)
  └─ abort：add(seq_id)
schedule()：drain 累积器 → SchedulerOutput.finished_seq_ids = 上步结束 | 本步被抢占
EngineCore.step：executor.execute_model(seqs, sched_output.finished_seq_ids)
  UniProc：worker.execute("run", seqs, finished)
  MultiProc：transport.broadcast("run", seqs, finished) → 各 rank ModelRunner 同步回收行
ModelRunner.run：input_batch.update(seqs, finished) → make_inputs(seqs) → 前向/采样 → 按 req_id 重排
```

被抢占的请求 KV 块已释放、将重新 prefill，故必须连同结束请求一起回收其旧行——
`finished_seq_ids = finished | preempted`。

### 4) 序列化补 seq_id

`Sequence.__getstate__/__setstate__` 增加 `seq_id`（首字段）：rank>0 子进程重建的 seq
需 seq_id 作为自身 InputBatch 的行索引键。decode 仍只传 last_token（不传完整 token_ids）。

### 5) warmup / CUDA graph 兼容

- InputBatch 在 `ModelRunner.__init__` 内、`warmup_model()` 之前建好（按 `max_num_seqs` ×
  `ceil(max_model_len/block_size)` 定容）。warmup 跑一遍最大批 prefill 后 `input_batch.clear()`，
  真正推理从空批开始。warmup 序列无 KV 块 → `has_cache=False` → slot=-1、block_table=None，
  与对齐前一致。
- `capture_cudagraph` 仍自建静态张量直调 `self.model(...)`，不经 InputBatch；replay 路径
  （`run_model`）从 `attn_md` 拷值到 graph vars 的逻辑不变。

## 包结构

```
nanovllm/engine/block_table.py   # CpuGpuBuffer + BlockTable（块表/ slot_mapping 常驻缓冲）
nanovllm/engine/input_batch.py   # InputBatch（行管理 add/remove/condense + update + make_inputs）
nanovllm/engine/model_runner.py  # run 重构：update → make_inputs → run_model → 按 req_id 重排
nanovllm/engine/sched/output.py  # SchedulerOutput += finished_seq_ids
nanovllm/engine/sched/scheduler.py  # finished_req_ids 累积 + schedule 下发
nanovllm/engine/{rpc,worker,executor/*}.py  # 透传 finished_seq_ids
nanovllm/engine/sequence.py      # __getstate__/__setstate__ += seq_id
```
