# InputBatch 增量更新对齐 — V1 现状调研

> 对照基准：本机 `/home/cb/work/vllm/vllm` @ tag `v0.15.1`（V1 架构）。

## V1 组件

| 文件 | 职责 | nano 对应 |
|---|---|---|
| `v1/worker/gpu_input_batch.py::InputBatch` | 跨步常驻的批状态：`req_id → row` 持久槽位、`_req_ids` 行表、各类持久 CPU/GPU 缓冲（token/position/采样参数…）、`add_request`/`remove_request`/`condense`（行回收） | `engine/input_batch.py::InputBatch`（最小子集） |
| `v1/worker/block_table.py::BlockTable` | 常驻 (max_num_reqs × max_blocks) 块表 + slot_mapping 缓冲；`add_row`/`append_row`/`move_row`/`compute_slot_mapping`/`commit_*` | `engine/block_table.py::BlockTable` |
| `v1/utils.py::CpuGpuBuffer` | 一对 CPU(pinned)+GPU 张量 + numpy 视图，增量写后整体异步上传 | `engine/block_table.py::CpuGpuBuffer` |
| `v1/worker/gpu_model_runner.py` | 每步据 `scheduler_output.{scheduled,finished_req_ids}` 更新 InputBatch，按行序展开输入、前向、采样，输出按 req_id 对齐 | `engine/model_runner.py`（重构 run） |

## 关键观察

1. **持久行槽位**：请求首次调度时占一个行号（`req_id_to_index`），其块表行、采样参数等
   常驻；结束/抢占后行变空，`condense` 把末尾活跃行滑入空洞回收槽位——避免每步重建批。
2. **块表增量**：decode 每步通常只新增 ≤1 个 KV 块，`append_row` 仅写新块；只有 prefill /
   行重写才整行 `add_row`。`commit_block_table(n)` 只异步上传前 n 行。对齐前 nano 每步都重建
   一张 padded 块表并整表 H2D，是主要浪费点。
3. **slot_mapping 向量化**：按 (req_indices, positions) 两个展开数组一次算出，写入常驻缓冲。
4. **finished_req_ids 经调度输出下发**：上一步结束的请求在下一步调度输出里带回 runner，
   runner 据此回收其行（V1 EngineCore 把 finished 串进 SchedulerOutput）。
5. **输出按 req_id 对齐**：模型按 InputBatch 行序前向/采样，得到行序 logits/token，再由
   `req_id_to_index` 映射回逻辑请求顺序。

## nano 取舍（不引入）

LoRA / spec-decode / 多模态(encoder)/ prompt-embeds / pooling / 多 KV 组（MultiGroupBlockTable）
/ hybrid kernel_block 拆分 / CP·DCP。nano 单组同构 full-attention，只取"持久行 + 增量块表 +
行回收 + 输出按 req_id 对齐"这条主干。
