# Worker/Executor 层对齐 — 测试设计

## 测试矩阵

| 文件 | 覆盖点 | 依赖 |
|---|---|---|
| `test_executor.py` | `Executor.get_class` 工厂分派：TP=1→UniProcExecutor，TP>1→MultiProcExecutor（轻量 stub 提供 tensor_parallel_size，不构造 GPU Worker） | CPU |
| `test_engine_core.py`（改） | 注入由 `FakeWorker` 改为 `FakeExecutor`（`execute_model`）；EngineCore step/abort/finish 逻辑不变全绿 | CPU |
| `test_rpc.py`（不变） | ShmTransport 编解码/收发，未改动 | CPU |
| `test_qwen3.py` | TP=1 端到端：LLM→EngineCore→UniProcExecutor→Worker→ModelRunner，实跑 GPU 推理 | GPU |

## 限制

- `UniProcExecutor` / `MultiProcExecutor` 的真实实例化需 GPU + NCCL，单测只覆盖 `get_class`
  分派与 EngineCore 对 Executor 接口的使用（fake 注入）。
- **TP>1（MultiProcExecutor）需多卡，未真机验证**——与对齐前同一限制。进程编排顺序逐位
  保留原实现，理论等价。

## 回归

- 全量套件：**204 passed, 4 skipped**（+2：test_executor），较 KV cache 轮 +2 用例。
- GPU `test_qwen3` 绿 → TP=1 执行器链端到端等价；`example.py`/`bench.py` 不改。
