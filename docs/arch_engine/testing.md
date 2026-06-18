# 引擎层对齐 — 测试设计

全部新增测试均为 **CPU 纯逻辑**（不触 GPU/NCCL），通过注入假 Worker / 假 EngineCore /
假 tokenizer 隔离 GPU 依赖。运行：`pytest tests/test_processor.py tests/test_output_processor.py
tests/test_engine_core.py tests/test_async_llm.py`。

## 测试矩阵

| 文件 | 覆盖点 |
|---|---|
| `test_processor.py` | str 编码 / list[int] 透传 / request_id 自增 / 显式 request_id / 空 prompt 拒绝 |
| `test_output_processor.py` | 增量 detokenize 的 delta 与累计一致；停止串择最早命中；停止串截断+abort+STOP；core 侧 finish_reason 透传；结束后状态清理；未知 request_id 忽略 |
| `test_engine_core.py` | prefill+decode 至 LENGTH；EOS → STOP；chunked prefill 未完成步不产 token；abort 释放请求；空调度返回空 outputs |
| `test_async_llm.py` | generate() 逐步 yield 增量；增量累计 == 最终文本；末条 finished/finish_reason 正确 |

## 隔离手段

- **EngineCore**：`object.__new__` 绕过 `__init__`（避免拉起 Worker/ModelRunner），
  注入真实 `Scheduler`（纯 Python）+ `FakeWorker.call()`（返回固定 token）。
- **AsyncLLM**：`object.__new__` + `FakeEngineCore`（脚本化 `step()` 序列），
  用 `asyncio.run` 驱动，无需 pytest-asyncio。
- **tokenizer**：`FakeTokenizer.encode/decode = ord/chr`，确定性、可读。

## 回归

- 既有 `test_scheduler / test_sequence / test_rpc` 全绿（Scheduler 仅新增 `abort`，
  Sequence/Worker/ModelRunner/rpc 未改）。
- 全量套件：**182 passed, 4 skipped**（skip 为需模型权重的 loader 用例），较对齐前 +18 用例。
- `LLMEngine.generate()` 返回结构不变，`example.py` / `bench.py` 无需改动。
