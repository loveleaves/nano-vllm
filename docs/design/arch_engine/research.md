# 引擎层对齐 — V1 现状调研

> 对照基准：本机 `/home/cb/work/vllm/vllm` @ tag `v0.15.1`（V1 架构）。

## 背景：为什么把引擎拆成多个组件

**问题**：单体 LLMEngine 把"输入 tokenize、调度执行、输出 detokenize、停止判定、流式"全揉在一起，
难测试、难复用、难做异步。

**核心思想——按职责分层（对齐 V1）**，每段是可独立测试的组件，靠显式数据契约（dataclass）相连：
```
Processor        输入处理：文本 → token → EngineCoreRequest
EngineCore       调度 + 执行循环：持 Scheduler + Executor，产 EngineCoreOutputs（只含 token，不含文本）
OutputProcessor  输出处理：增量 detokenize、停止串匹配、finish reason → RequestOutput（面向用户）
Detokenizer      增量解码（token → 文本片段）
AsyncLLM         异步流式入口（async generator，逐步 yield 增量）
```

**作用 / 收益**：各段独立演进/单测；EngineCore 不碰文本、Processor/OutputProcessor 不碰 GPU；
为异步流式（AsyncLLM）与后续 EngineCore 进程化（P）铺好边界。

## V1 引擎层组件（`vllm/v1/engine/`）

| 文件 | 职责 | nano 对应 |
|---|---|---|
| `__init__.py` | `EngineCoreRequest / EngineCoreOutput(s) / FinishReason` 等 msgspec 结构体 | `core_types.py`（dataclass） |
| `processor.py` + `input_processor` | 输入处理：tokenize、多模态、LoRA 解析 → EngineCoreRequest | `processor.py`（仅 tokenize） |
| `core.py` | EngineCore：Scheduler + Executor，跑在**独立进程** | `core.py`（同进程类） |
| `core_client.py` / `coordinator.py` | 前端 ↔ EngineCore 的 ZMQ 通信、多 core 协调（DP） | ❌ 不对齐（同进程） |
| `detokenizer.py` | IncrementalDetokenizer（前缀缓冲流式解码） | `detokenizer.py`（全量 decode 简化版） |
| `output_processor.py` | 增量 detokenize、停止串、logprobs、finish reason → RequestOutput | `output_processor.py`（无 logprobs） |
| `llm_engine.py` | 同步 facade，装配上述组件 | `llm_engine.py` |
| `async_llm.py` | 异步 facade，async generator + RequestOutputCollector | `async_llm.py` |
| `parallel_sampling.py` / `logprobs.py` | n>1 并行采样 / logprobs | ❌ 不在本轮 |

## 关键观察

1. **数据契约固化**：V1 用 `EngineCoreRequest/Output` 把"输入处理→调度执行→输出处理"
   三段解耦，每段只认结构体，互不依赖实现。nano 直接复用这一思想。
2. **进程隔离是性能手段而非抽象本身**：EngineCore 跑独立进程是为了不让 GPU 循环被
   tokenize/detokenize/HTTP 阻塞；其**类边界**（add_request/step/abort）在同进程下同样成立。
   故本轮先对齐类边界，进程隔离留作后续可平移项。
3. **OutputProcessor 承担全部文本逻辑**：EngineCore 只吐 token id，文本（detokenize、
   停止串截断、finish reason 字符串）全在 OutputProcessor，使 EngineCore 与 tokenizer 无关。
4. **AsyncLLM 复用同一套组件**：异步与同步入口共享 Processor/EngineCore/OutputProcessor，
   仅在外层加"后台 handler + 每请求队列"。
