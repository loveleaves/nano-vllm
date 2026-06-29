# nano-vllm 引擎层对齐 V1 — 详细设计文档

> 基于 `research.md`。目标：把单进程"一把梭"的 `LLMEngine` 拆成 V1 风格的
> **Processor / EngineCore / OutputProcessor** 三段，并新增 **AsyncLLM** 异步流式入口。
> 范围按用户决策：**组件拆分 + 异步通路，保持同进程、保留同步 LLMEngine**。

## Motivation

对齐前 `LLMEngine.step()` 在一个进程里串起：tokenize → schedule → worker.call →
postprocess → detokenize，职责高度耦合：
- 输入（tokenize）/ 输出（detokenize、停止串、finish reason）逻辑散落在 engine 与
  `generate()` 里，无法独立测试，也没有 streaming；
- 调度执行循环没有独立边界，难以替换/复用（例如未来上异步或独立进程）。

V1 的解法是把这三段拆成显式组件，并用 `EngineCore*` 结构体固化它们之间的数据契约。

## 范围决策（与 V1 的取舍）

| V1 特性 | 本次是否对齐 | 说明 |
|---|---|---|
| 组件拆分（Processor/EngineCore/OutputProcessor） | ✅ | 本次核心 |
| EngineCore* 数据契约结构体 | ✅ | 用 dataclass（同进程，无需 msgspec.Struct） |
| 增量 detokenize / 停止串 / finish reason | ✅ | OutputProcessor + IncrementalDetokenizer |
| AsyncLLM 异步流式（async generator） | ✅ | 后台 handler + 每请求 RequestOutputCollector |
| **EngineCore 独立进程 + ZMQ client** | ❌ | 教学项目保持同进程；EngineCore 仍是自洽类，边界清晰，未来可平移到独立进程 |
| 完整采样 / 多模态 / spec decode / structured output | ❌ | 引擎层之外，不在本轮 |

> 同进程下 detokenize O(n²) 全量解码（见 detokenizer.py 注释）换取实现简洁，
> 且保证最终文本严格等于旧版 `tokenizer.decode(tokens)`。

## Architecture

### 包结构（新增/改写）

```
nanovllm/engine/
├── core_types.py       # 新增：FinishReason / EngineCoreRequest / EngineCoreOutput(s) / RequestOutput
├── processor.py        # 新增：Processor —— tokenize → EngineCoreRequest
├── core.py             # 新增：EngineCore —— 持有 Scheduler+Worker，调度执行循环
├── detokenizer.py      # 新增：IncrementalDetokenizer + check_stop_strings
├── output_processor.py # 新增：OutputProcessor / RequestState / OutputProcessorOutput
├── async_llm.py        # 新增：AsyncLLM + RequestOutputCollector
├── llm_engine.py       # 改写：瘦身为三组件装配的 facade
├── scheduler.py        # 微改：新增 abort(seq)（停止串/外部中止）
├── worker.py           # 不变
├── model_runner.py     # 不变
├── sequence.py         # 不变（EngineCore 动态挂 seq.request_id 属性，不入 __getstate__）
└── rpc.py              # 不变
```

### 数据流（同步，单步）

```
LLM.generate(prompts, sps)
  └─ for each: Processor.process_inputs ─► EngineCoreRequest
                 ├─ EngineCore.add_request   (建 Sequence, scheduler.add)
                 └─ OutputProcessor.add_request (建 RequestState + detokenizer)
  └─ while not finished:
       EngineCore.step() ──► EngineCoreOutputs(outputs=[EngineCoreOutput,...], num_tokens)
         (schedule → worker.call("run") → scheduler.postprocess → 收集每请求新 token)
       OutputProcessor.process_outputs(outputs) ──► [RequestOutput], reqs_to_abort
         (增量 detokenize + 停止串截断 + finish reason)
       EngineCore.abort_requests(reqs_to_abort)   # 停止串命中者释放 KV
```

### 数据流（异步流式，AsyncLLM）

```
async for ro in AsyncLLM.generate(prompt, sp):   # 逐 step yield 增量 RequestOutput
   ...
  内部：add_request → _pending + RequestOutputCollector(queue)
        后台 _run_output_handler 协程：
          drain _pending → EngineCore.add_request   (在协程内，与 step 错开，无并发写)
          await loop.run_in_executor(None, EngineCore.step)  (GPU 期间释放 GIL)
          OutputProcessor.process_outputs → 各 collector.put(ro)
```

### 关键设计点

- **request_id**：Processor 分配（自增字符串）；EngineCore 把它挂到 `seq.request_id`，
  用于回填 EngineCoreOutput。该属性不进 `Sequence.__getstate__`，故不随 RPC 跨进程传输
  （rank0 持有原对象，子进程无需）。
- **"是否产出 token"判定**：EngineCore 用本步前后 `num_completion_tokens` 之差判断——
  chunked prefill 未覆盖完整 prompt 的步不产 token（与旧 `is_prefill` 跳过一致）。
- **finish_reason**：`num_completion_tokens >= max_tokens` → LENGTH，否则（EOS）→ STOP；
  停止串命中由 OutputProcessor 判定为 STOP 并回报 abort。
- **向后兼容**：`LLMEngine.generate()` 返回结构不变（`[{"text","token_ids"}]`，按输入序），
  `example.py` / `bench.py` 无需改动。
