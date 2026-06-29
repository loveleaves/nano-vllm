# nano-vllm 服务入口（OpenAI API server）对齐 V1 — 详细设计

> 基于 `research.md`。目标：在既有 `AsyncLLM`（async generator + delta_text）之上，
> 复刻 V1 的 OpenAI 兼容 HTTP 入口——FastAPI app + protocol/serving 分层 + SSE 流式 +
> CLI/uvicorn 启动。复用 nano 现有引擎栈，**不引入独立 EngineCore 进程 / ZMQ**。

## 范围决策（与 V1 的取舍）

| V1 特性 | 是否对齐 | 说明 |
|---|---|---|
| protocol / serving / api_router 分层 | ✅ | 合并为 protocol.py + serving.py + api_server.py（nano 规模） |
| /v1/completions、/v1/chat/completions（流 + 非流） | ✅ | SSE + JSON 聚合 |
| /v1/models、/health | ✅ | — |
| Pydantic v2 + 采样参数映射 | ✅ | `_SamplingMixin` 共用 |
| CLI + uvicorn | ✅ | `python -m nanovllm.entrypoints.openai.api_server` |
| completion logprobs | ⚠️ | 仅非流式、基础版 |
| n>1/echo/suffix/best_of、tools、chat logprobs、multimodal、gRPC | ❌ | 单序列/精简定位 |
| MQ engine_client、独立进程 | ❌ | 直接持同进程 AsyncLLM |

## Architecture

```
entrypoints/openai/
├── protocol.py     # Pydantic：UsageInfo/ErrorResponse/ModelCard
│                   # _SamplingMixin(to_sampling_params) → CompletionRequest/ChatCompletionRequest
│                   # *Response / *StreamResponse / CompletionLogprobs
├── serving.py      # OpenAIServing(base) + OpenAIServingCompletion + OpenAIServingChat
├── api_server.py   # build_app(engine, name) + lifespan + 4 路由 + _to_response + run_server/main
└── cli_args.py     # make_arg_parser + engine_kwargs_from_args
```

### 请求处理流程

```
HTTP POST /v1/completions
  → FastAPI 解析 CompletionRequest(Pydantic 校验)
  → app.state.serving_completion.create_completion(req)
      ├── n!=1 → ErrorResponse
      ├── 归一化 prompt（str/list[int]→1；list[str]/list[list[int]]→N）
      ├── req.to_sampling_params() → SamplingParams
      ├── stream:  返回 async generator（SSE：每步 data:{json}\n\n，收尾 [DONE]）
      └── 非流式: asyncio.gather 各 prompt 跑到末值 → choices + usage → CompletionResponse
  → _to_response()：ErrorResponse→JSON(状态码) / __aiter__→StreamingResponse / 对象→JSON
```

Chat 路径同构，差别在：先 `tokenizer.apply_chat_template(messages, add_generation_prompt)`
渲染成 prompt 字符串再喂 `engine.generate`；流式首块发 `delta.role="assistant"`。

### 关键设计点

- **复用 AsyncLLM**：serving 只依赖 `engine.generate(prompt, sp, request_id)`（async gen）+
  `engine.tokenizer` + `engine.exit()`；不碰 Scheduler/Worker。
- **流式增量**：直接用 `RequestOutput.delta_text`（OutputProcessor 已算好本步新增文本）。
- **非流式聚合**：消费生成器到 `finished`，取末值的 cumulative `text`/`token_ids`/`finish_reason`。
- **logprobs**：从 `RequestOutput.logprobs`（list[dict[int,float]]，对齐 token_ids）+ tokenizer
  decode 构造 `tokens/token_logprobs/top_logprobs/text_offset`；仅非流式 completion。
- **lifespan**：引擎在 `run_server` 外部建好传入 `build_app`，关闭时 `engine.exit()` 回收显存。
- **错误码**：`ErrorResponse.error.code` 即 HTTP 状态码（默认 400）。

### 采样参数映射（OpenAI → SamplingParams）

| OpenAI | SamplingParams | 备注 |
|---|---|---|
| max_tokens | max_tokens | None→默认 64 |
| temperature/top_p/top_k/min_p | 同名 | 0 温度=greedy |
| presence/frequency/repetition_penalty | 同名 | — |
| seed | seed | 持久 generator 续流 |
| stop | stop | str→[str] |
| logprobs(int) | logprobs | 仅 completion 透传 |
| n | —（必须 1） | 否则 ErrorResponse |

## 依赖

新增 `serve` 可选组：`fastapi` / `uvicorn` / `pydantic`；`dev` 加 `httpx`（TestClient 所需）。
