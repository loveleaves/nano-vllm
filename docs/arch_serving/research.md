# 服务入口（OpenAI API server）对齐 — V1 现状调研

> 对照基准：本机 `/home/cb/work/vllm/vllm` @ tag `v0.15.1`。
> 起点：nano-vllm 仅有 `LLM.generate()`（同步）与 `AsyncLLM.generate()`（异步生成器），
> **无 HTTP 服务入口**——是全局对比文档标注的"广度差距"之一。本轮（O）补 OpenAI 兼容 server。

## 背景：什么是服务入口（OpenAI 兼容 API server）

**问题**：`LLM.generate()` 是**库**调用——同进程、阻塞、单机本地。要把推理引擎当**服务**对外提供
（多客户端、HTTP、流式），需要一层 HTTP 入口把网络请求翻译成引擎调用。

**核心思想——OpenAI 兼容 + 三件套分层**：复刻 OpenAI 的 REST 协议（`/v1/completions`、
`/v1/chat/completions`、`/v1/models`），现有 OpenAI 客户端/生态可零改动接入。每个端点分三层：
- **protocol**：Pydantic 请求/响应模型（OpenAI schema）+ 采样参数映射到 SamplingParams。
- **serving**：把请求翻译成对 `AsyncLLM.generate` 的调用，再把流式产出组装成响应。
- **api_server**：FastAPI app + 路由 + lifespan（持引擎）+ uvicorn 启动。

**流式（SSE）**：chat/completions 支持 `stream=true`，用 Server-Sent Events 逐块下发
（`data: {json}\n\n` … `data: [DONE]`），客户端实时拿到增量文本（delta）。

**作用 / 收益**：让 nano 从"库"变成可被任意 OpenAI 客户端调用的"服务"；复用同进程 AsyncLLM 的连续
批与流式（每个 HTTP 请求 = 一次 AsyncLLM.generate，后台 handler 把它们混排进同一批）。

## vLLM 服务入口结构

```
vllm/entrypoints/
├── api_server.py / launcher.py            # 通用启动
├── openai/
│   ├── api_server.py                      # FastAPI app + lifespan(engine_client) + 挂路由 + run_server(uvicorn)
│   ├── cli_args.py                        # argparse → 引擎/服务配置
│   ├── completion/{protocol,serving,api_router}.py
│   ├── chat_completion/{protocol,serving,api_router}.py
│   ├── models/{protocol,serving,api_router}.py
│   └── ...（responses/tool_parsers/reasoning_parsers/run_batch）
└── grpc/、anthropic/、sagemaker/、mcp/      # 其它协议入口
```

每个端点组三件套：
- **protocol.py**：Pydantic 请求/响应模型（OpenAI schema）。
- **serving.py**：`OpenAIServing*` 处理器，把请求翻译为 `engine.generate()` 调用，
  再组装响应；流式产 SSE。
- **api_router.py**：FastAPI 路由，依赖注入取 `app.state.openai_serving_*`。

## 关键机制

- **app 装配**：`api_server.py` 用 lifespan 异步上下文建/收 engine_client（AsyncLLM/MQ client），
  存进 `app.state`；路由从 state 取处理器。
- **流式 SSE**：`StreamingResponse(generator, media_type="text/event-stream")`；
  generator 逐块 `yield f"data: {json}\n\n"`，收尾 `yield "data: [DONE]\n\n"`。
- **非流式**：消费生成器到末值，聚合成 JSON，`JSONResponse(model_dump())`。
- **错误**：`ErrorResponse` → `JSONResponse(content=..., status_code=...)`。
- **CLI**：`cli_args.make_arg_parser()` + `run_server()` 经 uvicorn 启动。

## 与 nano 的差距（本轮范围）

| V1 特性 | 是否对齐 | 说明 |
|---|---|---|
| FastAPI app + lifespan 持引擎 | ✅ | `build_app(engine, model_name)` |
| /v1/completions（流 + 非流） | ✅ | SSE + JSON |
| /v1/chat/completions（流 + 非流） | ✅ | apply_chat_template 渲染 prompt |
| /v1/models、/health | ✅ | 单模型卡 |
| Pydantic OpenAI 协议 + 采样参数映射 | ✅ | `_SamplingMixin.to_sampling_params` |
| CLI + uvicorn 启动 | ✅ | `python -m ...api_server --model` |
| completion logprobs | ⚠️ | 仅非流式，基础结构（tokens/token_logprobs/top_logprobs/text_offset） |
| n>1 / best_of / echo / suffix | ❌ | nano 单序列/请求 |
| tools / function_call / 结构化输出 | ❌ | 依赖 structured_output（未对齐） |
| chat logprobs / multimodal / gRPC / Anthropic | ❌ | 超出精简范围 |
| MQ/ZMQ engine_client（独立 EngineCore 进程） | ❌ | nano 引擎同进程，直接持 AsyncLLM |
