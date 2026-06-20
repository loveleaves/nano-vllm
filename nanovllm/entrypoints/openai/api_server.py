"""
OpenAI 兼容 API 服务入口（对齐 vLLM `entrypoints/openai/api_server.py`）。

FastAPI app + lifespan（持 AsyncLLM 引擎）+ 路由（/health、/v1/models、/v1/completions、
/v1/chat/completions）。流式走 SSE（text/event-stream），非流式返回 JSON。

启动：
  python -m nanovllm.entrypoints.openai.api_server --model /path/to/Qwen3-1.7B
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from nanovllm.engine.async_llm import AsyncLLM
from nanovllm.entrypoints.openai.cli_args import engine_kwargs_from_args, make_arg_parser
from nanovllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    CompletionRequest,
    ErrorResponse,
    ModelCard,
    ModelList,
)
from nanovllm.entrypoints.openai.serving import (
    OpenAIServingChat,
    OpenAIServingCompletion,
)


def build_app(engine: AsyncLLM, served_model_name: str) -> FastAPI:

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # 引擎已在外部建好，此处仅在关闭时回收
        yield
        engine.exit()

    app = FastAPI(title="nano-vllm OpenAI API", lifespan=lifespan)
    app.state.engine = engine
    app.state.model_name = served_model_name
    app.state.serving_completion = OpenAIServingCompletion(engine, served_model_name)
    app.state.serving_chat = OpenAIServingChat(engine, served_model_name)

    @app.get("/health")
    async def health():
        return JSONResponse({"status": "ok"})

    @app.get("/v1/models")
    async def show_models():
        return ModelList(data=[ModelCard(id=served_model_name)]).model_dump()

    @app.post("/v1/completions")
    async def create_completion(request: CompletionRequest, raw_request: Request):
        handler: OpenAIServingCompletion = raw_request.app.state.serving_completion
        result = await handler.create_completion(request)
        return _to_response(result)

    @app.post("/v1/chat/completions")
    async def create_chat_completion(request: ChatCompletionRequest, raw_request: Request):
        handler: OpenAIServingChat = raw_request.app.state.serving_chat
        result = await handler.create_chat_completion(request)
        return _to_response(result)

    return app


def _to_response(result):
    """统一把处理器返回值映射为 HTTP 响应：错误→JSON(含状态码)、流→SSE、对象→JSON。"""
    if isinstance(result, ErrorResponse):
        return JSONResponse(content=result.model_dump(), status_code=result.error.code)
    # async generator（流式）：鸭子类型判断
    if hasattr(result, "__aiter__"):
        return StreamingResponse(result, media_type="text/event-stream")
    return JSONResponse(content=result.model_dump())


def run_server(args):
    import uvicorn

    served_model_name = args.served_model_name or args.model
    engine = AsyncLLM(args.model, **engine_kwargs_from_args(args))
    app = build_app(engine, served_model_name)
    uvicorn.run(app, host=args.host, port=args.port)


def main():
    args = make_arg_parser().parse_args()
    run_server(args)


if __name__ == "__main__":
    main()
