"""服务化示例：启动 OpenAI 兼容 API 服务。

在 LLM.generate（同步，example.py）/ AsyncLLM 流式（example_async.py）之外，本脚本
把 AsyncLLM 包成 HTTP 服务（FastAPI + uvicorn），暴露与 OpenAI 兼容的端点：
  GET  /health
  GET  /v1/models
  POST /v1/completions          （流式 SSE + 非流式）
  POST /v1/chat/completions     （流式 SSE + 非流式，自动套 chat 模板）

启动：
  python example_server.py
  # 等价于 CLI：python -m nanovllm.entrypoints.openai.api_server --model ~/model/Qwen3-1.7B

另开一个终端，用 curl 或 OpenAI 客户端访问：

  # 非流式 completion
  curl http://localhost:8000/v1/completions -H 'Content-Type: application/json' -d '{
        "model": "qwen3", "prompt": "Hello, my name is", "max_tokens": 32, "temperature": 0.6}'

  # 流式 chat（SSE）
  curl -N http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
        "model": "qwen3",
        "messages": [{"role": "user", "content": "introduce yourself"}],
        "max_tokens": 128, "stream": true}'

  # 也可直接用官方 openai python 客户端（pip install openai）：
  #   from openai import OpenAI
  #   client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
  #   print(client.chat.completions.create(
  #       model="qwen3",
  #       messages=[{"role": "user", "content": "introduce yourself"}]).choices[0].message.content)

依赖：pip install -e ".[serve]"   # fastapi / uvicorn / pydantic
"""
import os

import uvicorn

from nanovllm import AsyncLLM
from nanovllm.entrypoints.openai.api_server import build_app


def main():
    path = os.path.expanduser("~/model/Qwen3-1.7B/")
    # AsyncLLM 承载连续批 + 流式；服务把每个 HTTP 请求映射为一次 AsyncLLM.generate
    engine = AsyncLLM(path, enforce_eager=True, tensor_parallel_size=1)
    app = build_app(engine, served_model_name="qwen3")

    # uvicorn 关闭时触发 lifespan，回收引擎（engine.exit）
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
