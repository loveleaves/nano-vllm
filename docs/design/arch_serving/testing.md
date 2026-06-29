# 服务入口（OpenAI API server）— 测试

> 单测：`tests/test_openai_server.py`（11 例，`-m unit`，免 GPU）。
> 用 FastAPI `TestClient` + 假引擎 `FakeAsyncLLM`（含假 tokenizer + 两步 generate 生成器），
> 不起真模型即可验证 HTTP 层 + 协议层 + 流式装配。

## 运行

```bash
source .venv/bin/activate
pytest tests/test_openai_server.py -m unit -v
# 全量回归
pytest -m unit -q          # 298 passed
```

依赖：`pip install fastapi "uvicorn[standard]" pydantic httpx`（或 `pip install -e '.[serve,dev]'`）。

## 覆盖矩阵

| 用例 | 验证点 |
|---|---|
| `test_health` | /health → 200 ok |
| `test_models` | /v1/models 列出单模型卡 |
| `test_completion_non_stream` | text 聚合 / finish_reason=stop / usage 计数 |
| `test_completion_stream_sse` | SSE 分块格式 + `[DONE]` + 增量拼接==全文 |
| `test_completion_logprobs` | tokens/token_logprobs/text_offset 正确（text_offset 按 decode 长度累加） |
| `test_completion_multi_prompt` | list[str] → 多 choices，index 对齐 |
| `test_completion_stream_multi_prompt_rejected` | 流式 + 多 prompt → 400 |
| `test_completion_n_not_one_rejected` | n=2 → 400 |
| `test_chat_non_stream` | message.role/content + **apply_chat_template 渲染喂入引擎** |
| `test_chat_stream_sse` | 首块 delta.role=assistant + 末块 finish_reason + `[DONE]` |
| `test_to_sampling_params_mapping` | OpenAI 参数 → SamplingParams 全字段映射 |

## 结果

- `tests/test_openai_server.py`：11 passed
- 全量 `-m unit`：**298 passed**，4 deselected，无回归。

## 手动 GPU 联调（需真模型）

```bash
python -m nanovllm.entrypoints.openai.api_server --model ~/model/Qwen3-1.7B --enforce-eager
# 另开终端
curl http://localhost:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3","prompt":"Hello,","max_tokens":16}'
curl http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3","messages":[{"role":"user","content":"hi"}],"stream":true}'
```
