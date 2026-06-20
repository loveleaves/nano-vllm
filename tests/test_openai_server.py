"""
OpenAI 服务入口单测（FastAPI TestClient + 假引擎，免 GPU）。

运行：pytest tests/test_openai_server.py -m unit -v
"""
import json

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from nanovllm.engine.core_types import FinishReason, RequestOutput  # noqa: E402
from nanovllm.entrypoints.openai.api_server import build_app  # noqa: E402
from nanovllm.entrypoints.openai.protocol import CompletionRequest  # noqa: E402


# ─── 假 tokenizer / 假引擎 ───────────────────────────────────────────────────
class FakeTokenizer:
    eos_token_id = 0

    def decode(self, ids):
        return "".join(f"<{i}>" for i in ids)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        body = " ".join(f"{m['role']}:{m['content']}" for m in messages)
        return f"[CHAT]{body}{'[GEN]' if add_generation_prompt else ''}"


class FakeAsyncLLM:
    """模拟 AsyncLLM：generate 逐步 yield 两个增量块，记录收到的 prompt。"""

    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self.seen_prompts = []
        self.exited = False

    def exit(self):
        self.exited = True

    async def generate(self, prompt, sampling_params, request_id=None):
        self.seen_prompts.append(prompt)
        steps = [
            ("Hello", [10], {10: -0.1, 11: -2.0}, False, None),
            (" world", [12], {12: -0.2}, True, FinishReason.STOP),
        ]
        text = ""
        token_ids = []
        logprobs = []
        for delta, new_ids, lp, finished, reason in steps:
            text += delta
            token_ids = token_ids + new_ids
            logprobs = logprobs + [lp]
            yield RequestOutput(
                request_id=request_id or "x",
                prompt_token_ids=[1, 2, 3],
                token_ids=list(token_ids),
                text=text,
                delta_text=delta,
                finished=finished,
                finish_reason=reason,
                logprobs=list(logprobs),
            )


@pytest.fixture
def client():
    engine = FakeAsyncLLM()
    app = build_app(engine, "test-model")
    app.state._fake_engine = engine
    with TestClient(app) as c:
        c._engine = engine
        yield c


# ─── 基础端点 ────────────────────────────────────────────────────────────────
class TestBasicEndpoints:

    @pytest.mark.unit
    def test_health(self, client):
        r = client.get("/health")
        assert r.status_code == 200 and r.json()["status"] == "ok"

    @pytest.mark.unit
    def test_models(self, client):
        r = client.get("/v1/models")
        body = r.json()
        assert body["object"] == "list"
        assert body["data"][0]["id"] == "test-model"


# ─── /v1/completions ─────────────────────────────────────────────────────────
class TestCompletions:

    @pytest.mark.unit
    def test_completion_non_stream(self, client):
        r = client.post("/v1/completions", json={
            "model": "test-model", "prompt": "Hi", "max_tokens": 8})
        assert r.status_code == 200
        body = r.json()
        assert body["object"] == "text_completion"
        assert body["choices"][0]["text"] == "Hello world"
        assert body["choices"][0]["finish_reason"] == "stop"
        assert body["usage"]["completion_tokens"] == 2
        assert body["usage"]["prompt_tokens"] == 3

    @pytest.mark.unit
    def test_completion_stream_sse(self, client):
        r = client.post("/v1/completions", json={
            "model": "test-model", "prompt": "Hi", "stream": True})
        assert r.status_code == 200
        chunks = [l for l in r.text.split("\n\n") if l.strip()]
        assert chunks[-1] == "data: [DONE]"
        # 拼接增量文本
        texts = []
        for c in chunks[:-1]:
            payload = json.loads(c[len("data: "):])
            texts.append(payload["choices"][0]["text"])
        assert "".join(texts) == "Hello world"

    @pytest.mark.unit
    def test_completion_logprobs(self, client):
        r = client.post("/v1/completions", json={
            "model": "test-model", "prompt": "Hi", "logprobs": 2})
        lp = r.json()["choices"][0]["logprobs"]
        assert lp is not None
        assert lp["tokens"] == ["<10>", "<12>"]
        assert lp["token_logprobs"] == [-0.1, -0.2]
        assert lp["text_offset"] == [0, 4]   # len("<10>")==4

    @pytest.mark.unit
    def test_completion_multi_prompt(self, client):
        r = client.post("/v1/completions", json={
            "model": "test-model", "prompt": ["A", "B"]})
        choices = r.json()["choices"]
        assert len(choices) == 2 and choices[1]["index"] == 1

    @pytest.mark.unit
    def test_completion_stream_multi_prompt_rejected(self, client):
        r = client.post("/v1/completions", json={
            "model": "test-model", "prompt": ["A", "B"], "stream": True})
        assert r.status_code == 400

    @pytest.mark.unit
    def test_completion_n_not_one_rejected(self, client):
        r = client.post("/v1/completions", json={
            "model": "test-model", "prompt": "Hi", "n": 2})
        assert r.status_code == 400


# ─── /v1/chat/completions ────────────────────────────────────────────────────
class TestChat:

    @pytest.mark.unit
    def test_chat_non_stream(self, client):
        r = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}]})
        assert r.status_code == 200
        body = r.json()
        assert body["object"] == "chat.completion"
        msg = body["choices"][0]["message"]
        assert msg["role"] == "assistant" and msg["content"] == "Hello world"
        # 验证 chat 模板被套用后喂给引擎
        assert client._engine.seen_prompts[-1] == "[CHAT]user:hello[GEN]"

    @pytest.mark.unit
    def test_chat_stream_sse(self, client):
        r = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}], "stream": True})
        chunks = [l for l in r.text.split("\n\n") if l.strip()]
        assert chunks[-1] == "data: [DONE]"
        first = json.loads(chunks[0][len("data: "):])
        assert first["choices"][0]["delta"]["role"] == "assistant"
        # 末块带 finish_reason
        last = json.loads(chunks[-2][len("data: "):])
        assert last["choices"][0]["finish_reason"] == "stop"


# ─── 协议层映射 ──────────────────────────────────────────────────────────────
class TestProtocol:

    @pytest.mark.unit
    def test_to_sampling_params_mapping(self):
        req = CompletionRequest(
            model="m", prompt="x", max_tokens=32, temperature=0.7, top_p=0.9,
            top_k=20, min_p=0.1, presence_penalty=0.5, frequency_penalty=0.3,
            repetition_penalty=1.2, seed=42, stop=["\n"], logprobs=3)
        sp = req.to_sampling_params()
        assert sp.max_tokens == 32 and sp.temperature == 0.7
        assert sp.top_p == 0.9 and sp.top_k == 20 and sp.min_p == 0.1
        assert sp.presence_penalty == 0.5 and sp.frequency_penalty == 0.3
        assert sp.repetition_penalty == 1.2 and sp.seed == 42
        assert sp.stop == ["\n"] and sp.logprobs == 3
