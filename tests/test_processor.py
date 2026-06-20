"""Processor 单元测试（输入处理：tokenize + request_id 分配）。"""
import pytest

from nanovllm.engine.processor import Processor
from nanovllm.sampling_params import SamplingParams


class FakeTokenizer:
    eos_token_id = 999

    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(i) for i in ids)


@pytest.mark.unit
def test_encode_str_prompt():
    p = Processor(FakeTokenizer())
    req = p.process_inputs("ab", SamplingParams())
    assert req.prompt_token_ids == [97, 98]


@pytest.mark.unit
def test_token_ids_passthrough():
    p = Processor(FakeTokenizer())
    req = p.process_inputs([1, 2, 3], SamplingParams())
    assert req.prompt_token_ids == [1, 2, 3]


@pytest.mark.unit
def test_auto_request_id_increments():
    p = Processor(FakeTokenizer())
    r0 = p.process_inputs([1], SamplingParams())
    r1 = p.process_inputs([2], SamplingParams())
    assert (r0.request_id, r1.request_id) == ("0", "1")


@pytest.mark.unit
def test_explicit_request_id():
    p = Processor(FakeTokenizer())
    req = p.process_inputs([1], SamplingParams(), request_id="abc")
    assert req.request_id == "abc"


@pytest.mark.unit
def test_empty_prompt_rejected():
    p = Processor(FakeTokenizer())
    with pytest.raises(AssertionError):
        p.process_inputs([], SamplingParams())
