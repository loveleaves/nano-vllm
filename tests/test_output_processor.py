"""OutputProcessor / IncrementalDetokenizer 单元测试。"""
import pytest

from nanovllm.engine.core_types import EngineCoreOutput, EngineCoreRequest, FinishReason
from nanovllm.engine.detokenizer import IncrementalDetokenizer, check_stop_strings
from nanovllm.engine.output_processor import OutputProcessor
from nanovllm.sampling_params import SamplingParams


class FakeTokenizer:
    def decode(self, ids: list[int]) -> str:
        return "".join(chr(i) for i in ids)


def _add(op: OutputProcessor, request_id: str, sp: SamplingParams) -> EngineCoreRequest:
    req = EngineCoreRequest(request_id, prompt_token_ids=[1], sampling_params=sp)
    op.add_request(req)
    return req


# ─── 增量 detokenize ──────────────────────────────────────────────────────────

@pytest.mark.unit
def test_detokenizer_delta_and_cumulative():
    d = IncrementalDetokenizer(FakeTokenizer())
    assert d.update([72, 105]) == "Hi"        # "Hi"
    assert d.update([33]) == "!"              # 增量只含新增
    assert d.text == "Hi!"                    # 累计 == 全量 decode
    assert d.output_token_ids == [72, 105, 33]


@pytest.mark.unit
def test_detokenizer_empty_update():
    d = IncrementalDetokenizer(FakeTokenizer())
    assert d.update([]) == ""


# ─── 停止串检测 ────────────────────────────────────────────────────────────────

@pytest.mark.unit
def test_check_stop_strings_picks_earliest():
    assert check_stop_strings("aXbY", ["Y", "X"]) == "X"
    assert check_stop_strings("abc", ["Z"]) is None
    assert check_stop_strings("abc", None) is None


# ─── OutputProcessor 端到端 ────────────────────────────────────────────────────

@pytest.mark.unit
def test_process_outputs_builds_request_output():
    op = OutputProcessor(FakeTokenizer())
    _add(op, "r0", SamplingParams())
    res = op.process_outputs([EngineCoreOutput("r0", [72, 105], finished=False)])
    (ro,) = res.request_outputs
    assert ro.text == "Hi" and ro.delta_text == "Hi"
    assert ro.token_ids == [72, 105] and not ro.finished
    assert res.reqs_to_abort == []


@pytest.mark.unit
def test_finish_reason_passthrough_from_core():
    op = OutputProcessor(FakeTokenizer())
    _add(op, "r0", SamplingParams())
    res = op.process_outputs([
        EngineCoreOutput("r0", [65], finished=True, finish_reason=FinishReason.LENGTH)
    ])
    (ro,) = res.request_outputs
    assert ro.finished and ro.finish_reason is FinishReason.LENGTH
    assert "r0" not in op.request_states           # 结束后清理


@pytest.mark.unit
def test_stop_string_truncates_and_aborts():
    op = OutputProcessor(FakeTokenizer())
    _add(op, "r0", SamplingParams(stop=["STOP"]))
    # 文本 "abSTOPcd"：命中 "STOP"，截断为 "ab"，标记结束并请求 abort
    res = op.process_outputs([
        EngineCoreOutput("r0", [97, 98, 83, 84, 79, 80, 99, 100], finished=False)
    ])
    (ro,) = res.request_outputs
    assert ro.text == "ab" and ro.delta_text == "ab"
    assert ro.finished and ro.finish_reason is FinishReason.STOP
    assert res.reqs_to_abort == ["r0"]


@pytest.mark.unit
def test_unknown_request_id_ignored():
    op = OutputProcessor(FakeTokenizer())
    res = op.process_outputs([EngineCoreOutput("ghost", [65])])
    assert res.request_outputs == []
