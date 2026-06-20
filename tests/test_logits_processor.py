"""
Logits Processor 框架（R）+ 引导解码（S）单测（CPU，免 GPU）。

运行：pytest tests/test_logits_processor.py -m unit -v
"""
import math

import pytest
import torch

from nanovllm.sample.metadata import SamplingMetadata
from nanovllm.sample.sampler import Sampler
from nanovllm.sample.logits_processor import (
    BadWordsLogitsProcessor,
    LogitBiasLogitsProcessor,
    MinTokensLogitsProcessor,
    PenaltiesLogitsProcessor,
    build_logits_processors,
)
from nanovllm.sample.guided import (
    ChoiceGrammar,
    GuidedDecodingLogitsProcessor,
    build_grammar,
)


def _greedy_md(n=1, **kw) -> SamplingMetadata:
    return SamplingMetadata(temperature=torch.zeros(n), all_greedy=True,
                            all_random=False, **kw)


# ─── R：LogitsProcessor 框架 ─────────────────────────────────────────────────
class TestFrameworkOrder:

    @pytest.mark.unit
    def test_default_chain_order(self):
        chain = build_logits_processors()
        names = [type(p).__name__ for p in chain]
        assert names == [
            "PenaltiesLogitsProcessor", "BadWordsLogitsProcessor",
            "LogitBiasLogitsProcessor", "MinTokensLogitsProcessor",
            "GuidedDecodingLogitsProcessor",
        ]


class TestLogitBias:

    @pytest.mark.unit
    def test_bias_added(self):
        logits = torch.zeros(1, 5)
        md = _greedy_md(logit_bias={0: {3: 10.0}})
        out = LogitBiasLogitsProcessor().apply(logits.clone(), md)
        assert out[0, 3].item() == 10.0
        assert out[0, 0].item() == 0.0

    @pytest.mark.unit
    def test_noop_when_empty(self):
        logits = torch.randn(1, 5)
        md = _greedy_md()
        assert torch.equal(LogitBiasLogitsProcessor().apply(logits.clone(), md), logits)

    @pytest.mark.unit
    def test_bias_flips_greedy(self):
        # 原 argmax=0；对 token 4 加大偏置后应选 4
        logits = torch.tensor([[5., 0., 0., 0., 1.]])
        md = _greedy_md(logit_bias={0: {4: 100.0}})
        out = Sampler()(logits, md)
        assert out.sampled_token_ids.item() == 4


class TestMinTokens:

    @pytest.mark.unit
    def test_suppress_eos_below_threshold(self):
        logits = torch.zeros(1, 5)
        md = _greedy_md(min_tokens={0: 3}, eos_token_id=2,
                        output_token_ids=[[7]])   # 已生成 1 < 3
        out = MinTokensLogitsProcessor().apply(logits.clone(), md)
        assert out[0, 2].item() == -math.inf

    @pytest.mark.unit
    def test_allow_eos_at_threshold(self):
        logits = torch.zeros(1, 5)
        md = _greedy_md(min_tokens={0: 2}, eos_token_id=2,
                        output_token_ids=[[7, 8]])   # 已生成 2 >= 2
        out = MinTokensLogitsProcessor().apply(logits.clone(), md)
        assert out[0, 2].item() == 0.0


class TestPenaltiesBadWordsWrappers:

    @pytest.mark.unit
    def test_penalties_noop(self):
        logits = torch.randn(1, 5)
        md = _greedy_md(no_penalties=True)
        assert torch.equal(PenaltiesLogitsProcessor().apply(logits.clone(), md), logits)

    @pytest.mark.unit
    def test_bad_words_noop(self):
        logits = torch.randn(1, 5)
        md = _greedy_md(bad_words_token_ids=None)
        assert torch.equal(BadWordsLogitsProcessor().apply(logits.clone(), md), logits)


# ─── S：ChoiceGrammar FSM ────────────────────────────────────────────────────
class TestChoiceGrammar:

    @pytest.mark.unit
    def test_basic_choice_flow(self):
        g = ChoiceGrammar([[1, 2], [3]], eos_token_id=0)
        assert g.allowed_token_ids() == {1, 3}     # 两候选首 token
        g.accept(1)
        assert g.allowed_token_ids() == {2}        # 只剩 [1,2] 可行
        g.accept(2)
        assert g.allowed_token_ids() == {0}        # 候选完成 → 只允许 EOS
        assert not g.is_complete()
        g.accept(0)
        assert g.is_complete()

    @pytest.mark.unit
    def test_prefix_overlap(self):
        # ["a", "ab"] → 匹配 a 后既可完成（EOS）也可延伸（b）
        g = ChoiceGrammar([[1], [1, 2]], eos_token_id=0)
        assert g.allowed_token_ids() == {1}
        g.accept(1)
        assert g.allowed_token_ids() == {0, 2}     # 完成可用 + 可延伸
        g.accept(2)
        assert g.allowed_token_ids() == {0}
        g.accept(0)
        assert g.is_complete()

    @pytest.mark.unit
    def test_invalid_token_empties(self):
        g = ChoiceGrammar([[1, 2]], eos_token_id=0)
        g.accept(9)                                # 非法 token → 无可行候选
        assert g.allowed_token_ids() == set()

    @pytest.mark.unit
    def test_build_grammar_with_tokenizer(self):
        class Tok:
            eos_token_id = 0
            def encode(self, s, add_special_tokens=False):
                return [ord(c) for c in s]
        g = build_grammar(["hi"], Tok(), 0)
        assert g.allowed_token_ids() == {ord("h")}
        assert build_grammar(None, Tok(), 0) is None


class TestGuidedProcessorAndSampler:

    @pytest.mark.unit
    def test_mask_to_allowed(self):
        g = ChoiceGrammar([[1, 2], [3]], eos_token_id=0)
        logits = torch.zeros(1, 5)
        md = _greedy_md(grammars={0: g})
        out = GuidedDecodingLogitsProcessor().apply(logits.clone(), md)
        # 仅 {1,3} 保留，其余 -inf
        assert out[0, 1].item() == 0.0 and out[0, 3].item() == 0.0
        assert out[0, 0].item() == -math.inf and out[0, 2].item() == -math.inf

    @pytest.mark.unit
    def test_sampler_constrains_and_advances(self):
        g = ChoiceGrammar([[1, 2], [3]], eos_token_id=0)
        sampler = Sampler()
        # logits argmax 本是 4，但仅 {1,3} 允许；3 的 logit 更大 → 选 3
        logits = torch.tensor([[0., 1., 0., 5., 9.]])
        md = _greedy_md(grammars={0: g})
        out = sampler(logits, md)
        assert out.sampled_token_ids.item() == 3
        assert g.allowed_token_ids() == {0}        # [3] 完成 → grammar 已推进

    @pytest.mark.unit
    def test_sampler_two_step_choice(self):
        g = ChoiceGrammar([[1, 2], [3]], eos_token_id=0)
        sampler = Sampler()
        # 第一步 1 的 logit 最大（在允许集 {1,3} 内）
        out1 = sampler(torch.tensor([[0., 9., 0., 5., 0.]]), _greedy_md(grammars={0: g}))
        assert out1.sampled_token_ids.item() == 1
        # 第二步只允许 2
        out2 = sampler(torch.tensor([[9., 9., 1., 9., 9.]]), _greedy_md(grammars={0: g}))
        assert out2.sampled_token_ids.item() == 2
