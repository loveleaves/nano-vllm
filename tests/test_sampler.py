"""Sampler 单元测试（CPU）：结构化 SamplingMetadata + greedy/top-k/top-p/penalties/logprobs。"""
import pytest
import torch

from nanovllm.layers.sample import Sampler, SamplingMetadata
from nanovllm.layers.sample.ops.bad_words import apply_bad_words
from nanovllm.layers.sample.ops.penalties import apply_all_penalties
from nanovllm.layers.sample.ops.topk_topp import apply_min_p, apply_top_k_top_p, random_sample


def _greedy_md(n):
    return SamplingMetadata(temperature=torch.zeros(n), all_greedy=True, all_random=False)


def _random_md(n, temp=1.0, top_p=None, top_k=None):
    return SamplingMetadata(
        temperature=torch.full((n,), temp), all_greedy=False, all_random=True,
        top_p=top_p, top_k=top_k)


class TestGreedy:

    @pytest.mark.unit
    def test_greedy_is_argmax(self):
        sampler = Sampler()
        logits = torch.zeros(5, 20)
        expected = torch.tensor([3, 7, 12, 0, 19])
        for i, idx in enumerate(expected):
            logits[i, idx] = 100.0
        out = sampler(logits, _greedy_md(5))
        assert torch.equal(out.sampled_token_ids, expected)
        assert out.sampled_token_ids.dtype == torch.int64

    @pytest.mark.unit
    def test_mixed_greedy_and_random_rows(self):
        sampler = Sampler()
        logits = torch.zeros(2, 10)
        logits[0, 4] = 100.0       # 行0 greedy → 必出 4
        logits[1, 9] = 100.0       # 行1 随机但 logit 极端 → 几乎必出 9
        md = SamplingMetadata(temperature=torch.tensor([0.0, 1.0]),
                              all_greedy=False, all_random=False)
        out = sampler(logits, md)
        assert out.sampled_token_ids[0].item() == 4


class TestRandom:

    @pytest.mark.unit
    def test_shape_dtype_range(self):
        sampler = Sampler()
        out = sampler(torch.randn(8, 500), _random_md(8))
        t = out.sampled_token_ids
        assert t.shape == (8,) and t.dtype == torch.int64
        assert (t >= 0).all() and (t < 500).all()

    @pytest.mark.unit
    def test_temperature_affects_agreement_with_argmax(self):
        torch.manual_seed(0)
        sampler = Sampler()
        logits = torch.randn(1000, 50)
        amax = logits.argmax(-1)
        low = (sampler(logits.clone(), _random_md(1000, 0.1)).sampled_token_ids == amax).float().mean()
        high = (sampler(logits.clone(), _random_md(1000, 10.0)).sampled_token_ids == amax).float().mean()
        assert low > high


class TestTopKTopP:

    @pytest.mark.unit
    def test_apply_top_k_only_masks_below_kth(self):
        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
        out = apply_top_k_top_p(logits.clone(), k=torch.tensor([2]), p=None)
        # 仅保留 top-2（4,5），其余 -inf
        assert torch.isinf(out[0, :3]).all() and out[0, 3] == 4.0 and out[0, 4] == 5.0

    @pytest.mark.unit
    def test_apply_top_p_keeps_nucleus(self):
        # 一个 token 概率压倒性 → top_p 很小也至少保留它
        logits = torch.tensor([[0.0, 0.0, 100.0, 0.0]])
        out = apply_top_k_top_p(logits.clone(), k=None, p=torch.tensor([0.5]))
        assert out[0, 2] == 100.0
        assert torch.isinf(out[0, [0, 1, 3]]).all()

    @pytest.mark.unit
    def test_top_k_restricts_sampled_tokens(self):
        torch.manual_seed(0)
        sampler = Sampler()
        logits = torch.zeros(200, 6)
        logits[:, 4] = 1.0
        logits[:, 5] = 1.2          # 只有 4,5 较大
        md = _random_md(200, temp=1.0, top_k=torch.tensor([2] * 200))
        out = sampler(logits, md)
        assert set(out.sampled_token_ids.tolist()) <= {4, 5}


class TestMinP:

    @pytest.mark.unit
    def test_min_p_masks_low_prob_tokens(self):
        # 一个 token 概率压倒性 → min_p=0.5 屏蔽其余
        logits = torch.tensor([[0.0, 0.0, 10.0, 0.0]])
        out = apply_min_p(logits.clone(), torch.tensor([0.5]))
        assert out[0, 2] == 10.0
        assert torch.isinf(out[0, [0, 1, 3]]).all()

    @pytest.mark.unit
    def test_min_p_zero_keeps_all(self):
        logits = torch.randn(1, 8)
        out = apply_min_p(logits.clone(), torch.tensor([0.0]))
        assert torch.equal(out, logits)   # min_p=0 不屏蔽任何 token


class TestSeedGenerators:

    @pytest.mark.unit
    def test_random_sample_with_generator_reproducible(self):
        probs = torch.softmax(torch.randn(1, 50), dim=-1)
        g1 = torch.Generator(); g1.manual_seed(123)
        g2 = torch.Generator(); g2.manual_seed(123)
        t1 = random_sample(probs.clone(), {0: g1})
        t2 = random_sample(probs.clone(), {0: g2})
        assert torch.equal(t1, t2)        # 同种子 → 同结果（可复现）

    @pytest.mark.unit
    def test_sampler_seed_reproducible(self):
        sampler = Sampler()
        logits = torch.randn(2, 100)
        def run():
            g = {0: torch.Generator(), 1: torch.Generator()}
            g[0].manual_seed(7); g[1].manual_seed(8)
            md = SamplingMetadata(temperature=torch.ones(2), all_greedy=False,
                                  all_random=True, generators=g)
            return sampler(logits.clone(), md).sampled_token_ids
        assert torch.equal(run(), run())


class TestBadWords:

    @pytest.mark.unit
    def test_single_token_bad_word_always_masked(self):
        logits = torch.zeros(1, 5)
        out = apply_bad_words(logits.clone(), {0: [[3]]}, [[1, 2]])
        assert torch.isinf(out[0, 3]) and not torch.isinf(out[0, 0])

    @pytest.mark.unit
    def test_multi_token_bad_word_masks_on_prefix_match(self):
        # 禁止序列 [2,3]：仅当已生成尾部为 [2] 时屏蔽 3
        logits = torch.zeros(2, 5)
        out = apply_bad_words(logits.clone(), {0: [[2, 3]], 1: [[2, 3]]},
                              [[9, 2], [9, 9]])   # row0 尾部=2(匹配)，row1 不匹配
        assert torch.isinf(out[0, 3])             # row0：屏蔽 3
        assert not torch.isinf(out[1, 3])         # row1：不屏蔽


class TestPenalties:

    @pytest.mark.unit
    def test_repetition_penalty_suppresses_seen_token(self):
        logits = torch.tensor([[1.0, 1.0, 1.0]])
        out = apply_all_penalties(
            logits.clone(), prompt_token_ids=[[]], output_token_ids=[[0]],
            presence_penalties=torch.tensor([0.0]),
            frequency_penalties=torch.tensor([0.0]),
            repetition_penalties=torch.tensor([2.0]))
        assert out[0, 0].item() == pytest.approx(0.5)   # 出现过且 logit>0 → /2
        assert out[0, 1].item() == 1.0

    @pytest.mark.unit
    def test_frequency_and_presence_penalty(self):
        logits = torch.zeros(1, 3)
        out = apply_all_penalties(
            logits.clone(), prompt_token_ids=[[]], output_token_ids=[[1, 1]],
            presence_penalties=torch.tensor([0.5]),
            frequency_penalties=torch.tensor([0.25]),
            repetition_penalties=torch.tensor([1.0]))
        # token1 出现 2 次：-freq*2 - pres*1 = -0.5 - 0.5 = -1.0
        assert out[0, 1].item() == pytest.approx(-1.0)
        assert out[0, 0].item() == 0.0

    @pytest.mark.unit
    def test_no_penalties_path_unchanged(self):
        sampler = Sampler()
        logits = torch.zeros(3, 8)
        logits[range(3), [1, 2, 3]] = 50.0
        # greedy + no_penalties：结果就是 argmax，惩罚分支被跳过
        out = sampler(logits.clone(), _greedy_md(3))
        assert out.sampled_token_ids.tolist() == [1, 2, 3]


class TestLogprobs:

    @pytest.mark.unit
    def test_logprobs_returned_with_shape(self):
        sampler = Sampler()
        logits = torch.randn(4, 100)
        md = SamplingMetadata(temperature=torch.zeros(4), all_greedy=True,
                              all_random=False, max_num_logprobs=5)
        out = sampler(logits, md)
        lp = out.logprobs_tensors
        assert lp is not None
        assert lp.logprob_token_ids.shape == (4, 6)   # 采样 token + top-5
        assert lp.logprobs.shape == (4, 6)
        assert lp.selected_token_ranks.shape == (4,)
        # 第 0 列为采样 token 自身
        assert torch.equal(lp.logprob_token_ids[:, 0].long(), out.sampled_token_ids)

    @pytest.mark.unit
    def test_greedy_sampled_token_rank_is_one(self):
        sampler = Sampler()
        logits = torch.zeros(2, 10)
        logits[0, 3] = 10.0
        logits[1, 7] = 10.0
        md = SamplingMetadata(temperature=torch.zeros(2), all_greedy=True,
                              all_random=False, max_num_logprobs=3)
        out = sampler(logits, md)
        # greedy 取最大 logit → 排名必为 1
        assert out.logprobs_tensors.selected_token_ranks.tolist() == [1, 1]
