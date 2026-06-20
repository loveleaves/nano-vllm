"""
投机解码单测（CPU，免 GPU）：n-gram 草案器 + 拒绝采样 + 编排（含贪心等价性）。

运行：pytest tests/test_spec_decode.py -m unit -v
"""
import pytest

from nanovllm.sample.rejection_sampler import RejectionSampler
from nanovllm.spec_decode import NgramProposer, SpeculativeDecoder


# ─── NgramProposer ───────────────────────────────────────────────────────────
class TestNgramProposer:

    @pytest.mark.unit
    def test_basic_match(self):
        # [1,2,3,1,2] → 尾部 [1,2] 在 index0 出现，提议其后的 k 个 token（仅余 [3,1,2]）
        p = NgramProposer(min_n=1, max_n=3, k=4)
        assert p.propose([1, 2, 3, 1, 2]) == [3, 1, 2]

    @pytest.mark.unit
    def test_prefers_longer_ngram(self):
        # period-3 序列：尾部 [1,2,3] 命中 index0，提议其后的 token（仅余 [1,2,3]）
        p = NgramProposer(min_n=1, max_n=3, k=4)
        assert p.propose([1, 2, 3, 1, 2, 3]) == [1, 2, 3]

    @pytest.mark.unit
    def test_k_limit(self):
        p = NgramProposer(min_n=1, max_n=2, k=2)
        assert len(p.propose([5, 6, 5, 6, 5, 6])) <= 2

    @pytest.mark.unit
    def test_no_match_returns_empty(self):
        p = NgramProposer(min_n=2, max_n=3, k=4)
        assert p.propose([1, 2, 3, 4, 5]) == []

    @pytest.mark.unit
    def test_short_sequence(self):
        assert NgramProposer().propose([7]) == []


# ─── RejectionSampler ────────────────────────────────────────────────────────
class TestRejectionSampler:

    @pytest.mark.unit
    def test_all_accepted_plus_bonus(self):
        # draft 全对 → 接受全部 + 奖励 token
        assert RejectionSampler.verify_greedy([1, 2], [1, 2, 3]) == [1, 2, 3]

    @pytest.mark.unit
    def test_mismatch_correction(self):
        # 第 2 位分歧 → 接受 [1] + 修正 9，停止
        assert RejectionSampler.verify_greedy([1, 2], [1, 9, 3]) == [1, 9]

    @pytest.mark.unit
    def test_first_token_mismatch(self):
        assert RejectionSampler.verify_greedy([5, 6], [7, 8, 9]) == [7]

    @pytest.mark.unit
    def test_empty_draft_single_token(self):
        # 无草案 → 退化为单 token
        assert RejectionSampler.verify_greedy([], [42]) == [42]

    @pytest.mark.unit
    def test_length_assert(self):
        with pytest.raises(AssertionError):
            RejectionSampler.verify_greedy([1, 2], [1])


# ─── 编排 + 贪心等价性 ───────────────────────────────────────────────────────
def _make_score_fn(target_next):
    """目标模型评分：对 history + draft[:i] 的每个位置取贪心 argmax（含奖励位）。"""
    def score_fn(history, draft):
        out, h = [], list(history)
        for d in draft:
            out.append(target_next(h))
            h.append(d)
        out.append(target_next(h))
        return out
    return score_fn


class TestSpeculativeDecoder:

    @pytest.mark.unit
    def test_step_accepts_correct_draft(self):
        # period-3 真序列：target_next = 复制 3 步前
        target_next = lambda h: h[-3]
        sd = SpeculativeDecoder(NgramProposer(min_n=1, max_n=3, k=4))
        accepted = sd.step([1, 2, 3, 1, 2, 3], _make_score_fn(target_next))
        assert accepted == [1, 2, 3, 1]   # 草案全中 + 奖励

    @pytest.mark.unit
    def test_greedy_equivalence(self):
        """投机解码贪心结果须逐 token 等于自回归贪心。"""
        target_next = lambda h: h[-3]          # period-3 可重复，n-gram 易命中
        prompt = [1, 2, 3]
        target_len = 30

        # 参考：纯自回归贪心
        ref = list(prompt)
        while len(ref) < target_len:
            ref.append(target_next(ref))

        # 投机解码
        sd = SpeculativeDecoder(NgramProposer(min_n=1, max_n=3, k=4))
        score_fn = _make_score_fn(target_next)
        seq = list(prompt)
        steps = 0
        while len(seq) < target_len:
            seq.extend(sd.step(seq, score_fn))
            steps += 1
            assert steps < target_len   # 防止死循环（每步至少进 1 token）

        assert seq[:target_len] == ref[:target_len]
        # 应明显少于逐 token 步数（投机加速）
        assert steps < target_len - len(prompt)

    @pytest.mark.unit
    def test_fallback_no_draft(self):
        # 非重复序列：n-gram 不命中 → 每步退化为单 token，仍正确
        target_next = lambda h: len(h)         # 严格递增，无重复
        sd = SpeculativeDecoder(NgramProposer(min_n=2, max_n=3, k=4))
        accepted = sd.step([0, 1, 2, 3], _make_score_fn(target_next))
        assert accepted == [4]
