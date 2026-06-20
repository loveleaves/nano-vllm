"""
投机解码引擎集成单测（CPU，免 GPU）。

object.__new__(EngineCore) + 真 Scheduler（纯 Python 块管理）+ 假 Executor（execute_model
基准步 + verify_spec 验证），检验 _step_spec 的多 token 扩展：接受全部 / 部分 / 结束截断 /
块回滚（truncate_blocks + num_cached 不变式）/ 引擎级贪心等价。GPU verify_spec 由集成边界
覆盖（见 docs/arch_spec_decode），此处只测编排与调度/块逻辑。
"""
import pytest

from nanovllm.engine.core import EngineCore
from nanovllm.engine.core_types import EngineCoreRequest, FinishReason
from nanovllm.engine.sched import Scheduler
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams
from nanovllm.spec_decode import NgramProposer, SpeculativeDecoder


class FixedProposer:
    """固定草案（忽略历史），用于精确控制 accept/reject 场景。"""
    def __init__(self, drafts):
        self.drafts = drafts
    def propose(self, token_ids):
        return list(self.drafts)


class SpecFakeExecutor:
    """execute_model 返回固定基准 token；verify_spec 返回固定 k+1 目标。"""
    def __init__(self, base_token, verify_targets):
        self.base_token = base_token
        self.verify_targets = verify_targets
    def execute_model(self, seqs, finished_seq_ids=None):
        return [self.base_token for _ in seqs], None
    def verify_spec(self, seq, num_drafts):
        return list(self.verify_targets)


class PeriodExecutor:
    """period-3 真实目标：next = 复制 3 步前（用于引擎级贪心等价）。"""
    def execute_model(self, seqs, finished_seq_ids=None):
        return [s.token_ids[-3] for s in seqs], None
    def verify_spec(self, seq, num_drafts):
        n, k = seq.num_tokens, num_drafts
        L0 = n - k
        return [seq.token_ids[:L0 + i][-3] for i in range(k + 1)]


def _make_spec_core(executor, spec_decoder, block_size=4, num_blocks=64, eos=999):
    Sequence.block_size = block_size
    ec = object.__new__(EngineCore)
    ec.executor = executor
    ec.scheduler = Scheduler(num_blocks, block_size, max_num_seqs=8,
                             max_num_batched_tokens=256, eos=eos)
    ec.requests = {}
    ec.async_scheduling = False
    ec._inflight = None
    ec.use_spec = True
    ec.spec_decoder = spec_decoder
    ec.scheduler_stats = None
    return ec


def _add(ec, rid, prompt, **sp):
    ec.add_request(EngineCoreRequest(rid, list(prompt), SamplingParams(**sp)))


# ─── 接受全部 / 部分 / 结束 ───────────────────────────────────────────────────
class TestSpecExtend:

    @pytest.mark.unit
    def test_accept_all_plus_bonus(self):
        ex = SpecFakeExecutor(base_token=5, verify_targets=[6, 7, 8])
        ec = _make_spec_core(ex, SpeculativeDecoder(FixedProposer([6, 7])))
        _add(ec, "r0", [1, 2, 3], max_tokens=20, ignore_eos=True)

        # 首步 = prefill + 基准 token + 投机扩展（base 后 seq 已进 decode，扩展即生效）
        out = ec.step()
        (o,) = out.outputs
        assert o.new_token_ids == [5, 6, 7, 8]    # 基准 5 + 草案 6,7 全中 + 奖励 8
        seq = ec.requests["r0"]
        assert seq.token_ids == [1, 2, 3, 5, 6, 7, 8]
        assert seq.num_cached_tokens == seq.num_tokens - 1   # decode 不变式

    @pytest.mark.unit
    def test_partial_accept(self):
        ex = SpecFakeExecutor(base_token=5, verify_targets=[6, 99, 8])  # 第2位分歧
        ec = _make_spec_core(ex, SpeculativeDecoder(FixedProposer([6, 7])))
        _add(ec, "r0", [1, 2, 3], max_tokens=20, ignore_eos=True)
        out = ec.step()
        (o,) = out.outputs
        assert o.new_token_ids == [5, 6, 99]      # 6 接受、7 被 99 修正后停止
        assert ec.requests["r0"].token_ids == [1, 2, 3, 5, 6, 99]

    @pytest.mark.unit
    def test_finish_by_max_tokens_mid_spec(self):
        ex = SpecFakeExecutor(base_token=5, verify_targets=[6, 7, 8])
        ec = _make_spec_core(ex, SpeculativeDecoder(FixedProposer([6, 7])))
        _add(ec, "r0", [1, 2, 3], max_tokens=3, ignore_eos=True)   # 仅允许 3 个 completion
        out = ec.step()
        (o,) = out.outputs
        assert o.new_token_ids == [5, 6, 7]       # 5,6,7 = 3 个 → LENGTH 截断
        assert o.finished and o.finish_reason is FinishReason.LENGTH
        assert "r0" not in ec.requests

    @pytest.mark.unit
    def test_finish_by_eos_mid_spec(self):
        ex = SpecFakeExecutor(base_token=5, verify_targets=[6, 999, 8])  # 999 = eos
        ec = _make_spec_core(ex, SpeculativeDecoder(FixedProposer([6, 7])), eos=999)
        _add(ec, "r0", [1, 2, 3], max_tokens=20)   # 不 ignore_eos
        out = ec.step()
        (o,) = out.outputs
        assert o.new_token_ids == [5, 6, 999]
        assert o.finished and o.finish_reason is FinishReason.STOP

    @pytest.mark.unit
    def test_no_draft_falls_back(self):
        # 草案为空 → 不扩展，等价普通 decode（只 1 个基准 token）
        ex = SpecFakeExecutor(base_token=5, verify_targets=[])
        ec = _make_spec_core(ex, SpeculativeDecoder(FixedProposer([])))
        _add(ec, "r0", [1, 2, 3], max_tokens=20, ignore_eos=True)
        out = ec.step()
        assert out.outputs[0].new_token_ids == [5]


# ─── 引擎级贪心等价 ──────────────────────────────────────────────────────────
class TestEngineGreedyEquivalence:

    @pytest.mark.unit
    def test_period3_equivalence(self):
        """period-3 真实目标 + 真 NgramProposer：投机完成的 completion 须等于参考贪心。"""
        max_tokens = 12
        # 参考：纯自回归 period-3
        ref = [1, 2, 3]
        while len(ref) - 3 < max_tokens:
            ref.append(ref[-3])
        ref_completion = ref[3:3 + max_tokens]

        ec = _make_spec_core(PeriodExecutor(),
                             SpeculativeDecoder(NgramProposer(max_n=3, k=4)))
        _add(ec, "r0", [1, 2, 3], max_tokens=max_tokens, ignore_eos=True)

        completion, steps = [], 0
        while ec.has_unfinished_requests():
            out = ec.step()
            for o in out.outputs:
                completion.extend(o.new_token_ids)
            steps += 1
            assert steps < 100

        assert completion == ref_completion
        # 投机加速：步数应明显少于逐 token（12 个 completion）
        assert steps < max_tokens
