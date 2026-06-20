"""投机解码编排：propose → score → verify。"""
from nanovllm.sample.rejection_sampler import RejectionSampler
from nanovllm.spec_decode.ngram_proposer import NgramProposer


class SpeculativeDecoder:
    """把 proposer + 目标模型评分 + 拒绝采样串成一步多 token 的编排。

    与 GPU 无关：目标模型评分以 score_fn 注入，便于单测与替换不同 proposer/目标后端。

    score_fn(token_ids, draft) -> list[int]：目标模型对 k+1 个位置（当前 token + draft）
    并行前向后的贪心结果（argmax），长度须为 len(draft)+1。GPU 通路里它就是"目标模型多
    位置前向 + 取 argmax"（见 docs/arch_spec_decode/design.md 的集成边界）。
    """

    def __init__(self, proposer: NgramProposer | None = None,
                 rejection_sampler: RejectionSampler | None = None):
        self.proposer = proposer or NgramProposer()
        self.rejection_sampler = rejection_sampler or RejectionSampler()

    def step(self, token_ids: list[int], score_fn) -> list[int]:
        """对给定历史推进一步，返回本步接受的 token 序列（≥1）。

        无草案（n-gram 未命中）时退化为普通单步：score_fn(token_ids, []) 返回 1 个 token。
        """
        draft = self.proposer.propose(token_ids)
        target = score_fn(token_ids, draft)
        return self.rejection_sampler.verify_greedy(draft, target)
