"""N-gram 草案器（对齐 vLLM `v1/spec_decode/ngram_proposer.py`）。

无需草案模型：取当前序列尾部长度 n 的 n-gram，在更早处找到相同 n-gram 的最近一次出现，
把其后的 k 个 token 作为提议续写。对"重复/回拷"型文本（代码、引用、列表）命中率高，
是最轻量的投机解码 proposer。
"""


class NgramProposer:
    """用历史 n-gram 匹配提议 k 个候选 token。

    min_n / max_n：尝试的 n-gram 长度范围（优先用更长的 n-gram，匹配更可信）。
    k：单步最多提议的 token 数（投机深度）。
    """

    def __init__(self, min_n: int = 1, max_n: int = 3, k: int = 4):
        assert 1 <= min_n <= max_n
        assert k >= 1
        self.min_n = min_n
        self.max_n = max_n
        self.k = k

    def propose(self, token_ids: list[int]) -> list[int]:
        """返回最多 k 个提议 token；无匹配时返回 []。"""
        n_tokens = len(token_ids)
        max_n = min(self.max_n, n_tokens - 1)
        for n in range(max_n, self.min_n - 1, -1):
            suffix = token_ids[-n:]
            # 在 [0, n_tokens-n) 范围内自后向前找相同 n-gram 的最近一次出现
            for start in range(n_tokens - n - 1, -1, -1):
                if token_ids[start:start + n] == suffix:
                    proposal = token_ids[start + n:start + n + self.k]
                    if proposal:
                        return proposal
                    break  # 命中但其后无 token，换更短 n-gram
        return []
