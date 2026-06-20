"""Grammar 接口 + ChoiceGrammar（候选集约束，trie/FSM 实现）。"""
from abc import ABC, abstractmethod


class Grammar(ABC):
    """逐 token 约束的有限状态机。

    allowed_token_ids: 当前状态允许的 token 集合；None 表示不约束（全允许）。
    accept: 用刚采样的 token 推进状态。
    is_complete: 约束已满足且已停止（接受 EOS 后）。
    """

    @abstractmethod
    def allowed_token_ids(self) -> set[int] | None:
        ...

    @abstractmethod
    def accept(self, token_id: int) -> None:
        ...

    @abstractmethod
    def is_complete(self) -> bool:
        ...


class ChoiceGrammar(Grammar):
    """约束输出 token 序列等于若干候选之一（已 tokenize 为 list[list[int]]）。

    状态 = 仍可行的候选 (tokens, pos) 集合。某候选 pos 抵达末尾即"可完成"，此时额外允许
    EOS；接受 EOS 即完成并停止。支持候选互为前缀（如 ["a","ab"]）：可完成与可延伸并存。
    """

    def __init__(self, choices_token_ids: list[list[int]], eos_token_id: int):
        # 过滤空候选
        self.candidates: list[tuple[tuple[int, ...], int]] = [
            (tuple(c), 0) for c in choices_token_ids if c
        ]
        self.eos_token_id = eos_token_id
        self._done = False

    def _complete_available(self) -> bool:
        return any(pos == len(toks) for toks, pos in self.candidates)

    def allowed_token_ids(self) -> set[int] | None:
        if self._done:
            return {self.eos_token_id}
        # 可延伸候选的下一个 token：均允许
        allowed = {toks[pos] for toks, pos in self.candidates if pos < len(toks)}
        # 若已有候选恰好匹配完整（pos 到末尾），则额外允许 EOS——这样能同时支持"互为前缀"的
        # 候选（如 ["a","ab"]：匹配到 a 后，既可发 EOS 结束于 "a"，也可发 b 继续走向 "ab"）。
        if self._complete_available():
            allowed.add(self.eos_token_id)
        return allowed

    def accept(self, token_id: int) -> None:
        if self._done:
            return
        if token_id == self.eos_token_id and self._complete_available():
            self._done = True
            self.candidates = []
            return
        # 推进：保留下一个 token 命中的候选
        self.candidates = [
            (toks, pos + 1)
            for toks, pos in self.candidates
            if pos < len(toks) and toks[pos] == token_id
        ]

    def is_complete(self) -> bool:
        return self._done


def build_grammar(guided_choice, tokenizer, eos_token_id: int) -> Grammar | None:
    """从 SamplingParams.guided_choice 构造 Grammar。

    guided_choice: list[str]，输出须等于其中之一。用 tokenizer 把每个候选编码为 token 序列
    （不加特殊 token）。返回 None 表示该请求无引导约束。
    """
    if not guided_choice:
        return None
    choices_ids = [tokenizer.encode(c, add_special_tokens=False) for c in guided_choice]
    return ChoiceGrammar(choices_ids, eos_token_id)
