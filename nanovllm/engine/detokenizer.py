"""
增量 detokenize（对齐 vLLM V1 `v1/engine/detokenizer.py` 的 IncrementalDetokenizer）。

V1 用前缀缓冲 + 逐 token 流式解码以避免重复 decode；nano 取教学化简版：维护完整
输出 token 列表，每步对全量 decode 一次并切出增量文本。保证 **最终累计文本严格等于
`tokenizer.decode(全部输出 token)`**（与旧版 generate 末尾的一次性 decode 完全一致），
代价是 O(n²) 解码——对教学/中等长度可接受，已在 design.md 注明权衡。
"""


class IncrementalDetokenizer:

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.output_token_ids: list[int] = []
        self.text = ""              # 累计文本

    def update(self, new_token_ids: list[int]) -> str:
        """追加新 token，返回本步新增（增量）文本。"""
        if not new_token_ids:
            return ""
        self.output_token_ids.extend(new_token_ids)
        full = self.tokenizer.decode(self.output_token_ids)
        delta = full[len(self.text):]
        self.text = full
        return delta


def check_stop_strings(text: str, stop: list[str] | None) -> str | None:
    """返回 text 中命中的首个停止串（按出现位置最靠前者），未命中返回 None。"""
    if not stop:
        return None
    best_pos = None
    best_str = None
    for s in stop:
        if not s:
            continue
        pos = text.find(s)
        if pos != -1 and (best_pos is None or pos < best_pos):
            best_pos = pos
            best_str = s
    return best_str
