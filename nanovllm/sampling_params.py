from dataclasses import dataclass


@dataclass
class SamplingParams:
    """
    单次生成请求的采样超参数（对齐 vLLM `SamplingParams` 的常用子集）。

    字段说明：
      temperature        — 采样温度；< 1e-5 视为 greedy（argmax，真贪心）
      max_tokens         — 最多生成的 token 数（不含 prompt）
      ignore_eos         — 为 True 时忽略 EOS token，强制生成到 max_tokens
      stop               — 停止字符串列表；OutputProcessor 在增量 detokenize 文本中
                           命中任一子串即终止该请求（finish_reason="stop"）
      top_p              — 核采样累积概率阈值（1.0 关闭）
      top_k              — top-k 截断（<=0 关闭）
      presence_penalty   — 存在惩罚（出现过即扣分）
      frequency_penalty  — 频率惩罚（按出现次数扣分）
      repetition_penalty — 重复惩罚（>1 抑制重复，=1 关闭）
      logprobs           — 每步返回的 top-logprobs 个数（None 不返回）
    """
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False
    stop: list[str] | None = None
    top_p: float = 1.0
    top_k: int = -1
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: float = 1.0
    logprobs: int | None = None

    def __post_init__(self):
        assert self.temperature >= 0.0, "temperature 必须 >= 0（0 表示 greedy）"
        assert self.max_tokens > 0, "max_tokens 必须大于 0"
        assert 0.0 < self.top_p <= 1.0, "top_p 必须落在 (0, 1]"
        assert self.repetition_penalty > 0.0, "repetition_penalty 必须 > 0"
        assert self.logprobs is None or self.logprobs >= 0, "logprobs 必须 >= 0"
        if isinstance(self.stop, str):
            self.stop = [self.stop]
