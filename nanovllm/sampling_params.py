from dataclasses import dataclass


@dataclass
class SamplingParams:
    """
    单次生成请求的采样超参数。

    字段说明：
      temperature  — 采样温度，值越大输出越随机
      max_tokens   — 最多生成的 token 数（不含 prompt）
      ignore_eos   — 为 True 时忽略 EOS token，强制生成到 max_tokens
      stop         — 停止字符串列表；OutputProcessor 在增量 detokenize 文本中
                     命中任一子串即终止该请求（finish_reason="stop"）
    """
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False
    stop: list[str] | None = None

    def __post_init__(self):
        assert self.temperature > 1e-10, "temperature 必须大于 0（不支持 greedy）"
        assert self.max_tokens > 0, "max_tokens 必须大于 0"
        if isinstance(self.stop, str):
            self.stop = [self.stop]
