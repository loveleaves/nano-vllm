"""LogitsProcessor 接口 + 默认装配。"""
from abc import ABC, abstractmethod

import torch


class LogitsProcessor(ABC):
    """对一批 logits 就地/返回式修改的可插拔处理器。

    apply 应：从 SamplingMetadata 读自己的配置；整批无该配置时直接返回原 logits
    （快速 no-op）；否则对相关行生效并返回（修改后的）logits。
    """

    @abstractmethod
    def apply(self, logits: torch.Tensor, metadata) -> torch.Tensor:
        ...


def build_logits_processors() -> list[LogitsProcessor]:
    """默认处理器链（顺序敏感）。

    惩罚 → bad_words → logit_bias → min_tokens → guided。引导（grammar 掩码）置于最后，
    使其约束相对其它处理器具有最终决定权。
    """
    # 延迟导入避免循环依赖（guided 依赖本模块的 LogitsProcessor）
    from nanovllm.sample.logits_processor.builtin import (
        BadWordsLogitsProcessor,
        LogitBiasLogitsProcessor,
        MinTokensLogitsProcessor,
        PenaltiesLogitsProcessor,
    )
    from nanovllm.sample.guided import GuidedDecodingLogitsProcessor

    return [
        PenaltiesLogitsProcessor(),
        BadWordsLogitsProcessor(),
        LogitBiasLogitsProcessor(),
        MinTokensLogitsProcessor(),
        GuidedDecodingLogitsProcessor(),
    ]
