"""
后端无关的通用注意力元数据。

对齐 vLLM V1 的 CommonAttentionMetadata：由 ModelRunner.prepare_inputs 每步构造一次，
各 AttentionMetadataBuilder 据此产出后端专属元数据。

nano 当前 FlashAttn / SDPA 两后端共享同一组字段，故沿用 utils.context.AttentionMetadata
的定义，此处仅给出对齐 V1 命名的别名（CommonAttentionMetadata）。
"""
from nanovllm.utils.context import AttentionMetadata

# V1 对齐命名别名。两者为同一 dataclass，便于阅读与未来扩展区分"通用/后端专属"。
CommonAttentionMetadata = AttentionMetadata

__all__ = ["CommonAttentionMetadata", "AttentionMetadata"]
