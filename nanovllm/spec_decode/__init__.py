"""
投机解码（对齐 vLLM V1 `v1/spec_decode/`）。

机制：草案器（proposer）先廉价提出 k 个候选 token，目标模型一次前向**并行验证**这 k 个
位置，拒绝采样（rejection sampler）接受最长正确前缀 + 1 个修正/奖励 token——一步前向产出
多 token，降低自回归延迟。

本包实现两个**与 GPU 无关、纯 CPU 可测**的核心组件 + 编排：
  - NgramProposer：无需草案模型，用历史 n-gram 匹配提议续写（最轻量的 proposer）。
  - RejectionSampler（见 sample/rejection_sampler.py）：贪心验证，接受最长匹配前缀 + 修正 token。
  - SpeculativeDecoder：propose → score（目标模型）→ verify 的编排。

集成边界：把 SpeculativeDecoder.step 的 score_fn 接到"目标模型对 k+1 个位置的并行前向 +
KV 写入/回滚 + 调度器按接受数推进"即为完整 GPU 通路——这部分侵入 Scheduler/InputBatch/
KV cache，作为 UniProc-only 的集成点单列（见 docs/arch_spec_decode/design.md），本轮交付
算法与组件。
"""
from nanovllm.spec_decode.ngram_proposer import NgramProposer
from nanovllm.spec_decode.spec_decoder import SpeculativeDecoder

__all__ = ["NgramProposer", "SpeculativeDecoder"]
