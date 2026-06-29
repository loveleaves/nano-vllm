# 模型适配专题（Model Adaptation）

> 在 nano-vllm 上适配新模型架构的设计、调研与测试文档。
> 核心案例：**Qwen3.5（内部架构名 Qwen3-Next）** —— GatedDeltaNet 线性注意力 + 全注意力的混合架构，含 dense 与 MoE 两个变体。
> 来源分支：`phase3_model_adapt`（自 `phase3` 分出）。

## 通用方法论

| 文档 | 内容 |
|------|------|
| [vllm_model_adapt.md](vllm_model_adapt.md) | **大模型适配推理引擎实战指南**（以 vLLM 为参照）：从架构分析、代码实现、权重映射、并行、混合算子到分级验证与 PR 的通用流程，Qwen3.5-35B-A3B 仅作运行示例 |
| [bug-prefill-cross-seq-attention.md](bug-prefill-cross-seq-attention.md) | **Bug 定位记录**：Prefill 跨序列注意力污染（第二条 prompt 输出乱码）的现象、根因与修复 |

## Qwen3.5-2B（dense，混合 GDN + 全注意力）

| 文档 | 内容 |
|------|------|
| [qwen35_2b_adaptation.md](qwen35_2b_adaptation.md) | 早期设计草稿：模型架构概述、线性/全注意力层参数（dense 适配的起点蓝本） |
| [qwen35_adaptation/research.md](qwen35_adaptation/research.md) | 技术调研报告：对照 vLLM v0.21.0 `qwen3_next.py` 交叉验证草稿，指出需补充/修订之处 |
| [qwen35_adaptation/design.md](qwen35_adaptation/design.md) | 详细设计文档：混合模型实现、仅对全注意力层分配 KV cache、线性注意力状态生命周期管理 |
| [qwen35_adaptation/testing.md](qwen35_adaptation/testing.md) | 测试文档：测试环境、CPU 单元/集成测试与 GPU 集成测试用例清单 |

## Qwen3.5-35B-A3B（MoE 变体）

| 文档 | 内容 |
|------|------|
| [qwen35_moe_adaptation/research.md](qwen35_moe_adaptation/research.md) | 技术调研报告：MoE 块（256 专家 top-8 + 1 共享专家）与 GDN `nv>nk` 维度差异的上游对照 |
| [qwen35_moe_adaptation/design.md](qwen35_moe_adaptation/design.md) | 详细设计文档：修复 GDN nk/nv 维度 Bug、实现 MoE FFN、减层 config 适配 8GB 显存 |
| [qwen35_moe_adaptation/testing.md](qwen35_moe_adaptation/testing.md) | 测试文档：8GB 显存 + 减层权重下的单元/集成/回归测试结果 |
