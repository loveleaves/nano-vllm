# Nano-vLLM 文档索引

本目录包含四类文档：**自研实现的设计与测试文档**、**工程优化专题调研报告**、**模型适配专题**、以及学习上游实现时的**源码阅读笔记**。

## 一、自研实现：设计与测试文档

> 理解本仓库 `my_nano` / 各 phase 分支自研实现的首选入口。

| 文档 | 内容 |
|------|------|
| [architecture.md](architecture.md) | **架构设计文档**：项目概述、核心设计目标、分层架构与关键设计决策（从这里开始） |
| [detailed_design.md](detailed_design.md) | **详细设计文档**：逐模块实现细节、设计原则与已知限制 |
| [phase1_test_design.md](phase1_test_design.md) | Phase 1 测试设计：Config / Sequence / BlockManager / Scheduler 的测试范围与策略 |
| [phase2_test_design.md](phase2_test_design.md) | Phase 2 测试设计：RMSNorm / RoPE / Attention / Linear / Sampler / Qwen3 的测试范围与策略 |

## 二、工程优化专题调研报告（nano-vllm vs vLLM）

> 针对 Phase 4 各项工程优化，与开源 vLLM 的设计/实现逐项对比。

| 文档 | 内容 |
|------|------|
| [prefix_caching-调研报告-20260614.md](prefix_caching-调研报告-20260614.md) | 前缀缓存：链式块哈希、引用计数共享的设计与实现对比 |
| [chunked_prefill-调研报告-20260614.md](chunked_prefill-调研报告-20260614.md) | Chunked Prefill：长 prompt 分块处理的设计与实现对比 |
| [tensor_parallelism-调研报告-20260614.md](tensor_parallelism-调研报告-20260614.md) | 张量并行：Column/Row/QKV 并行与多进程通信的设计与实现对比 |
| [cuda_graph-调研报告-20260614.md](cuda_graph-调研报告-20260614.md) | CUDA Graph：decode 阶段按 batch size 录制复放的设计与实现对比 |
| [flashattention_triton-调研报告-20260614.md](flashattention_triton-调研报告-20260614.md) | FlashAttention 与 Triton：prefill/decode 注意力与 KV 写入的设计与实现对比 |

## 三、模型适配专题（Qwen3.5 dense / MoE）

> 在 nano-vllm 上适配新模型架构（Qwen3.5 / Qwen3-Next 混合架构）的设计、调研与测试文档，来源分支 `phase3_model_adapt`。完整索引见 [model_adaptation/README.md](model_adaptation/README.md)。

| 文档 | 内容 |
|------|------|
| [model_adaptation/vllm_model_adapt.md](model_adaptation/vllm_model_adapt.md) | vLLM 模型适配实战指南：在已支持同系模型的引擎上新增一个模型的通用流程 |
| [model_adaptation/bug-prefill-cross-seq-attention.md](model_adaptation/bug-prefill-cross-seq-attention.md) | Bug 定位记录：Prefill 跨序列注意力污染的现象、根因与修复 |
| [model_adaptation/qwen35_2b_adaptation.md](model_adaptation/qwen35_2b_adaptation.md) | Qwen3.5-2B 早期设计草稿：混合架构与线性/全注意力层参数 |
| [model_adaptation/qwen35_adaptation/](model_adaptation/qwen35_adaptation/) | **Qwen3.5-2B（dense）**：research（调研）/ design（详细设计）/ testing（测试）三件套 |
| [model_adaptation/qwen35_moe_adaptation/](model_adaptation/qwen35_moe_adaptation/) | **Qwen3.5-35B-A3B（MoE）**：research / design / testing 三件套，含 GDN nv>nk 修复与 8GB 减层适配 |

## 四、上游源码阅读笔记

> 学习上游 [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) 实现时整理的笔记。

| 文档 | 内容 |
|------|------|
| [05_setup.md](05_setup.md) | **uv 环境搭建、flash-attn 编译、运行 example.py** |
| [01_architecture.md](01_architecture.md) | 项目概述、目录结构、请求生命周期、Qwen3 模型树、多进程架构 |
| [02_core_concepts.md](02_core_concepts.md) | PagedAttention、前缀缓存、Chunked Prefill、张量并行、CUDA Graph、FlashAttention、全局 Context、Gumbel 采样、Fused Add-Norm |
| [03_data_flow.md](03_data_flow.md) | Prefill/Decode 完整数据流（逐步展开）、Scheduler 调度策略伪代码 |
| [04_implementation_guide.md](04_implementation_guide.md) | 权重加载机制、性能优化对照表、从零实现路线图（25 步）、可扩展点（8 个）、调试技巧 |
| [06_sleep_mode_design.md](06_sleep_mode_design.md) | Sleep Mode 设计分析：vLLM CUDA VMM 机制解读、nano-vllm 实现方案、三级睡眠、分步唤醒、RLHF 用法 |
| [07_triton_custom_ops.md](07_triton_custom_ops.md) | Triton 定制算子开发指南：编程模型、现有 kernel 解析、SiluAndMul/RMSNorm/Add-RMSNorm 三个示例 |
| [dispatch.md](dispatch.md) | 执行模式与调度机制 |
| [op_adapt.md](op_adapt.md) | 大模型算子适配 |
