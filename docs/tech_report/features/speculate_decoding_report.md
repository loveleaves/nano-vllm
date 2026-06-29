# Speculative Decoding总览

## 一、什么是 Speculative Decoding

Speculative Decoding 是一种无损技术，通过使用更小、更快的草稿模型提前提议 Token，再由较大的基础模型统一验证，从而降低延迟、不影响输出质量。草稿模型智能地提前草拟多个 Token，基础模型以单次前向传播完成验证。每个被接受的 Token 都保证来自与单独使用主模型相同的分布。

典型的实现流程是：草稿模型在目标大模型生成一个 Token 的时间内提议 5-8 个候选续写；目标模型再以并行方式验证所有候选 Token，从而在不改变输出质量的前提下实现 2-3× 的加速。

## 二、vLLM 支持的技术路线

vLLM 支持多种投机解码方法：基于模型的方法（如 EAGLE、MTP、草稿模型、PARD 和 MLP）提供最佳延迟降低效果；而更简单的方法（如 n-gram 和 Suffix Decoding）在不增加峰值流量工作负载的情况下提供适度加速。

### 技术路线
![vllm_speculative_decoding_landscape](../imgs/vllm_speculative_decoding_landscape.png)

### EAGLE 系列（当前 SOTA）

EAGLE 的关键洞察是：在特征层进行自回归比在 Token 层更为简单，草稿模型复用目标模型的 Embedding 层和 LM Head，中间插入单层可训练 Transformer。相比之下，Medusa 的草稿准确率约 0.6，EAGLE 可达约 0.8。

EAGLE-2 在此基础上引入上下文感知的动态草稿树；EAGLE-3 则进一步抛弃特征预测，改用直接 Token 预测并融合多层特征，通过训练时测试（Train-time Testing）提升多步预测质量。

EAGLE-3 草稿模型从验证模型的三个层中获取隐状态作为输入，捕获验证模型的潜在特征；训练时测试模拟多步草稿采样过程，确保模型不仅学习预测第一个 Token，也能预测后续 Token。

**P-EAGLE（2026年最新）**：EAGLE 的自回归草稿存在隐性瓶颈——越多推测 Token，草稿模型需要的串行前向传播越多，最终会侵蚀收益。P-EAGLE 通过单次前向传播生成全部 K 个草稿 Token，在 NVIDIA B200 上相比 EAGLE-3 额外提速 1.69×，已在 vLLM v0.16.0 中集成，预训练检查点已在 HuggingFace 发布（GPT-OSS 120B/20B、Qwen3-Coder 30B）。

### N-Gram 与 Suffix Decoding（零开销轻量方法）

与 N-Gram 相比，Suffix Decoding 不仅可以对 Prompt 和历史生成内容进行模式匹配，还使用频次统计提议最可能的续写，并为每个请求在每次迭代时自适应地指定推测 Token 数量以获得更好的接受率。

### 动态投机解码

vLLM 的动态投机解码根据实时负载动态决策是否启用投机解码，目标是在中低 QPS 的内存瓶颈工作负载下降低逐 Token 延迟。这解决了高 QPS 时验证开销反而拖累吞吐的核心生产痛点。

## 三、Speculators 生态

Speculators 与 vLLM 的紧密集成将投机解码从研究技术转变为生产就绪特性，支持跨多种模型架构和配置的无缝部署。通过 Speculators 训练的 DFlash 模型可无缝在 vLLM 中运行。

Speculators 最新发布了 Gemma 4 31B-it 的 Speculator 检查点（包含 DFlash 和 EAGLE-3 两种），为 Gemma 4 提供了生产级别的投机解码支持；DFlash 训练算法新增了带锚点块的辅助隐状态支持。

## 四、无损保证

vLLM 的无损保证分三个层次：理论无损（投机解码采样在硬件数值精度限制内理论上无损）、算法无损（vLLM 的拒绝采样器实现经过算法验证）以及端到端无损（完整推理链路通过测试验证）。

## 五、关键结论

从技术路线演进来看，投机解码的发展逻辑十分清晰：

**草稿质量**从 MLP 并行头（Medusa，接受率 ~0.6）→ 特征级自回归（EAGLE，~0.8）→ 多层隐状态 + 训练测试（EAGLE-3，更高）持续提升；**草稿速度**从串行自回归（EAGLE 系列）→ 单次前向并行（P-EAGLE）大幅跃升；**工程成熟度**方面，2025-2026 年间统一训练库、动态开关、公开检查点三管齐下，使投机解码从研究工具升级为生产标准。

选型建议：通用对话低中 QPS 场景首选 EAGLE-3；DeepSeek 系列选 MTP；RAG/代码补全选 N-Gram 或 Suffix；高并发生产环境必须开启 Dynamic Speculative Decoding。完整的 Markdown 技术报告已导出，包含所有章节、代码示例和参考文献。如需进一步深入某一具体方法（如 EAGLE-3 训练流程、Suffix Decoding 自适应策略，或批处理下的性能调优），可以继续探讨。

---

# Speculative Decoding 详细技术报告

> 基于 vLLM 官方文档及最新研究成果的综合技术分析
> 报告日期：2026年6月

## 1. 概述

### 1.1 背景与动机

大语言模型（LLM）推理的核心瓶颈在于**自回归解码**：每一步只能生成一个 Token，每步都需要完整的前向传播，即便 GPU 拥有远超单 Token 生成所需的算力。这种串行解码模式导致推理延迟高、GPU 利用率低。

Speculative Decoding（投机解码/推测解码）是解决上述问题的核心技术之一。其基本原理是：使用一个轻量级"草稿模型"（Draft Model）提前预测多个 Token，再由大模型（Verifier/Target Model）通过一次前向传播并行验证这些 Token 的正确性。被接受的 Token 与大模型直接生成的 Token 分布一致，因此该方法**理论上无损**（Lossless）。

典型加速效果可达 **2-3× 甚至更高**，具体取决于草稿接受率与验证开销。

### 1.2 vLLM 的定位

vLLM 是当前最主流的开源 LLM 推理框架，其对 Speculative Decoding 的支持已从单一草稿模型扩展为多技术路线并存、生产级就绪的完整体系。

根据官方文档，vLLM 支持以下方法：

| 方法类别 | 代表方法 | 延迟收益 | 特点 |
|----------|----------|----------|------|
| 基于模型（高性能） | EAGLE / EAGLE-3、MTP、PARD、MLP、Draft Model | 最高 | 需要专用草稿模型/权重 |
| 无模型（轻量） | N-Gram、Suffix Decoding | 中等 | 零额外训练，峰值流量友好 |
| 动态调节 | Dynamic Speculative Decoding | 自适应 | 根据负载动态开关 |

## 2. 核心原理

### 2.1 拒绝采样机制（Rejection Sampling）

Speculative Decoding 的理论基础是**拒绝采样**：

1. 草稿模型自回归生成 K 个候选 Token（draft tokens）
2. 目标大模型以一次前向传播并行计算所有候选 Token 的概率
3. 按照拒绝采样算法逐 Token 接受/拒绝：
   - 若接受概率 ≥ 1，直接接受
   - 否则以一定概率拒绝，并从修正后的分布中重新采样
4. 在第一个被拒绝 Token 处截断，目标模型从此处接续生成

该机制保证输出分布与目标模型完全一致，是"无损"的严格数学保证。

### 2.2 关键度量

- **Token 接受率（Acceptance Rate / α）**：衡量草稿质量，越高越好
- **推测长度（K）**：每轮投机的 Token 数，需与接受率和验证开销平衡
- **加速比（Speedup）**：与基线的 TTFT、TPOT、吞吐量比较


## 3. 主要技术路线详解

### 3.1 EAGLE 系列（当前 SOTA）

EAGLE（Extrapolation Algorithm for Greater Language-model Efficiency）是目前学术和工程领域公认的最优方案，经过三代演进：

**EAGLE（原版）**：
- 在特征层（Feature Level）进行自回归，而非 Token 层
- 草稿模型复用目标模型的 Embedding 层和 LM Head，中间插入单层可训练 Transformer
- 关键洞察：特征层的自回归比 Token 层更简单，预测准确率更高（~0.8 vs Medusa 的 ~0.6）
- 使用静态 Draft Tree 进行 Tree-based 并行解码

**EAGLE-2**：
- 引入**动态草稿树（Dynamic Draft Tree）**，基于上下文自适应调整树结构
- 接受率进一步提升

**EAGLE-3（2025年，当前 SOTA）**：
- 抛弃特征预测，改为直接 Token 预测结合多层特征融合（Multi-layer Feature Fusion）
- 草稿模型输入来自目标模型三个中间层的隐状态（Hidden States），捕获更丰富的上下文信息
- 引入 **Train-time Testing**：训练时模拟多步草稿生成过程，使用稀疏 Attention Mask（通过 FlexAttention 实现），让模型学会预测不止第一个 Token，也包括后续 Token
- vLLM 通过 `speculators` 库支持 EAGLE-3 的端到端训练与部署

**P-EAGLE（2026年，vLLM v0.16.0+）**：
- EAGLE 的草稿生成仍是自回归的，K 步投机需要 K 次串行前向传播，形成隐性瓶颈
- P-EAGLE 将所有 K 个草稿 Token 以**单次前向传播**并行生成
- 在 NVIDIA B200 上实测比 EAGLE-3 额外提速 1.69×
- 预训练检查点已在 HuggingFace 提供（GPT-OSS 120B/20B、Qwen3-Coder 30B）

**启用方式（以 EAGLE 为例）：**
```python
llm = LLM(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    speculative_config={
        "method": "eagle",
        "model": "yuhuili/EAGLE-LLaMA3-Instruct-8B",
        "num_speculative_tokens": 5,
    }
)
```

**CUDA Graph 注意事项**：启用 EAGLE 后，验证阶段每步处理 (1+K) 个 Token。`cudagraph_capture_sizes` 须为 n×(K+1) 的整数倍。例如 K=4 且批次 1-4 时，`cudagraph_capture_sizes=[5,10,15,20]`。

### 3.2 MTP（Multi-Token Prediction）

MTP 是 DeepSeek 等模型原生引入的多 Token 预测机制，利用模型自身的 MTP 权重层并行预测多个 Token，无需独立草稿模型，天然适配 DeepSeek 系列。

限制：受制于 DeepSeek 仅暴露单层 MTP 权重，当 `num_speculative_tokens > 1`（尤其 ≥ 3）时，精度和性能无法有效保证。

### 3.3 MLP 投机器（MLP Speculator）

基于 MLP 的轻量草稿头，与 Medusa 类似，在目标模型次顶层特征上并行预测多个 Token。相比 EAGLE，精度略低（接受率约 0.6），但推理开销极小，适合对延迟要求高但可接受一定精度损失的场景。

vLLM 与 Red Hat 的 `speculators` 库集成，支持包括 MLP 在内的多种草稿头的统一训练与管理。

### 3.4 PARD（Parallel Auto-Regressive Decoding）

PARD 是一种并行自回归草稿方法，与 MLP/EAGLE 类思路相关，提供中等强度的加速。具体实现细节由 vLLM 社区持续完善。

### 3.5 独立草稿模型（Draft Model）

最传统的方案：使用一个独立的小模型（通常是目标模型的同系列小版本）作为草稿模型。优点是方法通用，无需专属训练；缺点是需要额外显存加载草稿模型权重。

```python
llm = LLM(
    model="facebook/opt-6.7b",
    speculative_config={
        "model": "facebook/opt-125m",
        "num_speculative_tokens": 5,
    }
)
```

vLLM 还支持**并行草稿模型（Parallel Draft Model）**，通过并行化草稿生成阶段进一步提升效率。

### 3.6 N-Gram 投机

无需任何模型：在当前 Prompt 中查找最近 N 个已生成 Token 的模式，将历史匹配的后续 Token 作为草稿提议。

特点：
- 零训练成本，零额外显存
- 对 RAG、代码补全、文档续写等场景效果显著
- 峰值流量期间不增加额外负载，适合高 QPS 场景

```python
speculative_config={
    "method": "ngram",
    "num_speculative_tokens": 5,
    "ngram_prompt_lookup_max": 4,
}
```

### 3.7 Suffix Decoding

Suffix Decoding 是 N-Gram 的升级版，在 vLLM 中被视为独立方法：

- 匹配范围：Prompt + 历史生成内容（而非仅当前 Prompt）
- 使用频次统计提议最可能的续写
- **自适应投机长度**：每次迭代动态决定当前 Request 的投机 Token 数，以最大化接受率
- 更适合长对话、反复出现相似模式的场景

### 3.8 动态投机解码（Dynamic Speculative Decoding）

vLLM 内置的自适应开关机制，根据实时 QPS 和系统负载动态决定是否启用投机解码：

- 低 QPS / Memory-Bound 时：开启投机解码，最大化降低延迟
- 高 QPS / Compute-Bound 时：关闭投机解码，避免草稿验证开销拖累吞吐量

这是 vLLM 面向生产环境的重要工程创新，解决了投机解码"在高负载时反而有害"的核心痛点。


## 4. 工程实现与生态

### 4.1 Speculators 库

vLLM 与 Red Hat 联合推出 `vllm-project/speculators`，是一个面向生产的统一草稿模型训练与管理库，支持：

- EAGLE-3、DFlash 等主流算法的端到端训练
- 通过隐状态提取（Hidden State Extraction）生成训练数据
- 训练好的模型可无缝导入 vLLM 部署
- Gemma 4、Qwen3 等主流模型系列已有对应的 Speculator 检查点

**DFlash** 是 Speculators 最新支持的算法，使用多层锚点块（Anchored-Block Drafting）并利用验证模型多层辅助隐状态，进一步提升草稿质量。

### 4.2 无损保证体系

vLLM 从三个层次保证投机解码的无损性：

1. **理论无损（Theoretical Losslessness）**：拒绝采样算法严格保证输出分布与目标模型一致，误差仅来自硬件浮点精度
2. **算法无损（Algorithmic Losslessness）**：vLLM 实现的拒绝采样器经过收敛性测试验证
3. **端到端无损（End-to-end Losslessness）**：完整推理链路测试，确保批次组合不影响输出一致性

### 4.3 方法选择指南

官方提供了定性对照表，以下为综合建议：

| 场景 | 推荐方法 | 理由 |
|------|----------|------|
| 通用对话（低/中 QPS） | EAGLE-3 | 最高接受率，最优延迟降低 |
| DeepSeek 系列模型 | MTP | 原生支持，无需额外模型 |
| RAG / 代码补全 | N-Gram / Suffix Decoding | 输出与 Prompt 高度重叠，接受率高 |
| 高 QPS 生产环境 | Dynamic Speculative Decoding | 自适应开关，保护吞吐 |
| 资源受限 / 通用加速 | Draft Model（同系列小模型） | 简单通用 |
| 超大规模模型（如 120B） | P-EAGLE | 单次前向传播并行草稿 |


## 5. 技术趋势分析

### 5.1 草稿质量的持续提升

从 Medusa（MLP 头，接受率 ~0.6）→ EAGLE（特征级自回归，~0.8）→ EAGLE-3（多层隐状态 + Train-time Testing，更高）→ P-EAGLE（并行草稿，进一步降低草稿开销），核心驱动力是提升接受率与降低草稿延迟的双重目标。

### 5.2 从单点到系统工程

早期研究关注草稿模型本身的质量，现代工作更多关注系统级优化：

- **批处理下的投机解码**：如何在大批次下保持加速（不同请求接受率不同，批处理对齐困难）
- **动态草稿长度**：自适应调整 K 值
- **KV Cache 管理**：草稿 Token 的 KV Cache 如何高效分配与回收

### 5.3 生产级工具链成熟

2025-2026 年间，投机解码从"研究特性"升级为"生产标准"的关键转变：

- vLLM、TensorRT-LLM 均原生支持
- 统一训练库（Speculators）降低草稿模型准备门槛
- 动态开关机制解决高 QPS 下的兼容性问题
- 主流模型（Llama 3、Qwen3、Gemma 4）均有公开的 Speculator 检查点

### 5.4 多模态与长上下文扩展

最新研究已开始将投机解码扩展到视觉语言模型（VLM）和长上下文场景（如 LongSpec、TriForce 的层级投机解码），是下一步技术演进的重要方向。


## 6. 局限性与适用条件

| 维度 | 说明 |
|------|------|
| 最优场景 | 低/中 QPS、Memory-Bound 工作负载 |
| 不适场景 | 高 QPS、Compute-Bound（验证开销反而降低吞吐） |
| 采样约束 | 贪心解码（Greedy）效果最好；Temperature > 0 时接受率下降 |
| 草稿模型依赖 | EAGLE 类方法需要特定架构的草稿模型，跨架构迁移有成本 |
| 浮点精度 | 理论无损，但硬件浮点误差可能导致极小的输出分布差异 |
| 内存开销 | 草稿模型需要额外显存；N-Gram / Suffix 方法无此问题 |


## 7. 参考文献

- vLLM 官方文档：https://docs.vllm.ai/en/latest/features/speculative_decoding/
- Speculators 库：https://github.com/vllm-project/speculators
- Li et al. (2024). *EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty*. ICML 2024.
- Li et al. (2024). *EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees*. EMNLP 2024.
- Li et al. (2025). *EAGLE-3: Scaling Up Inference Acceleration via Training-Time Test*. NeurIPS 2025.
- vLLM Blog (2025). *Speculators v0.3.0: Speculative Decoding Training Support*.
- vLLM Blog (2026). *P-EAGLE: Faster LLM Inference with Parallel Speculative Decoding*.
- Chen et al. (2023). *Accelerating Large Language Model Decoding with Speculative Sampling*.
- Leviathan et al. (2023). *Fast Inference from Transformers via Speculative Decoding*. ICML 2023.