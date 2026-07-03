# 主流大模型 Attention 技术路线调研报告

> 版本：2026年7月　|　范围：从标准 Multi-Head Attention 出发，梳理当前主流大语言模型在**显存/带宽优化**、**计算/IO 优化**、**稀疏化**、**线性化**、**分布式长上下文**五个方向上的代表性技术路线，结合原始论文与工程博客进行原理解析与横向对比。

---

## 目录

1. [问题背景：为什么 Attention 需要被重新设计](#1-问题背景为什么-attention-需要被重新设计)
2. [技术全景图](#2-技术全景图)
3. [路线一：KV Cache 压缩 —— MHA → MQA → GQA → MLA](#3-路线一kv-cache-压缩--mha--mqa--gqa--mla)
4. [路线二：计算与 IO 优化 —— FlashAttention 系列 与 PagedAttention](#4-路线二计算与-io-优化--flashattention-系列-与-pagedattention)
5. [路线三：稀疏注意力 —— 从静态窗口到动态选择](#5-路线三稀疏注意力--从静态窗口到动态选择)
6. [路线四：线性注意力与状态空间模型（SSM）——从二次到线性](#6-路线四线性注意力与状态空间模型ssm从二次到线性)
7. [路线五：分布式长上下文 —— Ring Attention](#7-路线五分布式长上下文--ring-attention)
8. [主流开源/商业模型 Attention 选型一览](#8-主流开源商业模型-attention-选型一览)
9. [技术对比与选型建议](#9-技术对比与选型建议)
10. [趋势与展望](#10-趋势与展望)
11. [参考文献与资料来源](#11-参考文献与资料来源)

---

## 1. 问题背景：为什么 Attention 需要被重新设计

2017 年 *Attention Is All You Need* 提出的标准 Multi-Head Attention（MHA）为每个头独立维护 Query/Key/Value 投影，计算与显存复杂度均随序列长度呈 **O(n²)** 增长。在大模型时代，这一设计暴露出两类互相独立又彼此叠加的瓶颈：

- **训练/预填充（prefill）阶段**：QKᵀ 相似度矩阵是 O(n²) 的，且传统实现需要把这个大矩阵反复读写到 GPU 高带宽内存（HBM），导致算子是**内存带宽受限（memory-bound）**而非算力受限。
- **自回归解码（decode）阶段**：为避免重复计算，系统会缓存历史 token 的 K、V（即 **KV Cache**）。随着上下文变长、并发请求增多，KV Cache 的显存占用甚至可能超过模型参数本身，成为长上下文推理的头号瓶颈。

围绕这两类瓶颈，业界形成了五条相对独立又常常组合使用的技术路线，如下图所示。

## 2. 技术全景图

```
                     ┌─────────────────────────────────────────┐
                     │        标准 Multi-Head Attention (MHA)    │
                     └─────────────────────────────────────────┘
                                       │
        ┌──────────────┬──────────────┼──────────────┬──────────────────┐
        ▼              ▼              ▼              ▼                  ▼
  【KV Cache压缩】  【IO/计算优化】  【稀疏化】     【线性化/SSM】     【分布式并行】
  MQA→GQA→MLA     FlashAttention   滑动窗口/Sink    Linear Attention   Ring Attention
                   1/2/3           NSA / MoBA /DSA  Lightning/Mamba    DeepSpeed Ulysses
                   PagedAttention  长文本Top-K选择   混合(Hybrid)架构
```

五条路线解决的问题并不相同：KV Cache 压缩关注"缓存该存多少"；IO/计算优化关注"同样的数学结果如何算得更快"；稀疏化和线性化关注"能不能不用算全部 n²"；分布式并行则关注"单卡装不下的超长序列如何跨设备协作"。在实际的旗舰模型中，这些技术通常是**叠加组合**使用的（例如 DeepSeek-V3.2 = MLA + DSA 稀疏选择 + FlashAttention 内核；Kimi Linear = 线性注意力 + 少量全注意力层的混合架构）。

---

## 3. 路线一：KV Cache 压缩 —— MHA → MQA → GQA → MLA

这条路线的核心矛盾是：**推理阶段每个 token 生成时都要读取全部历史 K、V**，头数越多、每头维度越大，缓存和带宽开销越大；但头的数量和维度又直接决定模型的表达能力。四种方案依次是对"共享粒度"的不同取舍。

### 3.1 Multi-Head Attention（MHA，基线）

标准做法：每个注意力头拥有独立的 Q、K、V 投影矩阵。表达能力最强，但 KV Cache 大小与头数成正比，解码阶段带宽压力最大。

### 3.2 Multi-Query Attention（MQA）

提出者：Noam Shazeer，2019。核心思想是**让所有 Query 头共享同一组 Key/Value 头**，只保留多个 Query 头。这样 KV Cache 体积直接压缩到 1/n_heads，大幅降低解码阶段的显存带宽占用；代价是模型质量有所下降，且从头训练 MQA 模型可能出现训练不稳定的问题。

### 3.3 Grouped-Query Attention（GQA）

论文：*GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*（Ainslie et al., 2023）。GQA 是 MHA 与 MQA 之间的**插值方案**：把 Query 头分成 g 组，每组内的 Query 头共享一套 K/V 投影。当 g 等于头总数时退化为 MHA，当 g=1 时退化为 MQA。论文同时给出了一种"升级训练（uptraining）"配方：可以用约 5% 的原始预训练算力，把已经训练好的 MHA checkpoint 转换为 GQA 模型，而不必从零训练。GQA 因为改动小、效果稳、工程实现简单，很快被 Llama 2/3、Mistral、Qwen、Gemma 等主流开源模型采纳，成为 2023–2024 年事实上的标准配置。

### 3.4 Multi-head Latent Attention（MLA）

论文：*DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model*（DeepSeek-AI，2024）。MLA 与 GQA 的思路完全不同：GQA 是"减少 KV 头的数量"，MLA 则是**对 KV 做低秩联合压缩**——将 Key 和 Value 一起投影到一个低维的"潜在向量"（latent vector）并只缓存这个压缩后的潜在表示，推理时再用一个上投影矩阵将其解压还原为各头独立的 K、V。这样既保留了"每个 Query 头都能看到独立、充分表达的 K/V"这一 MHA 式的表达能力，又把缓存体积压缩到接近 MQA 的水平。

DeepSeek-V2 论文的消融实验显示：GQA 在建模效果上通常弱于 MHA，而 MLA 在同等甚至更低的 KV Cache 开销下，效果可以与 MHA 持平甚至略微超过。此外还有理论工作证明，GQA 总能被 MLA 等价表示（缓存开销不变），但反过来不成立，说明 MLA 是更"通用"的压缩形式。业界普遍认为，MLA 之所以在大模型和长上下文场景下更具吸引力，是因为它能在相同压缩率下保持更好的建模质量；不过也有从业者指出，MLA 的优势更多体现在较大规模模型上，中小模型（<100B 量级）用 GQA 往往调参更容易。

需要注意的是，MLA 的实现和服务化比 GQA 更复杂：需要额外的低秩上/下投影矩阵，并涉及"权重吸收（weight absorption）"等推理期特殊优化技巧，对旋转位置编码（RoPE）部分还需要"解耦 RoPE"设计，才能保证压缩的一致性。MLA 已经被 DeepSeek-V2/V3/R1、Kimi K2、GLM 系列等模型采用。

**四种方案对比示意：**

| 方案 | KV 头数 | KV Cache 大小 | 建模质量 | 训练/服务复杂度 |
|---|---|---|---|---|
| MHA | = Query 头数 | 最大 | 最好（基线） | 最简单 |
| MQA | 1 | 最小 | 有明显损失 | 简单，但训练不稳定 |
| GQA | 1 < g < 头数 | 中等，可调 | 接近 MHA，略逊 | 简单，工程成熟 |
| MLA | 低秩潜在向量 | 接近 MQA 水平 | 持平甚至优于 MHA | 较复杂，需权重吸收/RoPE 解耦 |

---

## 4. 路线二：计算与 IO 优化 —— FlashAttention 系列 与 PagedAttention

这条路线不改变注意力的数学定义（依然是精确 softmax attention），而是重新设计**算子如何在 GPU 内存层级之间搬运数据**，属于纯粹的系统/工程优化。

### 4.1 FlashAttention（Dao et al., NeurIPS 2022）

论文：*FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*。核心洞察是：标准注意力实现的瓶颈并非浮点运算量，而是把 O(n²) 大小的注意力矩阵在 GPU 高带宽内存（HBM）与片上高速缓存（SRAM）之间反复读写。FlashAttention 提出**分块（tiling）+ 核融合（kernel fusion）**：将 Q、K、V 切成小块，在 SRAM 内完成一个块的注意力计算与在线 softmax（online softmax，即维护运行时的最大值与归一化项，避免物化完整的注意力矩阵），全程不把中间的 n×n 矩阵写回 HBM。该方法是**数学上精确**的（非近似），却能带来数倍的端到端加速，并让训练更长序列成为可能。

### 4.2 FlashAttention-2（Dao, ICLR 2024）

在 FlashAttention 基础上重新设计了并行策略：增加了对序列长度维度的并行切分，并优化前向传播内层循环对 K/V 块的处理顺序，从而提升 GPU 占用率（occupancy）与整体工作划分效率，在现代 GPU 上进一步提高吞吐。

### 4.3 FlashAttention-3（Shah, Bikshandi, Dao 等, NeurIPS 2024）

针对 Hopper 架构等新一代 GPU 的异步执行能力与低精度（FP8 等）计算单元做了专门优化：利用**异步（asynchrony）**特性使 GPU 的矩阵乘（GEMM）与非矩阵乘操作（如 softmax）能够并行重叠，并结合低精度数值格式进一步提速。论文指出，即使是 FlashAttention-2 在新硬件上相对于高度优化的 GEMM kernel 利用率也明显偏低（如 35% 左右），这正是 FlashAttention-3 要解决的问题。

### 4.4 灵活注意力内核：FlexAttention 等

FlashAttention 系列内核针对特定的注意力变体（因果掩码、滑动窗口等）手工优化，扩展到新的掩码/打分模式往往代价很高。PyTorch 团队提出的 *Flex Attention* 等工作试图提供一种可编程模型，让用户能以较小代价定义自定义的注意力模式（如任意掩码或分数修正）并仍获得接近手写内核的性能。

### 4.5 PagedAttention 与 vLLM（Kwon et al., SOSP 2023）

论文：*Efficient Memory Management for Large Language Model Serving with PagedAttention*。这项工作瞄准的是服务/推理系统层面的 **KV Cache 内存管理**问题：传统推理系统为每个请求预分配一段"连续"显存来存放 KV Cache，但由于实际生成长度难以预知，往往造成严重的内部碎片和外部碎片，论文统计显示真正用于存放有效 KV Cache 的显存占比有时低至 20% 左右。

PagedAttention 借鉴操作系统虚拟内存分页的思想：把 KV Cache 切分成固定大小的"块（block/page）"，允许同一请求乃至不同请求的 KV 块**离散地**存放在物理显存的任意位置，并通过一张"块表（block table）"维护逻辑块到物理块的映射；对于共享公共前缀（如相同 prompt 的并行采样、beam search 分支）的多个请求，还可以直接共享同一批物理块，实现写时复制（copy-on-write）。这一机制显著降低显存浪费与碎片，使同一硬件能够支撑更大的批量或更长的上下文，是当前 vLLM、SGLang 等主流推理框架的核心基础设施之一。

---

## 5. 路线三：稀疏注意力 —— 从静态窗口到动态选择

稀疏注意力路线的假设是：**并非每个 token 都需要关注全部历史 token**，只要挑出最相关的一小部分 K/V 参与计算，就能在几乎不损失效果的前提下把复杂度从 O(n²) 降到近似线性或对数级别。按照"选谁"的方式，可以分为静态规则型和动态学习型两类。

### 5.1 静态规则型：滑动窗口注意力 与 Attention Sink

**滑动窗口注意力（Sliding Window Attention, SWA）**：以 Mistral 7B（2023）为代表，每个 token 只关注前面固定窗口大小 W 内的历史 token。由于 Transformer 是多层堆叠的，信息仍可以通过"感受野累积"跨层传播——理论上经过 k 层后信息可以传播 k×W 的距离。Mistral 7B 用 W=4096 在 8K 上下文长度上实现了理论约 131K token 的等效感受野，同时结合 FlashAttention/xFormers 的适配实现了约 2 倍的速度提升。这种方法简单直接，但正如后续研究指出的，纯滑动窗口会直接丢弃更早的全局信息，长距离依赖能力有限。

**Attention Sink 与 StreamingLLM**（Xiao et al., MIT，ICLR 2024，论文：*Efficient Streaming Language Models with Attention Sinks*）：研究发现一个有趣现象——模型会把大量注意力权重"倾泻"到序列最初的几个 token 上，即便这些 token 在语义上并不重要，这一现象被称为 **attention sink**。如果直接用纯滑动窗口丢弃这些初始 token，模型性能会在序列超出窗口后迅速崩溃。StreamingLLM 的做法是：始终在 KV Cache 中保留最初的少量"sink token"（实验显示 4 个左右即可），再叠加一个滚动的滑动窗口，从而让模型能够以恒定的内存开销稳定处理数百万 token 级别的流式输入，无需重新微调。后续工作也表明，可以在预训练阶段显式引入专门的占位 sink token，进一步提升流式部署效果。

### 5.2 动态学习型：从"块检索"到"硬件对齐的原生稀疏"

静态规则（窗口、sink）依赖人工先验，无法自适应地找到"真正重要"的远距离 token。2025 年前后，DeepSeek 与月之暗面（Moonshot AI/Kimi）几乎同时发布了两条思路接近但独立完成的动态稀疏注意力方案：

**Native Sparse Attention（NSA）**——DeepSeek-AI，2025 年 2 月，论文：*Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention*。NSA 采用三路并行的分层策略：

1. **压缩（Compression）**：把连续的 token 块通过一个可学习的 MLP 压缩为一个粗粒度的表示，用于捕捉全局上下文；
2. **选择（Selection）**：基于压缩后的粗粒度相关性分数，动态挑选出最相关的若干细粒度 token 块参与精确计算，兼顾长距离检索精度；
3. **滑动窗口（Sliding Window）**：保留一个局部窗口以保证近邻上下文的精确建模。

三路结果最终融合输出。NSA 的另一个关键贡献是**硬件对齐**：基于 Triton 实现了专门的稀疏注意力 kernel，通过以组（而非单个头）为中心的数据加载、共享 KV 块的连续读取、循环调度优化等手段，最大化 Tensor Core 利用率，并且这种稀疏机制是**原生可训练**的——即从预训练阶段就可以使用，而不是只在推理阶段做事后剪枝，从而在预训练阶段本身就能获得加速，同时论文报告在多个基准上 NSA 与全注意力基线效果相当甚至更优。

**Mixture of Block Attention（MoBA）**——Moonshot AI（Kimi）联合清华大学等，2025 年 2 月，同一时期发布，论文：*Mixture of Block Attention for Long-Context LLMs*。MoBA 的核心思路是把**混合专家（MoE）**的路由思想搬到注意力机制上：将上下文切分为若干"块（block）"，用一个类似 MoE 门控网络（top-k gating，不需要额外训练参数）为每个 Query token 动态选出最相关的若干 KV 块参与注意力计算，同时始终保留当前 Query 所在块（对应最近邻上下文）。MoBA 的一个突出优势是**可以在全注意力与稀疏注意力之间无缝切换**，因而能较低成本地适配已有的全注意力预训练模型；据介绍，该架构已在 Kimi 平台实际部署验证一年以上，在处理百万级 token 长文本时相较全注意力可带来数倍到十余倍的速度提升。

NSA 与 MoBA 都属于"先粗筛/分块、再对入选块做精细注意力"的范式，但 NSA 更强调预训练期硬件级 kernel 的深度协同设计，MoBA 则更强调与现有全注意力架构的无缝兼容与工程可迁移性。

**DeepSeek Sparse Attention（DSA）**——DeepSeek-V3.2，2025 年 9 月起。DSA 是 NSA 思路在 DeepSeek 主力模型上的延续与简化，构建在 MLA 之上：先用一个轻量的"闪电索引器（lightning indexer）"为每个 Query 与历史所有 token 计算一个索引相关性分数，取 Top-K 得到该 Query 应该关注的 token 子集，再对这个子集执行"稀疏 MLA"计算，将注意力复杂度降低到近似 O(kL)（k 为选中 token 数，L 为序列长度），而非 O(L²)。DeepSeek 团队报告显示，DSA 显著提升长上下文效率的同时，与未引入稀疏注意力的前代模型（V3.1-Terminus）相比在多项基准与人类偏好评测上没有出现明显效果回退。

---

## 6. 路线四：线性注意力与状态空间模型（SSM）——从二次到线性

以上路线大多仍保留 softmax 注意力的基本形式，只是"少算一些"。另一条更激进的路线则试图从数学结构上把注意力**重写为线性（甚至常数状态）复杂度**的递归形式。

### 6.1 线性注意力的基本思路

线性注意力最早可追溯到 *Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention*（ICML 2020）：通过把 softmax(QKᵀ)V 中的相似度核函数替换为可分解的核（kernel trick），使得计算顺序可以调整为先算 KᵀV（"右乘"），再与 Q 相乘，从而将复杂度从与序列长度平方相关降低到与序列长度线性相关，并使模型在推理时可以等价地表示为一个类似 RNN 的常数大小状态的递归更新过程。

### 6.2 Lightning Attention 与 MiniMax 混合架构

MiniMax 于 2025 年 1 月发布的 MiniMax-01 系列（论文：*MiniMax-01: Scaling Foundation Models with Lightning Attention*）是较早大规模验证线性注意力可行性的主流模型之一。其 **Lightning Attention** 是基于 TransNormer 系列工作、面向 IO 效率专门优化的线性注意力实现，利用"右乘核技巧"将复杂度降至线性。但 MiniMax 团队在扩大模型规模的实验中发现，纯线性注意力在**长文本检索（retrieval，"大海捞针"类任务）**上存在明显短板。为此他们采用了**混合架构**：每 8 层中 7 层使用 Lightning Attention、1 层使用标准 Softmax Attention，用少量全注意力层弥补线性注意力在精确检索上的不足，同时保留长序列建模的整体效率优势。该架构结合 Varlen Ring Attention、线性序列并行优化（LASP+）等系统工程，实现了 400 万 token 级别的上下文窗口。后续的 MiniMax-M1（2025 年 6 月）在此基础上叠加大规模强化学习，成为全球首个开源权重的大规模"混合注意力"推理模型。

### 6.3 状态空间模型（SSM）：Mamba 及其混合架构

与线性注意力平行发展的另一条技术脉络是**状态空间模型（State Space Model, SSM）**，代表工作是 Mamba（Gu & Dao）。Mamba 引入**选择性（selective）**机制，让状态转移参数依赖于输入内容动态调整，在保持推理阶段常数级状态大小、训练阶段近似线性复杂度的同时，缓解了早期 SSM（如 S4）表达能力受限的问题，在语言建模等任务上取得了与 Transformer 相当的效果。但纯 SSM 类模型在需要精确检索/复制历史信息的任务上通常仍弱于全注意力模型。

针对这一短板，业界普遍采用 **SSM/线性注意力 + 少量全注意力层的混合（Hybrid）架构**：例如 Jamba 将 Mamba 模块与标准自注意力模块交替堆叠；Taipan 等研究提出用"选择性注意力层"仅对少数需要长程交互的 token 补充注意力计算；NVIDIA Nemotron 系列、Qwen3-Next（采用 Gated DeltaNet 一类线性注意力变体）、Kimi Linear 等新一代模型也都采用了类似的混合策略。此外，GPT-OSS、Gemma 3、MiMo 等模型则选择了另一种"异构交织"方案：在滑动窗口注意力与全局全注意力层之间交替，滑动窗口甚至可以小至 128 token，以极低的 KV Cache 开销获得整体效率提升。

### 6.4 Kimi Linear：表达力与效率的进一步平衡

Moonshot AI 于 2025 年发布的 *Kimi Linear: An Expressive, Efficient Attention Architecture* 进一步系统化了"稀疏注意力 vs 线性注意力"的权衡：论文指出，稀疏注意力（如 NSA/DSA）通常能更细粒度地检索历史信息，但仍需保留完整 KV Cache 以供 token 选择，因此在缓存效率上不如维持恒定状态大小的线性注意力模型；Kimi Linear 据此设计了以线性注意力为主、辅以少量注意力层的混合架构，力图在表达力与推理效率之间取得更优的平衡点。

---

## 7. 路线五：分布式长上下文 —— Ring Attention

前四条路线主要解决"单卡如何算得更快、存得更省"，而 **Ring Attention**（Liu et al., UC Berkeley，论文：*Ring Attention with Blockwise Transformers for Near-Infinite Context*）解决的是"当序列长度远超单卡显存容量时，如何跨多设备协同计算注意力"。

其思路可以理解为把 FlashAttention 式的**分块（blockwise）**计算从单卡内部的显存层级（HBM↔SRAM）扩展到多卡/多机之间：将 Q、K、V 沿序列长度维度切分到不同设备上，各设备负责一部分序列的计算；设备之间以"环形（ring）"拓扑相互传递 K/V 块，并使块间通信与本地块的注意力计算**完全重叠**，从而在不做任何近似的前提下，使可处理的上下文长度能够随参与计算的设备数量近似线性扩展。对于使用因果掩码的自回归模型，朴素的按位置切分会导致不同设备计算量严重不均衡（序列靠前的设备计算量明显小于靠后的设备），为此后续出现了 Striped Ring Attention 等改进方案，通过条带状（而非连续块状）划分让各设备计算量更均衡。

Ring Attention 不仅可用于训练（如 UC Berkeley 的大型世界模型 LWM，利用 Blockwise RingAttention 在长达百万 token 级别的视频与语言序列上训练），在自回归推理阶段也可以类比为一个跨机器节点的分布式 FlashDecoding：相比每张卡都复制一份完整 KV Cache 的常规并行策略，Ring Attention 通过分块与相邻节点通信的方式大幅减少了 KV Cache 的冗余副本数量。与之功能类似、常被一并讨论的还有 DeepSpeed Ulysses 等长序列并行方案。

---

## 8. 主流开源/商业模型 Attention 选型一览

| 模型/系列 | 核心 Attention 技术 | 备注 |
|---|---|---|
| Llama 2 / 3 | GQA | 2023 年首批大规模采用 GQA 的开源模型之一 |
| Mistral 7B / Mixtral | GQA + 滑动窗口注意力（SWA） | KV 头数 8、窗口 4096，二者叠加使用 |
| DeepSeek-V2 / V3 / R1 | MLA | 首次提出并大规模验证 MLA 有效性 |
| DeepSeek-V3.2 | MLA + DeepSeek Sparse Attention（DSA） | 在 MLA 基础上叠加轻量索引器做 Top-K 稀疏选择 |
| Kimi K2 | MLA | 采纳 DeepSeek 提出的 MLA 方案 |
| Kimi（早期长文本部署） / MoBA 论文 | MoBA（块级动态稀疏） | 已在 Kimi 平台长文本请求中实际部署 |
| Kimi Linear | 线性注意力为主 + 少量注意力层混合 | 兼顾表达力与 KV Cache 效率 |
| MiniMax-01 / M1 | Lightning Attention（线性）+ 少量 Softmax 层混合 | 每 8 层 7 线性 + 1 全注意力；支持数百万 token 上下文 |
| Qwen3-Next | Gated DeltaNet 一类线性注意力 + 注意力混合 | 属于 SSM/线性注意力混合路线 |
| GPT-OSS / Gemma 3 / MiMo 等 | 滑动窗口注意力 + 全局全注意力交替 | 异构交织，窗口可小至 128 token |
| Jamba、NVIDIA Nemotron 系列 | Mamba(SSM) + 标准注意力混合 | 状态空间模型与注意力交替堆叠 |
| GLM 5 | DSA 一类稀疏注意力 | 跟随 DeepSeek 的稀疏注意力路线 |
| 几乎所有主流训练/推理框架 | FlashAttention-2/3 + PagedAttention（vLLM/SGLang 等） | 属于底层通用优化，与上述架构选型正交、可叠加 |

> 说明：模型迭代速度很快，以上信息基于公开论文/技术报告与技术博客整理，具体某个模型版本的实现细节建议以其最新技术报告为准。

---

## 9. 技术对比与选型建议

| 维度 | GQA | MLA | 滑动窗口/Attention Sink | NSA / MoBA / DSA（动态稀疏） | 线性注意力 / SSM（含混合架构） | FlashAttention / PagedAttention |
|---|---|---|---|---|---|---|
| 解决的核心问题 | KV Cache 显存/带宽 | KV Cache 显存/带宽，且尽量不损失质量 | 长序列的计算/缓存开销 | 长序列下的计算量与检索精度平衡 | 长序列复杂度从二次降到线性/常数 | 单卡内的计算速度与显存管理效率 |
| 是否改变注意力数学本质 | 否（头共享） | 是（低秩压缩+解压） | 是（局部近似） | 是（动态子集近似） | 是（核函数/递归重写） | 否（精确等价） |
| 建模质量风险 | 略低于 MHA | 持平/略优于 MHA | 长距离依赖较弱，需 sink/混合缓解 | 需要良好的选择/路由机制，设计得当可持平全注意力 | 长程精确检索能力偏弱，通常需混合少量全注意力层 | 无（精确算法） |
| 工程实现复杂度 | 低 | 中高（权重吸收、RoPE 解耦） | 低 | 高（需硬件对齐 kernel、训练期支持） | 中高（核函数/状态更新的高效实现） | 中（kernel 级优化，但已有成熟开源实现） |
| 典型代表 | Llama、Mistral、Qwen | DeepSeek V2/V3/R1、Kimi K2 | Mistral、StreamingLLM、GPT-OSS/Gemma3 | DeepSeek NSA/DSA、Kimi MoBA | MiniMax Lightning Attention、Mamba/Jamba、Qwen3-Next、Kimi Linear | vLLM、SGLang 等几乎所有推理框架 |

**选型建议（一般性参考，非绝对结论）：**

- 若追求工程简单、生态成熟、效果稳健：**GQA** 仍是性价比最高的默认选择，尤其适合中小规模模型。
- 若模型规模较大、且极度看重长上下文下的推理成本：可考虑 **MLA**，但需要投入更多工程资源做推理期优化（权重吸收等）。
- 若上下文长度需求达到十万乃至百万 token 级别：**动态稀疏注意力（NSA/MoBA/DSA）或线性注意力/SSM 混合架构**往往是更合理的路线，二者各有侧重——前者精确检索能力更强但仍需完整 KV Cache 支持选择过程，后者缓存效率更高但通常需要叠加少量全注意力层弥补检索短板。
- 无论选择哪种上层架构，**FlashAttention 系列 + PagedAttention 式的显存管理**几乎是当前所有生产级系统的标配底座，属于"正交"优化，理应默认叠加使用。

---

## 10. 趋势与展望

1. **"KV Cache 压缩 + 稀疏化 + 线性化"的组合拳成为常态。** 以 DeepSeek-V3.2（MLA + DSA）、Kimi Linear（线性注意力 + 混合全注意力）为代表，单一技术路线正在被多技术叠加的混合设计取代。
2. **动态、可学习的稀疏/路由机制正在取代静态先验。** 从早期的滑动窗口、Attention Sink 等固定规则，演进到 NSA、MoBA、DSA 这类基于学习到的相关性分数动态选择 token 子集的方案，体现出"让模型自己决定关注哪里"的设计哲学。
3. **状态空间模型与注意力的边界日渐模糊。** Mamba、Gated DeltaNet 等线性/递归结构与标准注意力的混合架构（Jamba、Qwen3-Next、Kimi Linear、Nemotron 等）正成为兼顾超长上下文效率与检索精度的主流范式之一。
4. **硬件感知（hardware-aware）设计成为标配思路。** 无论是 FlashAttention 系列面向 GPU 内存层级的重构，还是 NSA/DSA 面向 Tensor Core 与显存访问模式专门设计 kernel，"算法设计与硬件特性协同优化"已成为高水平注意力研究的共同范式，而不再是单纯的算法论文。
5. **系统层（serving）优化与算法层创新协同演进。** PagedAttention/vLLM 代表的 KV Cache 内存管理创新，与模型侧的 MLA/稀疏注意力等算法创新相互独立又彼此增益，共同决定了大模型的最终推理成本。

---

## 11. 参考文献与资料来源

**基础与 KV Cache 压缩：**
- Vaswani et al., *Attention Is All You Need*, NeurIPS 2017.
- Shazeer, *Fast Transformer Decoding: One Write-Head is All You Need*（Multi-Query Attention）, 2019.
- Ainslie, Lee-Thorp, de Jong, Zemlyanskiy, Lebrón, Sanghai, *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*, arXiv:2305.13245, 2023.
- DeepSeek-AI, *DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model*, arXiv:2405.04434, 2024.
- DeepSeek-AI, *DeepSeek-V3 Technical Report*, arXiv:2412.19437, 2024/2025.

**计算/IO 优化：**
- Dao, Fu, Ermon, Rudra, Ré, *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*, NeurIPS 2022（arXiv:2205.14135）.
- Dao, *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*, ICLR 2024.
- Shah, Bikshandi, Zhang, Thakkar, Ramani, Dao, *FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision*, NeurIPS 2024（arXiv:2407.08608）.
- Dong et al., *Flex Attention: A Programming Model for Generating Optimized Attention Kernels*, arXiv:2412.05496, 2024.
- Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention*, SOSP 2023（arXiv:2309.06180）.

**稀疏注意力：**
- Xiao, Tian, Chen, Han, Lewis, *Efficient Streaming Language Models with Attention Sinks*（StreamingLLM）, ICLR 2024（arXiv:2309.17453）.
- Jiang et al., Mistral AI, *Mistral 7B*, arXiv:2310.06825, 2023.
- Yuan et al., DeepSeek-AI & 北京大学, *Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention*, arXiv:2502.11089, 2025.
- Moonshot AI et al., *MoBA: Mixture of Block Attention for Long-Context LLMs*, arXiv:2502.13189, 2025.
- DeepSeek-AI, *DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models*（DeepSeek Sparse Attention, DSA）, arXiv:2512.02556, 2025.

**线性注意力与状态空间模型：**
- Katharopoulos et al., *Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention*, ICML 2020.
- Qin et al., *The Devil in Linear Transformer*（TransNormer）, 2022.
- MiniMax, *MiniMax-01: Scaling Foundation Models with Lightning Attention*, arXiv:2501.08313, 2025.
- MiniMax, *MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning Attention*, 2025.
- Gu, Dao, *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*, 2023；Dao, Gu, *Transformers are SSMs*（Mamba-2）, ICML 2024.
- Moonshot AI (Kimi Team) et al., *Kimi Linear: An Expressive, Efficient Attention Architecture*, arXiv:2510.26692, 2025.
- Lieber et al., AI21, *Jamba: A Hybrid Transformer-Mamba Language Model*, 2024.

**分布式长上下文：**
- Liu, Zaharia, Abbeel, *Ring Attention with Blockwise Transformers for Near-Infinite Context*, UC Berkeley, 2023.
- Jacobs et al., *DeepSpeed Ulysses: System Optimizations for Enabling Training of Extreme Long Sequence Transformer Models*, 2023.

**技术博客与解读（辅助理解，非一手论文）：**
- Sebastian Raschka, *LLM Architecture Gallery*：GQA / MLA 专题解读，sebastianraschka.com。
- rasbt/LLMs-from-scratch，GitHub 仓库中 MLA 实现笔记。
- planetbanatt.net，*Understanding Multi-Head Latent Attention*。
- 知乎《大模型"注意力简史"：与两位 AI 研究者从 DeepSeek、Kimi 最新改进聊起》等中文技术访谈与解读文章。
- MIT HAN Lab，StreamingLLM 项目主页与 GitHub 仓库。
- vLLM 官方文档及多篇中文技术博客对 PagedAttention 原理的图解。