# 主流推理框架 MoE 技术方案调研报告
### ——以 vLLM、SGLang 为核心的 Mixture-of-Experts 推理系统技术分析

> 报告日期：2026-07
> 范围：MoE 架构演进、推理系统挑战、并行化与通信优化、计算核优化、负载均衡、vLLM/SGLang 工程实现对比

---

## 目录

1. 概述
2. MoE 架构原理与技术演进脉络
3. MoE 推理面临的系统挑战
4. 并行化技术路线：TP / EP / DP / PP
5. 通信优化：All-to-All 与专用通信库（DeepEP、PPLX、NCCL EP）
6. 计算优化：Fused MoE Kernel 与分组 GEMM（DeepGEMM 等）
7. 负载均衡：从辅助损失到 EPLB
8. 系统级协同优化：PD 分离、计算通信重叠、弹性 EP
9. vLLM 的 MoE 实现详解
10. SGLang 的 MoE 实现详解
11. vLLM vs SGLang 对比总结
12. 结论与展望
13. 参考资料

---

## 1. 概述

自 2023 年底 Mixtral 8x7B 开源以来，Mixture-of-Experts（MoE，混合专家）架构已成为大语言模型突破"参数量—算力"矛盾的主流路线：模型通过引入大量稀疏激活的专家（Expert）网络，在总参数量大幅增长的同时，将每个 token 实际激活的参数量（计算量）控制在较低水平。DeepSeek-V2/V3/R1、Qwen3-MoE、Kimi-K2、Llama4-Maverick、GLM-4.5 等主流开源模型均采用了 MoE 架构，其中 DeepSeek-V3 达到 671B 总参数、37B 激活参数的规模。

MoE architecture 给推理系统带来了与稠密（Dense）模型截然不同的挑战：专家权重体积巨大导致单卡难以容纳全部专家；专家的 token 分配（路由）在运行时才能确定，负载天然不均衡；跨设备的专家并行（Expert Parallelism, EP）引入了大规模 All-to-All 通信，通信开销往往成为端到端延迟的主要瓶颈之一。围绕这些问题，vLLM、SGLang 两大主流开源推理框架，以及 DeepSeek 官方开源的 DeepEP、DeepGEMM、EPLB 等基础设施库，共同构成了当前 MoE 推理技术栈的核心。本报告系统梳理该技术栈的原理、演进与工程实现。

---

## 2. MoE 架构原理与技术演进脉络

### 2.1 基本计算范式

MoE 层通常替换 Transformer 中的 FFN（前馈网络）模块。对输入 token 表示 **u**，MoE 层的计算可以概括为：

1. **路由（Routing）**：门控网络（Router/Gate）根据 token 表示计算每个专家的得分（通常经过 softmax 或 sigmoid），并选出 Top-K 个专家；
2. **专家计算（Expert Computation）**：被选中的 K 个专家各自对该 token 做一次 FFN 前向计算；
3. **加权合并（Combine）**：将 K 个专家的输出按路由权重加权求和，必要时再加上恒定参与计算的"共享专家"（Shared Expert）输出。

现代 MoE（如 DeepSeek 系列）的计算可以写成：

```
MoE(u) = Σ_{e∈TopK(u)} g_e · Expert_e(u) + Expert_shared(u)
```

由于每个 token 只激活 K 个专家（K 远小于专家总数 N），模型可以在总参数量（决定模型容量/知识存储能力）与激活参数量（决定单 token 计算 FLOPs）之间解耦，这正是 MoE 相比稠密模型的核心优势。

### 2.2 技术演进脉络

MoE 在大模型中的发展大致遵循如下脉络：

- **1991 年，Adaptive Mixtures of Local Experts（Jacobs & Hinton 等）**：最早提出多个"专家"子网络配合门控网络进行任务分工的思想，是 MoE 概念的起点。
- **2017 年，Outrageously Large Neural Networks（Shazeer 等，Sparsely-Gated MoE）**：首次将稀疏门控 MoE 引入深度学习大规模场景，训练出千亿参数级别模型，并提出了负载均衡辅助损失等关键机制，为后续工作奠定基础。
- **2020-2021 年，GShard（Google）**：提出以自动分片（Sharding）方式在 Transformer 中大规模部署 MoE，隔层替换 FFN 为 MoE，默认使用 Top-2 门控，并系统化了专家并行的分布式训练/推理范式。
- **2021 年，Switch Transformer（Google）**：将路由简化为 Top-1（每个 token 只选 1 个专家），大幅降低路由和通信复杂度，同时通过容量因子（capacity factor）等机制控制负载均衡，验证了极简路由在保持效果的同时可显著提升推理/训练效率。
- **2023 年 12 月，Mixtral 8x7B（Mistral AI）**：首个受到广泛关注的开源高质量 MoE 大模型，8 个专家、Top-2 路由，46.7B 总参数中约 13B 参数被激活，效果可比肩更大的稠密模型，直接推动了开源社区对 MoE 推理系统的工程投入。
- **2024 年初，DeepSeekMoE**：提出"细粒度专家切分（Fine-grained Expert Segmentation）"和"共享专家隔离（Shared Expert Isolation）"两大核心策略——将专家进一步切分为更小的粒度以增加专家数目和路由组合的灵活性，同时固定若干"共享专家"承担通用知识、减少路由专家之间的冗余，从而在同等计算量下取得更高的专家专精度。
- **2024 年 5 月，DeepSeek-V2**：将 DeepSeekMoE 架构与 Multi-head Latent Attention（MLA，一种低秩压缩 KV Cache 的注意力机制）结合，成为当时最强的开源稀疏模型之一。
- **2024 年 12 月，DeepSeek-V3**：671B 总参数、37B 激活参数，256 个路由专家中每个 token 激活 8 个，并引入了 **auxiliary-loss-free（无辅助损失）负载均衡策略**——通过为每个专家动态调整一个路由偏置项来实现均衡，避免传统辅助损失对模型效果的负面干扰；同时采用了 Multi-Token Prediction（MTP，训练时使用，推理时可选)。DeepSeek-V3/R1 的开源直接带动了 vLLM、SGLang 对大规模专家并行的工程投入。
- **同期，其他代表性 MoE 模型**：Qwen3-MoE、Kimi-K2（1万亿参数级）、Llama4 系列（如 Maverick，128 个路由专家、Top-1）、GLM-4.5 等，专家数目、粒度、路由策略各有差异，但都遵循"细粒度专家 + 部分共享专家 + Top-K 稀疏路由"的总体范式。

### 2.3 路由机制的演化要点

- **门控函数**：从最初的 softmax 竞争式路由，演化到 DeepSeek-V3 等模型采用的 **per-expert sigmoid 门控**，使 token 可以独立评估、选择多个专家而无需相互"竞争"归一化概率，路由更灵活。
- **负载均衡约束**：早期方法依赖训练时的辅助损失（Load Balancing Loss）迫使路由趋于均匀，但会干扰主任务目标；DeepSeek-V3 提出的无辅助损失方法通过运行时动态调整每个专家的偏置项来间接引导路由均衡，对模型质量影响更小。
- **分组路由（Group-limited Routing / Device-limited Routing）**：DeepSeek 系列在专家规模很大时，先将专家分组，再限制每个 token 的候选专家来自有限的组（进而映射到有限的设备），从而降低跨设备通信量，是后续 DeepEP、EPLB 等系统设计的重要前提。

---

## 3. MoE 推理面临的系统挑战

相较稠密模型，MoE 推理系统需要额外解决以下问题：

1. **显存墙（Memory Wall）**：专家权重总量巨大（如 DeepSeek-V3 达 671B 参数），单张 GPU 无法容纳全部专家，必须通过专家并行（EP）跨设备切分，或引入 CPU/NVMe 卸载、专家缓存（Expert Cache）等技术在资源受限场景下运行。
2. **动态负载不均衡**：专家的实际激活频率由输入数据在运行时决定，不同专家的负载可能相差数倍，若简单地将专家均匀分布到设备上，会出现"热点专家所在 GPU 成为瓶颈、冷门专家所在 GPU 空闲"的现象，需要负载均衡算法（如 EPLB）动态调整专家的物理放置。
3. **All-to-All 通信开销**：在 EP 模式下，每个 MoE 层前后都需要执行一次 Dispatch（将 token 发送到其路由到的专家所在设备）和一次 Combine（将专家计算结果收集回原设备）的全对全通信，通信量与批大小、隐藏维度、EP 并行度、跨节点带宽密切相关，往往成为端到端延迟的主要占比之一（据实测，在 8×H200 集群上通信阶段可能占据每步 12%-18% 的时间）。
4. **Prefill 与 Decode 阶段特性差异大**：Prefill 阶段 token 数量多、可以聚合较大批量摊薄通信开销（高吞吐模式）；Decode 阶段每步 token 数量少（尤其在低并发场景），All-to-All 通信的延迟难以隐藏，需要专门的低延迟（Low-Latency）通信内核。
5. **变长 GEMM 形状**：由于路由结果动态变化，每个专家实际处理的 token 数（GEMM 的 M 维）是运行时才确定的，传统静态 tuned GEMM 库难以高效应对这种"分组且变长"的矩阵乘法模式，需要专门的分组 GEMM（Grouped GEMM）内核。
6. **与其他并行/优化技术的耦合**：EP 需要与张量并行（TP）、数据并行（DP，尤其是"DP Attention"）、流水并行（PP）、量化、CUDA Graph、投机解码等技术协同设计，工程复杂度显著提高。

---

## 4. 并行化技术路线：TP / EP / DP / PP

MoE 推理系统中常见的并行策略包括张量并行（TP）、专家并行（EP）、数据并行（DP）、流水线并行（PP），实践中往往是多种策略的组合（Hybrid Parallelism）。

### 4.1 张量并行（Tensor Parallelism, TP）

TP 将每个专家内部的权重矩阵按列/行切分到多张 GPU 上，每张卡持有所有专家的一部分权重。TP 的优势是实现简单、通信模式规整（All-Reduce/All-Gather），劣势是每个专家的计算被切得很碎，在专家数目很多、每专家权重较小的细粒度 MoE（如 DeepSeek-V3 的 256 个专家）场景下，TP 度数增加会使矩阵计算效率下降，同时所有卡都需要保存全部专家的权重分片，无法通过增加并行度来降低单卡显存占用。

### 4.2 专家并行（Expert Parallelism, EP）

EP 与 TP 的核心区别在于：EP 是将"整个专家"作为最小切分单位分布到不同设备，而不是切分权重矩阵内部。每张 GPU 只需持有一部分专家的完整权重，因而可以通过增大 EP 并行度线性降低单卡的专家显存占用，这对 DeepSeek-V3/R1 这类大规模 MoE 模型是刚需。EP 的代价是需要在每个 MoE 层前后插入 All-to-All 通信，将 token 路由到其对应专家所在的设备，再将结果收集回来。当前 SGLang 与 vLLM 在部署 DeepSeek 系列、Qwen-MoE 等大模型时的默认选择都是启用大规模 EP（可扩展到十几甚至上百张 GPU）。

### 4.3 数据并行与"DP Attention"

传统数据并行（DP）指多份完整模型副本各自独立处理不同请求、互不通信。但在 MoE + EP 场景下，vLLM、SGLang 采用了一种特化的"**DP Attention**"策略：Attention 层（含 MLA）按 DP 方式复制，各 GPU 独立处理各自批次的注意力计算（节省 KV Cache 相关的重复计算/存储）；而 MoE 层则按 EP 切分专家，需要通过 All-to-All（或 All-Gather/Reduce-Scatter）将不同 DP rank 的 token 重新分发到专家所在设备、计算完成后再分发回对应的 DP rank。这种"Attention 走 DP、MoE 走 EP"的组合，是当前服务 DeepSeek 系列等超大规模 MoE 模型的主流拓扑，其显著优势是避免了纯 TP 模式下 Attention 计算和 KV Cache 的重复冗余。

### 4.4 流水线并行（PP）与混合策略

PP 按层切分模型到不同设备阶段化执行，可以与 EP/TP/DP 组合使用以支撑超大规模部署（如上百 GPU）。在实际选型上：

- **低并发、低延迟优先**：TP + EP（较小 EP 度）在首 token 时延（TTFT）上通常更有优势；
- **高并发、高吞吐优先**：DP + EP（更大规模的专家并行）能取得更高的整体吞吐，但会牺牲部分低负载场景下的延迟表现；
- 官方与社区的实测（如 vLLM MoE Playbook、Red Hat/AMD 等技术博客）显示，不同并行组合在吞吐与延迟上存在明显的交叉点，需要结合具体硬件拓扑（NVLink/RDMA 带宽）、模型专家粒度、请求并发度进行针对性调优。

---

## 5. 通信优化：All-to-All 与专用通信库

EP 模式下的核心通信原语是 **Dispatch**（token → expert）与 **Combine**（expert 输出 → 原设备），本质上是一次 All-to-All 通信。围绕如何高效实现这一通信，业界形成了多个专用库：

### 5.1 DeepEP（DeepSeek 开源）

DeepEP 是 DeepSeek 团队开源的专为 MoE/EP 场景设计的高吞吐、低延迟 GPU 全对全通信库，核心特性包括：

- 提供两种模式：**高吞吐（Normal / High-Throughput）模式**面向训练与推理 Prefill 阶段，通过 NVLink + RDMA（NVSHMEM）混合传输，充分利用节点内高带宽互联和跨节点 RDMA；**低延迟（Low-Latency）模式**基于纯 RDMA，专为推理 Decode 阶段优化，采用"基于 Hook 的通信-计算重叠"方法，在几乎不占用 GPU 流处理器（SM）资源的情况下将通信延迟隐藏在计算之后；
- 原生支持 FP8 等低精度进行 dispatch/combine，降低通信数据量；
- 针对 DeepSeek-V3 提出的分组限制路由（group-limited gating）算法做了专门优化，减少跨节点通信范围；
- V2 版本相比 V1 大幅重构，以远少于原来的 SM 占用达成同等或更优性能，并进一步扩大了可支持的跨节点通信规模。

DeepEP 已被 SGLang、vLLM 均集成为核心的 EP All-to-All 后端之一。

### 5.2 PPLX（Perplexity 开源）

PPLX 是 Perplexity AI 开源的另一套 All-to-All 通信内核，特点是实现相对更简洁、与 CUDA Graph 兼容性更好，适合分块预填充（Chunked Prefill）等场景。据 Red Hat/vLLM 社区实测，在单节点场景下 PPLX 通常优于 DeepEP，而在跨节点（多机）场景下 DeepEP 往往表现更优——这与两者的通信路径设计侧重不同有关（PPLX 更多针对节点内 NVLink 优化，DeepEP 对跨节点 RDMA 做了更深入的定制）。

### 5.3 NCCL EP 与其他后端

随着 EP 成为主流需求，NVIDIA NCCL 也在推进统一的专家并行通信 API（NCCL EP），目标是提供官方原生支持的高吞吐/低延迟 EP 通信原语，避免各推理框架各自维护定制通信库的碎片化问题；该能力已作为 `nccl_high_throughput` / `nccl_low_latency` 等后端集成进 vLLM 的 All2All Manager 抽象中。除此之外，还有面向国产 AI 芯片的 DeepEP 适配（如 Ascend 平台的 DeepEP-Ascend）、面向 AMD GPU 的 MoRI-EP（提供节点内/跨节点/低延迟三种模式）等。

### 5.4 通信后端抽象层设计

vLLM 与 SGLang 均采用了"可插拔通信后端"的设计理念：将 Dispatch/Combine 抽象为统一接口（vLLM 中称为 `FusedMoEPrepareAndFinalize`，SGLang 中称为 `Dispatcher`），底层可以自由切换 DeepEP、PPLX、NCCL EP、MoRI、Mooncake/NIXL（用于 PD 分离场景下的 KV 与 token 传输）等具体实现，从而在不同硬件拓扑、不同推理阶段（Prefill/Decode）灵活选择最优通信策略。例如 SGLang 在 PD 分离部署时，Prefill 实例使用 DeepEP 的 Normal 模式，Decode 实例使用 Low-Latency 模式，以匹配两阶段截然不同的通信特性需求。

---

## 6. 计算优化：Fused MoE Kernel 与分组 GEMM

### 6.1 Fused MoE Kernel 的设计动机

朴素实现中，MoE 层的计算需要经历：为每个专家收集其对应的 token（Gather/Permute）→ 逐专家做矩阵乘法 → 结果按路由权重加权、写回原位置（Scatter/Unpermute），涉及大量小算子和显存搬运开销。vLLM、SGLang 均实现了 **Fused MoE Triton Kernel**，将排序、分组矩阵乘法、加权求和等步骤尽可能融合进少数几个高效的 GPU Kernel 中，减少 kernel 启动开销与中间结果的显存读写。SGLang 团队在与 vLLM 团队的联合博客中对比了双方的 Fused MoE Triton Kernel 性能，通过持续的联合调优（如 `tuning_fused_moe_triton.py`、`benchmark_vllm_vs_sglang_fused_moe_triton.py` 等工具），推动了两个开源项目在该模块上的相互借鉴与共同演进。

### 6.2 分组 GEMM（Grouped GEMM）与 DeepGEMM

由于路由结果的动态性，每个专家在一个批次内实际需要处理的 token 数量（GEMM 的 M 维）是运行时才知道、且各专家互不相同，这催生了"分组 GEMM"这一专门问题：N、K（专家权重的输出/输入维度）在同一层内固定，仅 M 维（token 数）按专家分组、且各组长度可变。

DeepSeek 团队开源的 **DeepGEMM** 是这一领域的代表性工作：

- 专为 NVIDIA Hopper / Blackwell 架构的 FP8 Tensor Core 定制，采用细粒度（如 128×128 分块）的动态缩放因子（Scaling Factor），在保留 FP8 高吞吐优势的同时降低量化误差；
- 提供两种分组 GEMM 布局：**连续布局（Contiguous Layout）**，将不同专家的 token 拼接成一个张量，适用于 Prefill/训练等专家 token 数预先可知的场景；**掩码布局（Masked Layout）**，适用于结合 CUDA Graph 的 Decode 阶段（CPU 侧无法提前得知每个专家的 token 数，需要通过 Mask 处理变长问题）；
- 核心代码量极小（约300行核心 kernel），基于 CUTLASS/CuTe 的部分理念但不重度依赖其模板体系，兼具高性能与可读性，实测性能可比肩甚至超过经过专家调优的商业库（相对提升可达 1.2×~2.7×，具体取决于矩阵形状）；
- 近期版本（如支持 DeepSeek-V3.2 的 lightning indexer 打分 kernel、"Mega MoE" 融合 kernel）进一步将通信与计算做了更深度的算子级融合，尝试用一个 kernel 同时完成 NVLink 通信与张量核心计算的重叠。

除 DeepGEMM 外，业界还有 CUTLASS Grouped GEMM、Triton 实现的分组 GEMM 等多种技术路线，vLLM 的 Fused MoE Modular Kernel 设计允许针对不同硬件、不同量化格式插拔不同的分组 GEMM 实现（如 `TritonExperts`、`BatchedTritonExperts`、DeepGEMM-backed Experts 等）。

### 6.3 低精度量化

FP8（E4M3/E5M2）已成为 MoE 大模型推理的主流精度选择之一：既可以将专家权重与激活量化为 FP8 存储/计算，在 Hopper/Blackwell 等支持原生 FP8 Tensor Core 的硬件上获得接近 2 倍于 BF16 的算力吞吐，又能显著降低 All-to-All 通信阶段传输的数据量（DeepEP 原生支持 FP8 dispatch/combine）。除 FP8 外，INT8/INT4（W4A16、W8A8 等）量化方案也被广泛用于边缘/资源受限场景下的 MoE 推理，以进一步压缩专家权重的显存占用。需要注意的是，量化通信与量化计算是两个独立议题——多数早期系统只对 GEMM 计算做了 FP8 加速，而通信阶段仍使用高精度格式，近期工作（如 FP8-Flow-MoE 等研究）开始探索"通信与计算全链路 FP8"的一体化方案，以进一步压缩端到端开销。

---

## 7. 负载均衡：从辅助损失到 EPLB

### 7.1 训练期负载均衡

早期 MoE 训练主要依赖**辅助损失（Auxiliary Loss / Load Balancing Loss）**：在主任务损失之外增加一项惩罚项，鼓励路由结果在各专家间趋于均匀分布。这类方法简单有效，但会与主任务目标产生一定冲突，可能损害模型质量。DeepSeek-V3 提出的 **无辅助损失（Auxiliary-Loss-Free）负载均衡策略**是该方向的重要改进：为每个专家维护一个可动态调整的路由偏置（bias）项，在计算 Top-K 时叠加该偏置来间接引导路由趋于均衡，训练过程中根据各专家实际负载持续更新偏置，而不直接干扰主任务的梯度信号，从而在保证负载均衡的同时更好地保留模型效果。

### 7.2 推理期负载均衡：EPLB

即使训练阶段做了负载均衡，**推理阶段**的实际请求分布仍会导致专家负载的动态不均衡（不同任务、不同 Prompt 分布下热点专家可能不同）。为此 DeepSeek 团队开源了 **EPLB（Expert Parallelism Load Balancer）**，核心思路是"冗余专家 + 启发式装箱"：

1. **统计专家负载**：持续采集各专家在最近一段时间内的实际激活/计算负载；
2. **冗余复制热点专家**：对负载较高的专家进行复制（生成多个物理副本），使其可以被分摊到多张 GPU 上处理，缓解单点热点；
3. **启发式装箱（Packing）**：将（含冗余副本的）专家物理实例重新打包分配到各 GPU，使各 GPU 承载的总负载尽量均衡；
4. **两种策略**：
   - **分层负载均衡（Hierarchical）**：适用于节点数可以整除专家分组数的场景，先将专家组均匀分配到各节点（减少跨节点通信），再在节点内部对专家做副本复制与装箱，通常用于 Prefill 阶段（EP 并行度相对较小）；
   - **全局负载均衡（Global）**：忽略专家分组，直接基于负载在全局范围内复制、装箱专家，通常用于 Decode 阶段（EP 并行度较大）的场景。

vLLM（含 Ascend 等硬件后端插件）与 SGLang 均已集成 EPLB（或参考其算法自研实现），通过 `--enable-eplb` 等参数开启，并可配合 `--eplb-config.num_redundant_experts` 等选项配置冗余专家数量。EPLB 的引入使得大规模 EP 部署在面对真实、动态变化的流量分布时仍能维持较高的 GPU 利用率均衡度，是当前 DeepSeek 系列等超大 MoE 模型规模化服务的关键基础设施之一。

---

## 8. 系统级协同优化：PD 分离、计算通信重叠、弹性 EP

### 8.1 Prefill-Decode（PD）分离与 MoE 的结合

PD 分离（Prefill-Decode Disaggregation）已成为大规模 LLM 服务的标准实践：Prefill 阶段是计算密集型（Compute-bound），关注首 Token 时延（TTFT）；Decode 阶段是访存/通信密集型（Memory/Communication-bound），关注单 Token 生成时延（TPOT）。二者若混部在同一批实例上运行，会因资源争抢而相互干扰，分离部署后可以针对两阶段的不同特性做独立优化与独立扩缩容。

在 MoE 场景下，PD 分离与 EP 的结合尤为重要：由于 Prefill、Decode 两阶段的通信特征差异巨大（Prefill 批量大、可摊薄通信；Decode 批量小、延迟敏感），像 DeepEP 这样的通信库无法在同一份进程/通信组内同时以"高吞吐模式"服务 Prefill、以"低延迟模式"服务 Decode。因此，SGLang、vLLM 等框架采用 PD 分离架构后，可以让 Prefill 实例专用高吞吐通信模式、Decode 实例专用低延迟通信模式，二者各自选择独立的 EP 并行度，从而分别达到各自阶段的最优效率。SGLang 团队在 2025 年公开的技术博客中展示了基于 12 节点（96 张 H100）、PD 分离 + 大规模 EP 的部署方案，在 2000 token 输入长度下达到约每节点 5.23 万 input tokens/s、2.23 万 output tokens/s 的吞吐，是较早在开源社区复现官方 DeepSeek 推理系统性能量级的公开实践之一。

### 8.2 计算与通信重叠（Overlap）

由于 All-to-All 通信往往难以完全隐藏在计算之内，SGLang、vLLM 都在探索**双批次重叠（Dual-Batch Overlap, DBO）**与**单批次重叠（Single-Batch Overlap, SBO）**等技术：将不同 micro-batch（或同一批次内可并行的子任务，如共享专家计算与路由专家通信）在时间上交错执行，使一部分的通信操作与另一部分的计算操作在同一时刻发生，从而尽量隐藏通信延迟。SGLang 提供了基于"Dispatcher-Hook"的可扩展重叠框架，允许在 Dispatch/Combine 前后插入自定义 Hook（例如让共享专家的计算与路由专家的 Combine 通信重叠执行）而无需侵入核心 MoE 模块逻辑；vLLM 也在其 RFC 中规划了 DBO 相关能力，并结合 CUDA Graph 进一步降低 Python 层调度开销。DeepGEMM 近期推出的 "Mega MoE" 融合 Kernel，则尝试把 NVLink 通信与张量核心计算融合进同一个 GPU Kernel 内原生重叠，是该方向更进一步的探索。

### 8.3 弹性专家并行（Elastic EP）

vLLM 在 2026 年上半年推出了 **Elastic Expert Parallelism（弹性 EP）**特性：允许在服务运行期间动态地扩容/缩容参与 EP 的 GPU/节点数量（例如借助 Ray 的动态集群管理，新增节点加入后可将部分专家动态迁移过去），而无需中断正在处理中的请求。该特性与 EPLB、通信后端协同工作，是应对云上资源弹性伸缩、容错自愈等场景的重要基础设施方向。

### 8.4 与 CUDA Graph、投机解码等技术的协同

大规模 EP 场景下，Decode 阶段每步的 Python 调度开销、Kernel Launch 开销占比会明显上升，因此需要 CUDA Graph 捕获整个前向计算图以降低 host 侧开销；但 CUDA Graph 通常要求张量形状（含各专家 token 数）固定，这与 MoE 路由的动态性天然冲突，因而催生了如前述 DeepGEMM 的"Masked Layout"分组 GEMM（用固定形状 + Mask 掩盖动态性）等适配方案。此外，投机解码（Speculative Decoding）、Multi-Token Prediction 等加速单请求生成速度的技术，也需要与 MoE 的路由、通信、批处理机制协同设计，是当前推理系统研究的活跃方向之一。

---

## 9. vLLM 的 MoE 实现详解

vLLM 围绕 `FusedMoE` 层构建了一套模块化（Modular Kernel）的 MoE 执行框架，主要设计要点包括：

### 9.1 FusedMoE Modular Kernel 架构

vLLM 将一次 MoE 前向计算拆分为两个可独立替换的抽象层：

- **`FusedMoEPrepareAndFinalize`**：负责量化、All-to-All 的 Dispatch（`prepare`）与 Combine（`finalize`，含 Top-K 权重应用和 Reduce）。针对不同通信后端提供不同子类实现，例如 `PplxPrepareAndFinalize`（对接 PPLX 内核）、`DeepEPHTPrepareAndFinalize`（对接 DeepEP 高吞吐模式）、`DeepEPLLPrepareAndFinalize`（对接 DeepEP 低延迟模式）等；
- **`FusedMoEPermuteExpertsUnpermute`（Experts Kernel）**：负责实际的专家计算（排序/分组、分组 GEMM、反排序），如 `TritonExperts`、`BatchedTritonExperts`（用于配合 batched 格式的输入，适配 PPLX/DeepEP-LL 等）。

二者通过标准化的输入/输出激活格式（标准格式 Standard 或批量格式 Batched）对接，使得 vLLM 可以自由组合"某种通信后端 + 某种专家计算 Kernel + 某种量化格式"，而不必为每种组合单独硬编码实现，显著提升了系统的可扩展性——这也是 vLLM 社区在 RFC #16037（"Data Parallel Attention and Expert Parallel MoEs"）中持续演进的核心设计。

### 9.2 All2All Manager 与多后端支持

vLLM 抽象出 `All2All Manager` 负责管理各类通信后端（PPLX、DeepEP HT/LL、NCCL EP 等）的初始化与缓冲区（Buffer）管理，`FusedMoEPrepareAndFinalize` 在需要时从对应 Manager 中获取通信句柄（Handle）来调用 Dispatch/Combine。这一设计使新通信后端（如近期集成的 NCCL EP）可以以插件形式接入，而无需改动上层 MoE 层逻辑。

### 9.3 并行策略与关键特性

- 支持 TP、EP、DP（含 DP Attention）、PP 及其组合，并通过 `--enable-expert-parallel`、`--data-parallel-size`、`--tensor-parallel-size` 等参数灵活配置；
- 集成 EPLB（`--enable-eplb` 及冗余专家数量配置），用于动态负载均衡；
- 支持 CUDA Graph 与 DP+All2All 后端的组合，以降低大规模 EP 部署下的调度开销；
- 2026 年上半年新增 **Elastic EP**（弹性专家并行），支持运行时扩缩容 EP 集群规模；
- 官方 ROCm（AMD）文档中也提供了针对 AITER（AI Tensor Engine for ROCm）优化 Kernel 的 MoE 支持，体现了 vLLM 在多硬件后端（NVIDIA / AMD / TPU / Trainium 等）上保持 MoE 能力一致性的工程投入。

### 9.4 典型部署命令示例（示意）

```bash
# DP + EP 大规模专家并行示例（示意性质，具体参数以官方文档为准）
vllm serve deepseek-ai/DeepSeek-V3 \
  --tensor-parallel-size 1 \
  --data-parallel-size 8 \
  --enable-expert-parallel \
  --enable-eplb \
  --eplb-config.num_redundant_experts 32
```

---

## 10. SGLang 的 MoE 实现详解

SGLang 的 MoE 实现同样围绕一个统一的 `FusedMoE` 类展开，并在此基础上派生出针对 EP 场景的 `DeepEPMoE` 等特化实现，整体设计强调"分阶段解耦 + Dispatcher 抽象 + Hook 扩展点"。

### 10.1 分阶段流水线设计

SGLang 将一次 MoE 前向拆解为 **Dispatch → Pre-permute → Core Runner（专家计算） → Post-permute → Combine** 五个阶段，各阶段之间通过标准接口衔接，使得引入新的通信后端、新的计算 Kernel、新的重叠策略时无需重构核心逻辑，只需扩展相应阶段的实现。这一设计与 vLLM 的 Modular Kernel 理念高度相似，反映出业界在 MoE 推理系统架构上正逐步收敛到相近的分层抽象范式。

### 10.2 Dispatcher 抽象与多通信后端

SGLang 支持的通信 Dispatcher 包括：

- **`DeepEPDispatcher`**：对接 DeepEP，支持 Normal（高吞吐）与 Low-Latency（低延迟）两种模式；
- **`MoriEPDispatcher`**：面向 AMD 平台，对接 MoRI 库，支持 Intra-Node、Inter-Node、Low-Latency 三种模式；
- **`StandardDispatcher`**：用于非 EP 或简单 TP-based MoE 场景，基于 All-Reduce/All-Gather 而非 All-to-All；
- 此外还支持 Mooncake、NIXL-EP、Ascend 平台的 `ascend_fuseep` 等后端，覆盖国产 AI 芯片与多种 RDMA 传输方案。

当前 DeepEP、Mooncake、NIXL-EP、`ascend_fuseep`、MoRI 等后端均要求 `ep_size == tp_size`；若需要 EP 度小于 TP 度的"混合 EP/TP"部署，则只能退化使用基于 All-Reduce/All-Gather 的 `none` 后端。

### 10.3 计算内核：DeepGemmRunnerCore 等

SGLang 通过 `DeepGemmRunnerCore` 集成 DeepGEMM 库提供的高性能分组 GEMM（同时支持 Contiguous 与 Masked 两种布局），并结合 Triton 自研的 Fused MoE Kernel，作为专家计算阶段（Core Runner）的可选实现，用户可通过 `--moe-runner-backend`（如 `deep_gemm`）指定。

### 10.4 EPLB 与重叠优化

- SGLang 集成了 DeepSeek 开源的 EPLB 算法，通过分析专家实际激活统计信息，动态计算专家的最优放置方案（含冗余复制与装箱），以 `--enable-eplb` 开启；
- 引入 **Single-Batch Overlap（SBO）** 机制（`--enable-single-batch-overlap`），通过 Dispatcher 的 Hook 系统在 Dispatch/Combine 前后插入自定义逻辑，实现如"共享专家计算与 DeepEP Combine 通信重叠"等细粒度优化；
- 支持 PD 分离模式下针对 Prefill/Decode 分别配置 DeepEP 模式（`--deepep-mode normal` / `--deepep-mode low_latency`）。

### 10.5 典型部署命令示例（示意）

```bash
python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3 \
  --moe-a2a-backend deepep \
  --moe-runner-backend deep_gemm \
  --tp 8 --ep 8
```

### 10.6 大规模 EP 生产实践

LMSYS/SGLang 团队公开的 96×H100（12 节点）DeepSeek 部署案例是社区内较早、较完整地复现官方 DeepSeek 推理系统性能量级的开源实践，验证了"PD 分离 + 大规模 EP + DeepEP + DeepGEMM"这一技术组合在真实生产环境下的可行性与有效性，也是当前 SGLang 在超大规模 MoE 模型服务领域获得广泛认可（包括被 xAI、微软 Azure 等采用于 Grok、DeepSeek-R1 等模型的生产部署）的重要技术支撑。

---

## 11. vLLM vs SGLang 对比总结

| 维度 | vLLM | SGLang |
|---|---|---|
| 核心抽象 | `FusedMoE` + Modular Kernel（`PrepareAndFinalize` + `PermuteExpertsUnpermute`） | `FusedMoE` + 分阶段 Dispatcher（Dispatch→Permute→Runner→Unpermute→Combine） |
| 通信后端 | PPLX、DeepEP（HT/LL）、NCCL EP 等，经 All2All Manager 统一管理 | DeepEP、MoRI-EP、Mooncake、NIXL-EP、Ascend `ascend_fuseep`、Standard（All-Reduce/Gather）等，经 Dispatcher 抽象 |
| 分组 GEMM/计算内核 | Triton Fused MoE Kernel、DeepGEMM 等，可插拔组合 | Triton Fused MoE Kernel、DeepGemmRunnerCore（DeepGEMM）等 |
| 负载均衡 | 集成/参考 EPLB，`--enable-eplb` | 集成 DeepSeek EPLB，`--enable-eplb` |
| 并行策略 | TP / EP / DP（含 DP Attention）/ PP 及组合，支持 Elastic EP（弹性扩缩容） | TP / EP / DP（含 DP Attention）/ PP 及组合，强调大规模 EP + PD 分离 |
| 计算通信重叠 | Dual-Batch Overlap（DBO）相关能力持续演进中 | Dual-Batch Overlap（DBO）+ Single-Batch Overlap（SBO，基于 Hook） |
| 硬件生态 | NVIDIA / AMD（CPU+GPU）/ Intel（CPU+GPU）/ Google TPU / AWS Trainium&Inferentia / Intel Gaudi / Arm，硬件覆盖面最广 | 以 NVIDIA GPU 为主，AMD GPU 支持通过与 DeepSeek 社区合作持续增强，也在扩展 Ascend NPU 支持 |
| 典型优势场景 | 硬件多样性要求高、需要跨厂商统一部署的场景；模型架构覆盖面广（含 Encoder-Decoder 等） | 超大规模 MoE 模型（DeepSeek-R1/V3、Qwen-MoE 等）的极限吞吐与低延迟场景；结构化输出（Constrained Decoding）场景 |
| 生态特点 | UC Berkeley Sky Computing Lab 起源，社区规模大，产品化程度高 | LMSYS 起源，与 DeepSeek/Qwen 等中国大模型社区联系紧密，在 MoE、RadixAttention（前缀复用）上投入突出 |

需要说明的是，两个项目在 MoE 相关模块上存在密切的相互借鉴与联合调优关系（例如公开的 Fused MoE Triton Kernel 对比调优博客），二者的底层依赖（DeepEP、DeepGEMM、EPLB 等）也高度重合，实际选型应结合具体硬件环境、模型规模、延迟/吞吐目标、团队工程能力等因素综合评估，且两者的性能特征会随版本快速演进，建议以官方文档与自有 Benchmark 为准。

---

## 12. 结论与展望

1. **MoE 架构的技术红利仍在释放**：从 GShard/Switch Transformer 的粗粒度路由，到 DeepSeekMoE 的细粒度专家 + 共享专家隔离，再到 DeepSeek-V3 的无辅助损失负载均衡，MoE 的架构设计与训练技巧仍在持续演进，直接影响推理系统的设计空间（如分组路由决定了 EPLB、DeepEP 的优化空间）。
2. **专家并行（EP）与专用通信库是当前的核心工程战场**：DeepEP、PPLX、NCCL EP 等通信库的持续竞争与融合，反映出 All-to-All 通信优化对超大规模 MoE 推理性能的决定性影响，未来大概率会走向更统一的官方通信 API（如 NCCL EP）与更深度的通信-计算融合 Kernel（如 DeepGEMM 的 Mega MoE）。
3. **系统架构正从"单一优化点"走向"全链路协同设计"**：PD 分离、DP Attention、EPLB、CUDA Graph、量化、投机解码等技术不再孤立存在，而是需要联合设计才能发挥最大效果，这对推理框架的模块化、可扩展架构（如 vLLM 的 Modular Kernel、SGLang 的分阶段 Dispatcher）提出了更高要求。
4. **vLLM 与 SGLang 呈现"竞争中收敛"的态势**：两者在核心抽象设计理念上高度趋同，共享大量底层基础设施（DeepEP/DeepGEMM/EPLB），同时又各自在硬件生态覆盖（vLLM）与超大规模 EP 极限性能（SGLang）上保持差异化优势，共同推动了开源 MoE 推理技术栈的快速成熟。
5. **未来值得关注的方向**：弹性专家并行（Elastic EP）与云原生弹性调度的结合；通信与计算的算子级融合（Mega Kernel）；FP4 等更低精度在 MoE 推理中的落地；面向国产 AI 芯片（Ascend 等）的 MoE 基础设施适配；以及资源受限场景（单卡/边缘设备）下的专家卸载与缓存（Expert Offloading/Caching）技术的进一步成熟。

---

## 13. 参考资料

**基础论文/技术报告**
- Shazeer et al., *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer*, 2017
- Lepikhin et al., *GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding*, 2020 (arXiv:2006.16668)
- Fedus et al., *Switch Transformer: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity*, 2021 (arXiv:2101.03961)
- Dai et al., *DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models*, 2024 (arXiv:2401.06066)
- DeepSeek-AI, *DeepSeek-V3 Technical Report*, 2024/2025 (arXiv:2412.19437)
- Jiang et al., *Mixtral of Experts*, 2024

**推理系统与调研综述**
- *LLM Inference Serving: Survey of Recent Advances and Opportunities* (arXiv:2407.12391)
- *Rethinking LLM Inference Bottlenecks: Insights from Latent Attention and Mixture-of-Experts* (arXiv:2507.15465)
- *Distributed Hybrid Parallelism for Large Language Models: Comparative Study and System Design Guide* (arXiv:2602.09109)
- *Revealing the Challenges of Attention-FFN Disaggregation for Modern MoE Models and Hardware Systems* (arXiv:2602.09721)
- *ASAP: A Disaggregated and Asynchronous Inference System for MoE Prefill* (arXiv:2606.22541)
- *NCCL EP: Towards a Unified Expert Parallel Communication API for NCCL* (arXiv:2603.13606)

**官方文档与工程博客**
- vLLM 官方文档：*Fused MoE Kernel Features*、*Fused MoE Modular Kernel* — docs.vllm.ai
- vLLM Blog：*Elastic Expert Parallelism in vLLM*（2026-05-14）
- vLLM GitHub RFC #16037：*Data Parallel Attention and Expert Parallel MoEs*
- Red Hat Developer：*Scaling DeepSeek-style MoEs with vLLM and llm-d using Wide EP*（2025-09）
- AMD ROCm Blog：*The vLLM MoE Playbook: A Practical Guide to TP, DP, PP and Expert Parallelism*
- SGLang 官方文档：*Expert Parallelism* — sgl-project.github.io / docs.sglang.io
- LMSYS Org Blog：*Deploying DeepSeek with PD Disaggregation and Large-Scale Expert Parallelism on 96 H100 GPUs*（2025-05-05）
- DeepSeek-AI GitHub：DeepEP、DeepGEMM、EPLB 项目仓库与官方说明站点（deepep.org）
- DeepWiki（kvcache-ai/sglang、sgl-project/sglang）：*Expert Parallelism and MoE Routing*、*Expert Parallelism for MoE Models*