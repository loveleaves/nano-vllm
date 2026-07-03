# DeepSeek 技术路线深度研究报告

> 调研范围：DeepSeek 公开发表的技术论文（arXiv/Nature）、官方博客与 API 文档、第三方技术解读与产业分析。
> 报告时间：2026 年 7 月
> 说明：本报告基于公开资料整理分析，部分前沿细节（尤其是 V4 系列）来自第三方技术拆解与官方报告，可能随后续官方披露而调整。

---

## 一、概述

DeepSeek（深度求索）成立于 2023 年，总部位于杭州，由量化私募高瓴 / High-Flyer 创始人梁文锋出资创立并担任 CEO。与多数大模型公司不同，DeepSeek 自创立之初即采取“研究优先、模型权重开放”的策略，绝大多数正式版本模型均以 MIT 协议开源权重，并配套发布详尽的技术报告。这一路线使其技术细节高度可追溯，也是本报告得以展开系统分析的基础。

DeepSeek 的技术路线可以概括为一条清晰的主线：**用极致的工程与算法效率，在有限算力（尤其是受出口管制约束的 H800 等“减配”GPU）条件下，逼近乃至达到国际一线闭源模型的能力水平**。这条主线先后催生了三个标志性的技术拐点：

1. **DeepSeek-V2（2024年5月）**：提出 Multi-head Latent Attention（MLA）与 DeepSeekMoE，首次证明"低成本训练 + 高效推理"的架构可以兼得；
2. **DeepSeek-R1（2025年1月）**：证明纯强化学习（不依赖大规模人工标注思维链）即可激发大模型的推理涌现能力，并带动开源推理模型对齐 OpenAI o1 级别能力，引发"AI 的 DeepSeek 时刻"；
3. **DeepSeek-V3.2 / V4（2025年末至2026年）**：技术重心从"参数规模竞赛"转向"长上下文效率"与"训练稳定性"，先后引入 DeepSeek Sparse Attention（DSA）、Manifold-Constrained Hyper-Connections（mHC）、Engram 条件记忆等架构创新，并将"思考"与"工具调用"深度融合，转向 Agent 优先设计。

---

## 二、技术路线演进总览

| 时间 | 模型/论文 | 关键技术贡献 |
|---|---|---|
| 2023.11 | DeepSeek-Coder / DeepSeek LLM 67B | 奠基性稠密 Transformer，验证团队基础训练能力与 Scaling Law 复现 |
| 2024.01 | DeepSeekMoE 16B | 细粒度专家切分（fine-grained expert segmentation）+ 共享专家隔离（shared expert isolation） |
| 2024.02 | DeepSeekMath | 提出 **GRPO**（Group Relative Policy Optimization），数学推理能力大幅提升 |
| 2024.03 | DeepSeek-VL | 面向真实场景的视觉-语言模型，混合视觉编码器 |
| 2024.05 | **DeepSeek-V2**（236B/21B 激活） | 首次提出 **MLA**，KV 缓存压缩 93.3%，吞吐提升 5.76 倍 |
| 2024.06 | DeepSeek-Coder-V2 | MoE 代码模型，长上下文与多语言代码能力 |
| 2024.12 | **DeepSeek-V3**（671B/37B 激活） | Auxiliary-Loss-Free 负载均衡、Multi-Token Prediction（MTP）、FP8 混合精度训练、DualPipe 流水线并行，训练成本仅 278.8 万 H800 GPU 小时 |
| 2025.01 | **DeepSeek-R1 / R1-Zero** | 证明纯 RL（GRPO）可从基座模型直接激发推理能力涌现；冷启动数据 + 多阶段训练解决可读性与语言混杂问题；蒸馏出 1.5B–70B 六个稠密推理模型 |
| 2025.01 | Janus-Pro | 解耦视觉编码的统一理解-生成多模态模型，图像生成超越 DALL·E 3 |
| 2025.05 | DeepSeek-R1-0528 | 推理效率与减少幻觉的迭代升级 |
| 2025.08–09 | DeepSeek-V3.1 / V3.1-Terminus | 思考/非思考混合模式（hybrid reasoning），Agent 与工具调用能力强化 |
| 2025.09–12 | **DeepSeek-V3.2-Exp / V3.2** | 引入 **DeepSeek Sparse Attention（DSA）**，长上下文训练/推理成本降低超 50%；V3.2-Speciale 主攻超长链推理与数学定理证明 |
| 2026.01 | mHC 论文 | **Manifold-Constrained Hyper-Connections**，约束残差流通道矩阵为双随机矩阵，提升超深网络训练稳定性 |
| 2026.04 | **DeepSeek-V4（V4-Pro 1.6T / V4-Flash 284B 预览版）** | 混合注意力架构 CSA+HCA、mHC、Engram 条件记忆模块，原生 100 万 token 上下文，MIT 协议开源 |

---

## 三、核心技术深度解析

### 3.1 Multi-head Latent Attention（MLA）：用低秩压缩解决 KV 缓存瓶颈

标准多头注意力（MHA）在自回归推理时必须缓存全部历史 token 的 Key/Value 向量，序列越长、批量越大，显存占用越高，这是制约长上下文与高吞吐推理的核心瓶颈。此前业界主流的缓解手段是 Multi-Query Attention（MQA）和 Grouped-Query Attention（GQA），二者本质上是"减少 KV 头数"，会损失一定的建模能力。

MLA 采取了不同的思路：**将 Key 和 Value 联合低秩压缩为一个共享的潜在向量（latent vector）**，推理时只需缓存这个低维潜向量，而非完整的 K、V 矩阵；解码时再通过上投影矩阵动态重构出对应的 K、V。由于潜向量的压缩维度远小于 MHA 中输出投影矩阵的维度，KV 缓存可以被大幅压缩。值得注意的是，标准 RoPE 旋转位置编码与低秩 KV 压缩天然不兼容（旋转矩阵会破坏低秩结构），DeepSeek 为此设计了**解耦 RoPE 策略**：用额外的多头 Query 与一个共享的 Key 专门承载位置信息，与压缩后的内容信息分离处理。

实测效果（DeepSeek-V2 论文）：相较于上一代稠密模型 DeepSeek 67B，KV 缓存降低 **93.3%**，最大生成吞吐提升至 **5.76 倍**，同时训练成本节省 42.5%。MLA 自 V2 起被后续 V3、R1、V3.1/V3.2 及多模态系列模型沿用，是 DeepSeek 技术栈中最具持续生命力的基础组件。

### 3.2 DeepSeekMoE：细粒度专家与共享专家隔离

传统 MoE 架构（如 GShard、Mixtral）通常使用数量有限、容量较大的专家，每个 token 路由到少数几个专家。DeepSeekMoE 提出两项关键改进：

- **细粒度专家切分（fine-grained expert segmentation）**：将传统专家进一步切分为更多、更小的子专家（同时按比例增加每个 token 激活的专家数量），使专家分工更加精细，缓解"全能型专家"导致的知识冗余问题；
- **共享专家隔离（shared expert isolation）**：划出若干恒定激活的"共享专家"，专门承载跨任务的通用知识，从而让其余"路由专家"能更专注于特定领域知识，进一步降低专家间的知识冗余。

消融实验表明，细粒度切分与共享专家隔离均能独立带来性能提升，二者叠加效果最佳。

**负载均衡的演进**：早期 DeepSeekMoE 与 V2 依赖辅助损失（auxiliary loss）来防止路由坍塌（即少数专家被反复选中、其余专家得不到训练）。但辅助损失会与语言建模主任务的梯度相互干扰。DeepSeek-V3 提出 **Auxiliary-Loss-Free 负载均衡**：为每个专家引入一个动态偏置项，根据该专家近期被使用的频率自动升降——使用过载则降低偏置（降低后续被选中概率），利用不足则提高偏置——该偏置仅参与 Top-K 路由筛选，不进入最终门控权重的计算，从而在不引入额外损失梯度的前提下实现负载均衡，这是 V3 相对 V2 最重要的架构改动之一。

### 3.3 Multi-Token Prediction（MTP）：训练目标与推理加速的统一

DeepSeek-V3 引入 MTP 训练目标：在主模型之外增加浅层 MTP 模块，训练时同步预测未来第 2 个、第 3 个 token（而非仅下一个 token）。这一设计带来两重收益：

- **训练阶段**：密度更高的训练信号使模型在相同数据量下学习效率更高，在多项评测上带来稳定增益；
- **推理阶段**：MTP 模块可直接复用为投机解码（speculative decoding）的草稿模型，实测带来约 1.8 倍的推理吞吐提升。

DeepSeek 还通过把模型最浅层（含嵌入层）与最深层（含输出头）部署在同一条流水线并行（PP）rank 上，使 MTP 模块与主模型物理共享嵌入和输出头参数，进一步节省显存。

### 3.4 FP8 混合精度训练与 DualPipe：算法-框架-硬件协同设计

DeepSeek-V3 是已知首批在 671B 规模上验证 **FP8 混合精度训练**可行性的工作之一。其方案并非简单全局转 FP8，而是细粒度（per-token 1×128、per-block 128×128）量化加上高精度 CUDA Core 累加，使训练损失相对 BF16 基线偏差控制在 0.25% 以内。

更关键的是配套的并行与通信工程：

- **DualPipe 流水线并行**：设计双向调度算法，使前向和反向计算块的计算与通信阶段相互重叠，显著减少流水线气泡（pipeline bubble），并解决跨节点专家并行（Expert Parallelism）带来的通信开销问题；
- **定制化跨节点 All-to-All 通信内核**：充分压榨 InfiniBand 与 NVLink 带宽，将每个 token 的跨节点通信限制在最多 4 个节点以内，降低网络流量；
- **放弃张量并行（TP）**：由于 H800 的 NVLink 带宽受限（出口管制导致的"减配"），DeepSeek 在训练阶段完全避免使用张量并行，转而通过显存的精细优化，在 16 路流水线并行、64 路专家并行（跨 8 节点）与 ZeRO-1 数据并行的组合下完成训练。

最终结果：DeepSeek-V3 在 2048 张 H800 GPU 上，以 **278.8 万 GPU 小时**完成 14.8 万亿 token 的预训练，且全程未出现不可恢复的 loss 尖峰或回滚，体现出极高的训练稳定性。这套"算法-框架-硬件协同设计"（co-design）方法论，是 DeepSeek 应对算力约束的核心方法论，也是其后续每一代模型持续沿用和深化的工程范式。

### 3.5 GRPO 与 DeepSeek-R1：纯强化学习激发推理涌现

GRPO（Group Relative Policy Optimization）最早在 DeepSeekMath（2024年2月）中提出，是 PPO（Proximal Policy Optimization）的变体。其核心改动是**取消独立的价值函数（critic model）**：对同一问题采样一组（group）输出，用组内奖励的均值与标准差对每个输出的奖励做归一化，直接得到相对优势（advantage）估计，从而省去训练并维护一个与策略模型同等规模的价值网络，大幅降低显存开销，尤其适合奖励可被规则化验证的任务（如数学、代码）。

DeepSeek-R1-Zero 在此基础上做了一个大胆实验：**完全跳过监督微调（SFT）冷启动阶段，直接以 DeepSeek-V3-Base 为起点用大规模 RL（GRPO）训练**。结果显示：

- 模型在训练过程中自发涌现出反思（self-reflection）、自我验证、动态调整解题策略等高级推理行为；
- AIME 2024 的 pass@1 准确率从 15.6% 一路提升至 71.0%，结合多数投票可达 86.7%，达到 OpenAI o1-0912 同等水平；
- 但 R1-Zero 存在明显缺陷：输出可读性差、中英文混杂（language mixing）。

为解决上述问题，DeepSeek-R1 在 R1-Zero 的基础上引入**多阶段训练 + 冷启动数据**：先用少量高质量、人类可读的长链思维数据对基座模型做轻量 SFT 冷启动，再进行面向推理的大规模 RL，RL 收敛后再做一轮拒绝采样生成新的 SFT 数据（涵盖推理与非推理场景）并二次微调，最后再做一轮面向全场景（含安全与人类偏好对齐）的 RL。这一"SFT 冷启动 → 推理 RL → 拒绝采样扩充数据 → 二次 SFT → 全场景 RL"的多阶段流程，最终使 R1 在推理基准上达到对标 OpenAI o1-1217 的水平，并同步开源了基于 Qwen、Llama 架构、从 R1 蒸馏出的 1.5B/7B/8B/14B/32B/70B 六个稠密推理模型，大幅降低了社区复现与部署"类 o1"推理能力的门槛。R1 论文后于 2025 年 9 月经修订发表于 *Nature*。

GRPO 此后成为整个行业训练推理模型的事实标准之一，被广泛应用于 DeepSeek 自身的 V3 后训练、V3.1/V3.2 系列，以及众多第三方开源推理模型的复现工作中。

### 3.6 DeepSeek Sparse Attention（DSA）：从"压缩 KV"到"稀疏选择"

V3.2-Exp（2025年9月）在 V3.1-Terminus 基础上引入 DeepSeek Sparse Attention，目标是解决长上下文场景下注意力计算复杂度随序列长度平方增长的问题。DSA 由两个核心组件构成：

1. **Lightning Indexer（闪电索引器）**：一个轻量级、低秩、多头、FP8 计算的打分网络，为当前 query token 与所有历史 token 快速计算"相关性得分"；
2. **细粒度 Token 选择器**：根据索引器得分，仅保留 Top-K（如 K=2048）个最相关的历史 token 参与正式的注意力计算，将核心注意力计算复杂度从 O(L²) 降至 O(L·K)。

工程实现上，DSA 在 MLA 的 MQA（Multi-Query Attention）模式下实例化，使每个潜向量（latent KV 条目）可在多个 Query 头之间共享，与 MLA 已有的压缩机制天然兼容。训练上采用两阶段策略：先冻结主干、仅用 KL 散度损失让索引器模仿稠密注意力的分布（dense warm-up），再解冻全部参数联合优化以适配稀疏模式。

效果上，V3.2-Exp 在长上下文场景下将 API 成本降低超过 50%，且基准性能与 V3.1-Terminus 基本持平。正式版 V3.2（2025年12月）将 DSA 与"思考"和"工具调用"深度融合——支持模型在长链推理过程中直接调用工具，并同时保留快速响应的非思考模式，配合超过 1800 种环境、8.5 万条复杂指令构建的大规模 Agent 训练数据，使其在不针对单一基准做专门优化的情况下，在多项 Agent 公开评测中取得开源模型最佳成绩。同期发布的 V3.2-Speciale 则专注于把推理能力推向极限，在 30 步以上的深度推理任务上表现突出，并继承了 DeepSeek-Math-V2 在数学定理证明方面的能力（在 2025 年 IMO 等测试中取得金牌级表现）。

### 3.7 mHC 与 V4：架构创新进入"残差流"与"记忆"层面

2026 年初发表的 mHC（Manifold-Constrained Hyper-Connections）论文，是 DeepSeek 在 Transformer 最基础组件——残差连接——上的一次系统性改造，建立在 ByteDance 此前 Hyper-Connections 工作之上：

- 将原本单一通道的残差流拓宽为 n_hc=4 倍宽度的多通道残差流，使信息可以通过多条并行路径在层间传递；
- 将负责跨通道信息混合的矩阵约束在 **Birkhoff 多面体**内，即强制其为双随机矩阵（每行每列之和为 1），从而将该矩阵的谱范数严格限定为 1，避免信号在极深网络中爆炸或消失；

这一约束机制为训练参数规模、层数远超 V3 的下一代超深网络提供了稳定性保障。

2026 年 4 月 24 日发布的 **DeepSeek-V4 预览版**（V4-Pro，1.6T 总参数/49B 激活参数；V4-Flash，284B 总参数/13B 激活参数，均原生支持 100 万 token 上下文，MIT 协议开源）是这条架构演进路线的集中体现，主要包含四项创新（部分细节来自第三方技术拆解，官方完整论文尚待全面披露）：

- **混合注意力架构（CSA + HCA）**：Compressed Sparse Attention（CSA）将每 m 个 token 的 KV 压缩为一个条目，再用类 DSA 的方式做 Top-K 稀疏选择；Heavily Compressed Attention（HCA）对可容忍更大近似误差的层做更激进的压缩。第三方评测显示，在 100 万 token 上下文下，V4-Pro 相比 V3.2 单 token 推理 FLOPs 降至约 27%，KV 缓存降至约 10%；
- **mHC**：如上所述，用于支撑更深、更宽网络的训练稳定性；
- **Engram 条件记忆模块**：将"事实性知识检索"与"在线计算推理"在结构上分离，提供近似 O(1) 复杂度的知识检索能力，这是 DeepSeek 首次在生产级模型中尝试"记忆与推理解耦"的设计；
- **Muon 优化器**：替代或补充 AdamW，用于进一步提升大规模训练的优化效率（细节披露有限）。

需要说明的是，社区与第三方分析人士（如 Sebastian Raschka 等）观察到，R2 作为独立"纯推理模型"的传闻在 2025 年中期一度流传，但梁文锋本人对早期 R2 效果不满意叠加芯片供应问题（DeepSeek 一度被官方鼓励改用华为昇腾芯片训练，但因稳定性、互联带宽与软件生态问题最终训练侧仍以 Nvidia 芯片为主、推理侧引入华为芯片）导致延期；目前主流判断是 R2 的技术能力已经融合进 V3.1/V3.2 的混合思考模式以及 V4 的统一架构中，而非作为独立模型发布。

---

## 四、模型谱系全景

DeepSeek 的模型矩阵已从单一语言模型扩展为多条产品线：

- **通用对话/推理主干线**：DeepSeek LLM → V2 → V2.5 → V3 → V3.1 → V3.2 → V4（Pro/Flash），同时维护独立的纯推理分支 R1 → R1-0528（部分推理能力已被后续 V3.1/V3.2 的"思考模式"吸收）；
- **代码模型线**：DeepSeek-Coder → DeepSeek-Coder-V2，专注代码生成、补全与仓库级理解；
- **数学/定理证明线**：DeepSeekMath → DeepSeek-Prover → DeepSeek-Math-V2，强调可验证奖励下的强化学习与形式化证明；
- **多模态理解线**：DeepSeek-VL → DeepSeek-VL2（MoE 化、动态分块、更强 OCR/图表理解）→ DeepSeek-OCR；
- **统一理解-生成线**：Janus → JanusFlow → Janus-Pro，通过解耦视觉编码（理解侧用 SigLIP 语义编码器，生成侧用 VQ tokenizer）在单一自回归框架内同时支持图文理解与文生图，GenEval 等指标超越 DALL·E 3、Stable Diffusion 3-Medium。

各产品线共享同一套底层技术基座（MLA/DeepSeekMoE → DSA → mHC 的演进序列），体现出 DeepSeek "核心架构组件高度复用、面向不同任务做轻量适配"的工程哲学。

---

## 五、关键技术趋势研判

**1. 从"参数规模竞赛"转向"长上下文效率竞赛"。** V1→V3 的演进主线是用 MLA、DeepSeekMoE 在更大参数规模下控制训练/推理成本；而 V3.2 之后，竞争焦点明显转向如何在 100 万级 token 上下文下把计算和显存开销降到最低（DSA → CSA/HCA），这与全行业对超长上下文 Agent 应用的需求增长相吻合。

**2. "思考"与"工具调用"的边界正在消失。** V3.2 首次让模型在长链推理过程中原生穿插工具调用，而不是"先想清楚再调用工具"的串行模式。配合大规模、多环境的 Agent 合成训练数据，DeepSeek 的技术重心已从"会做题的推理模型"转向"能在真实环境中完成多步任务的智能体"。

**3. 架构创新正从注意力层向残差流、记忆模块等更基础的组件下沉。** mHC 触及的是 Transformer 最核心的信息传递骨架，Engram 触及的是知识存储与调用方式。这意味着 DeepSeek（及整个行业）已经不满足于"调参式"的效率优化，而是开始重新设计 Transformer 的底层数据流拓扑，以支撑参数规模与上下文长度的进一步跃升。

**4. RL 与可验证奖励（verifiable reward）正在取代纯 SFT，成为提升模型能力上限的主要手段。** 从 GRPO 在 DeepSeekMath 的首次亮相，到 R1 全面验证"纯 RL 涌现推理"，再到 V3/V3.1/V3.2 后训练阶段持续依赖 RL 强化代码、数学、Agent 能力，DeepSeek 的技术路线印证了行业从"预训练 Scaling Law"向"后训练/推理时 Scaling Law"重心转移的大趋势。

**5. 算力约束正在反向驱动软硬件协同设计的精细化。** 出口管制下的 H800（以及一度尝试、后又部分回退的华为昇腾芯片）迫使 DeepSeek 在通信调度、混合精度、并行策略等系统层面做出大量定制化优化（DualPipe、FP8、定制通信内核）。这种"以工程效率对冲硬件劣势"的路径，客观上推动了开源社区对训练系统工程的关注度，也为资源受限团队提供了可复现的方法论参考。

**6. 开源策略持续重塑全球 AI 产业格局。** R1 发布后一度造成英伟达等美股 AI 概念股大幅波动，被部分舆论称为"AI 的 Sputnik 时刻"；其影响延续至今——MIT 协议下持续开放权重与技术报告，降低了全球范围内复现"准前沿水平"模型的门槛，加速了开源模型对闭源模型市场份额的侵蚀，也推动了 OpenAI 等原本完全闭源的厂商重新发布开放权重模型。与此同时，关于其训练数据来源透明度、是否使用其他厂商模型输出进行蒸馏训练等争议（包括 2026 年 2 月 Anthropic 指控 DeepSeek 利用大量虚假账号生成对话以训练自身模型的报道）也持续存在，是后续观察其技术与商业路径时需要关注的不确定因素。

---

## 六、挑战与争议

- **训练成本与算力使用的透明度争议**：官方披露的 "V3 训练成本约 558 万美元/278.8 万 H800 GPU 小时" 等数字引发业界广泛讨论，部分分析认为该数字仅覆盖最终一次成功训练的计算成本，未包含此前研发试错、人力及基础设施建设的综合成本；
- **训练数据与版权透明度**：尽管模型权重开放，DeepSeek 并未公开训练数据集的完整来源与授权情况，这与"完全开放"的开源 AI 定义仍有差距；
- **对齐与安全**：第三方研究指出，单纯依赖 RL 的对齐方式在无害性、可读性、对未见场景的泛化能力等方面仍存在不足，需要与 SFT 等监督手段结合；R1 系列模型也被观察到在涉及政治敏感议题时存在更明显的内容审查倾向；
- **地缘政治与供应链**：出口管制、芯片国产化替代（华为昇腾）尝试中暴露出的互联带宽、软件生态短板，是制约其下一代模型训练节奏的重要变量，也是 R2 延期的重要原因之一。

---

## 七、结论

DeepSeek 的技术路线呈现出清晰的"问题驱动"特征：每一代关键技术突破，几乎都对应着一个具体的工程或算法瓶颈——MLA 解决 KV 缓存显存瓶颈，DeepSeekMoE 解决专家冗余与训练成本问题，FP8/DualPipe 解决受限硬件下的训练效率问题，GRPO/R1 解决推理能力涌现对大规模人工标注的依赖问题，DSA/CSA-HCA 解决长上下文计算复杂度问题，mHC 解决超深网络训练稳定性问题。这种"以终为始、围绕瓶颈做系统性协同设计"的方法论，叠加坚持开源权重与技术报告的策略，使其成为过去两年中对全球开源大模型生态影响最深远的力量之一。

展望后续，DeepSeek 在 V4 中展现出的"统一通用与推理能力、原生超长上下文、记忆与推理解耦"的架构方向，以及 Agent 优先的产品设计思路，预计仍将是其 2026 年下半年及之后的核心技术主线；而训练数据透明度、对齐安全性与地缘政治层面的不确定性，则是评估其长期技术与商业路径时不可忽视的变量。

---

## 主要参考资料

1. DeepSeek-AI et al., *DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model*, arXiv:2405.04434
2. DeepSeek-AI et al., *DeepSeek-V3 Technical Report*, arXiv:2412.19437
3. DeepSeek-AI et al., *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*, arXiv:2501.12948（后发表于 *Nature*，2025年9月）
4. Shao et al., *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*, arXiv:2402.03300
5. Dai et al., *DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models*, arXiv:2401.06066
6. Wang et al., *Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts*
7. DeepSeek-AI, *DeepSeek-V3.2-Exp / DeepSeek-V3.2 技术报告*（GitHub / arXiv:2512.02556）
8. DeepSeek-AI, *mHC: Manifold-Constrained Hyper-Connections*, arXiv:2512.24880
9. DeepSeek API Docs，Change Log（api-docs.deepseek.com）
10. Sebastian Raschka, *A Technical Tour of the DeepSeek Models from V3 to V3.2*
11. Wikipedia "DeepSeek" 词条（持续更新）
12. World Economic Forum / Stanford HAI / Bloomberg 等关于 DeepSeek 产业影响的公开报道与分析

*（注：DeepSeek-V4 相关架构细节部分来自第三方技术拆解文章，截至本报告撰写时官方完整技术论文尚未全面公开，相关描述以"分析/推测"性质呈现，请读者以 DeepSeek 官方后续披露为准。）*