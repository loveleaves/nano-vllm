# LoRA Serving 技术调研报告

> 调研范围：LoRA 基本原理与主要变体、多 LoRA 并发推理服务（LoRA Serving）面临的系统性挑战、代表性学术论文与工程方案、开源/工业界实现现状，以及未来技术趋势。
>
> 报告日期：2026-07

---

## 目录

1. 背景与问题定义
2. LoRA 基本原理
3. LoRA 主要技术变体一览
4. LoRA Serving 的核心系统挑战
5. 代表性技术路线与论文详解
   - 5.1 朴素方案：Merge / Switch
   - 5.2 Punica：SGMV 批量算子
   - 5.3 S-LoRA：Unified Paging 与异构批处理
   - 5.4 CaraServe：CPU 辅助的冷启动消除与 rank-aware 调度
   - 5.5 dLoRA：动态 merge/unmerge 与跨副本迁移
   - 5.6 Chameleon：适配器缓存与多级队列调度
   - 5.7 面向异构 rank 的算子优化（MixLoRA / Dynamic SGMV）
   - 5.8 工业界实现：vLLM、LoRAX（Predibase）、TGI/Triton
6. 技术方案对比总结
7. 系统设计要点提炼
8. 未来趋势与开放问题
9. 参考资料

---

## 1. 背景与问题定义

“预训练 + 微调”是当前大语言模型（LLM）落地的主流范式：企业和开发者基于同一个开源基座模型（如 Llama、Qwen、Mistral 等），针对不同任务、不同客户、不同语言分别训练大量 LoRA（Low-Rank Adaptation）适配器。这带来一个新的服务问题：**如何在有限的 GPU 资源上，同时为成百上千个共享同一基座模型、但权重各不相同的 LoRA 适配器提供低延迟、高吞吐的推理服务**，而不是像传统方式那样为每个微调模型单独部署一套完整实例。

这正是 "LoRA Serving"（也称 Multi-LoRA Serving / Multi-Tenant LoRA Serving）要解决的问题，其核心矛盾在于：

- **参数层面**：LoRA 权重本身很小（通常是基座模型的万分之一到千分之一），理论上一张 GPU 可以缓存海量适配器；
- **计算层面**：不同请求可能对应不同的 LoRA 适配器（甚至不同的 rank），若采用传统的"合并权重（merge）后推理"方式，同一批（batch）内的请求就无法共享同一份基座权重矩阵乘法，batching 效率被严重破坏；
- **显存层面**：适配器权重与 KV Cache 需要在同一块 GPU 显存中动态共存、动态换入换出，管理不当会导致显存碎片化和频繁的加载延迟（冷启动问题）。

围绕这一矛盾，学术界与工业界在 2023-2026 年间提出了一系列系统性方案，构成了本报告要梳理的技术路线图。

---

## 2. LoRA 基本原理

LoRA 由 Hu 等人（Microsoft）于 2021 年提出，论文题为《LoRA: Low-Rank Adaptation of Large Language Models》。其核心假设是：大模型在适配下游任务时，权重的**变化量**具有较低的"内在秩"（intrinsic rank），因此可以用两个低秩矩阵的乘积来近似表示这一变化量，而不需要更新全部参数。

### 2.1 数学形式

对于预训练权重矩阵 $W_0 \in \mathbb{R}^{d \times k}$，LoRA 冻结 $W_0$，引入两个低秩矩阵 $A \in \mathbb{R}^{r \times k}$、$B \in \mathbb{R}^{d \times r}$（其中 $r \ll \min(d,k)$），前向计算变为：

$$h = W_0 x + \Delta W x = W_0 x + BA x$$

训练时只更新 $A$、$B$，$A$ 通常用高斯分布初始化、$B$ 初始化为零，保证训练起点等价于原始模型。输出还会乘以缩放系数 $\alpha / r$ 以控制适配强度。

可训练参数量为 $|\Theta| = 2 \times L_{LoRA} \times d_{model} \times r$，相比全量微调可减少几个数量级——论文中以 GPT-3 175B 为例，可训练参数可减少约 1 万倍，显存需求降低约 3 倍，同时在 RoBERTa、DeBERTa、GPT-2、GPT-3 等模型上取得与全量微调相当或更优的效果。

### 2.2 训练与推理的关系

原始论文建议的推理方式是**权重合并**：训练完成后把 $BA$ 加回 $W_0$，得到新的稠密权重 $W = W_0 + BA$，这样推理阶段不会引入任何额外延迟，也是 LoRA 相较于此前 Adapter、Prefix-Tuning 等方法的重要优势之一——不新增推理时延、不占用额外的序列长度。

但这一"合并优化"假设了**同一时刻只服务一个（合并后的）模型**。一旦需要在同一 GPU 上同时服务成百上千个不同任务的 LoRA 适配器，"先合并再推理"的假设就不再成立，这正是 LoRA Serving 问题的起点：多个请求各自绑定不同的 $(A,B)$，无法合并成同一个稠密矩阵，只能在运行时以 $y = xW_0 + xAB$ 的分离形式动态计算。

---

## 3. LoRA 主要技术变体一览

在原始 LoRA 之外，学术界提出了大量变体，它们从不同角度优化训练效果或效率，也间接影响 Serving 系统的设计假设（如是否要求统一 rank、是否需要量化基座模型等）：

| 变体 | 核心思路 | 对 Serving 的影响 |
|---|---|---|
| **QLoRA**（Dettmers 等，2023） | 将基座模型量化为 4-bit NormalFloat（NF4），配合双重量化（Double Quantization）和分页优化器（Paged Optimizer），在保持 LoRA 全精度训练效果的同时大幅降低显存占用，可在单张 48GB GPU 上微调 65B 模型 | 使"量化基座 + 多 LoRA 适配器"成为可能，LoRAX 等系统已支持 AWQ/4-bit 量化基座上挂载多个 LoRA |
| **LoRA+**（Hayou 等，2024） | 指出 $A$、$B$ 使用相同学习率会导致大宽度模型欠拟合，提出为 $A$、$B$ 设置不同学习率比例 | 主要影响训练效果，不直接改变 Serving 架构 |
| **MoSLoRA**（Wu 等，2024） | 将 LoRA 权重分解为两个子空间并引入可学习的 mixer 进行融合，在语言、多模态、扩散模型上均适用 | 增加了适配器内部结构复杂度，Serving 系统需要兼容更灵活的算子形式 |
| **VeRA / NoRA 等** | VeRA 通过共享随机投影矩阵进一步压缩可训练参数，NoRA 提出嵌套低秩结构 | 进一步降低适配器体积，理论上有利于在同一 GPU 上容纳更多适配器，但也带来 rank/结构异构性，需要 Serving 系统支持异构批处理 |

从 Serving 系统的角度看，这些变体共同带来的关键约束是：**适配器的 rank、目标模块（target modules）、量化格式可能各不相同**，这正是后文 S-LoRA、CaraServe、MixLoRA 等系统重点解决的"异构批处理"问题的根源。

---

## 4. LoRA Serving 的核心系统挑战

结合 S-LoRA、Punica、dLoRA、CaraServe、Chameleon 等论文的问题建模，可将挑战归纳为以下五类：

1. **Batching 失效问题**：若为每个适配器分别 merge 权重后推理，不同适配器的请求无法进入同一个 batch 一起做基座模型的矩阵乘法，GPU 利用率会随适配器数量增多而急剧下降；LMSYS 的 S-LoRA 博客明确指出，逐一切换加减 LoRA 权重的方案虽然能实现单适配器低延迟，但会显著降低并发场景下的整体吞吐并增加总延迟。
2. **显存管理与碎片化**：GPU 显存需要同时容纳基座模型权重、成百上千个 LoRA 适配器权重（rank 各异）、以及长度可变的 KV Cache，三者混合管理极易产生碎片化。
3. **冷启动 / 适配器加载延迟**：当请求命中的适配器不在 GPU 显存中时，需要从主机内存甚至磁盘加载，这个过程会显著拖慢首 token 生成时间（TTFT），CaraServe、Chameleon 等系统都将其列为核心痛点。
4. **异构 rank 下的算子效率**：不同适配器 rank 不同时，朴素的批量矩阵乘法（GEMM）效率低下；Punica 提出的 SGMV 算子要求同批适配器 rank 相同，rank 不同时会退化为串行处理，这也是后续 MixLoRA 等工作试图突破的限制。
5. **多副本 / 集群级负载均衡**：请求的输入输出长度高度可变，加上适配器的访问热度呈 Zipf 分布（少数适配器承接大部分流量），传统静态放置策略容易造成副本间负载不均，dLoRA、Punica 的调度器都专门处理这一问题。

---

## 5. 代表性技术路线与论文详解

### 5.1 朴素方案：Merge / Switch

最直接的做法是沿用 LoRA 论文建议的"训练后合并"策略：为每个到达的请求，先从内存取出对应 $(A,B)$，与 $W_0$ 相加得到临时的稠密权重，推理完成后再减去恢复。这种方式的优点是单请求延迟低、可复用现有推理引擎；缺点是：

- 无法实现跨适配器的请求批处理，吞吐随并发适配器数增加而快速下降；
- 频繁的加/减权重操作本身有开销，且当适配器数量超过显存容量时必须频繁换入换出。

HuggingFace PEFT 库的默认 LoRA 推理路径、以及早期 vLLM 对 LoRA 的"朴素支持"，都属于这一类，也是后续所有专用 Serving 系统的性能对比基线。

### 5.2 Punica：SGMV 批量算子（2023-10）

论文：*Punica: Multi-Tenant LoRA Serving*（Chen 等）

Punica 是较早系统性解决"多租户 LoRA 批处理"问题的工作，核心贡献是提出了一个新的 CUDA 算子——**Segmented Gather Matrix-Vector Multiplication（SGMV）**。

**基本思路**：与其为每个适配器单独合并权重，不如让 GPU 只保留一份基座模型权重，LoRA 的增量计算 $xAB$ 通过专用算子在运行时批量完成。SGMV 的做法是把同批请求按所属 LoRA 模型分段（segment），先并行计算各段的 $xA$，再计算 $\cdot B$，通过分组提升算子的运算强度（arithmetic intensity），从而更好地利用 GPU Tensor Core。

**关键设计**：
- 请求按 LoRA 模型分组批处理，组内使用 Tensor Core 加速，组间保持并行；
- 当每个请求对应不同 LoRA（组内只有一个请求）时，退化为纯粹的访存密集型（IO-bound）操作，Punica 为此专门设计了不使用 Tensor Core、以内存带宽利用率最大化为目标的调度路径；
- 系统层面的调度器按请求粒度做放置决策，并支持按迭代粒度在 GPU 间迁移旧请求，以合并负载、提升集群利用率。

**效果**：论文报告在固定规模 GPU 集群下，相比当时最先进的 LLM Serving 系统，Punica 在多 LoRA 场景下吞吐提升可达 12 倍，每 token 仅增加约 2ms 延迟。SGMV 算子后来被 vLLM、LoRAX 等系统直接借鉴或引用，是本领域公认的奠基性工作之一。

### 5.3 S-LoRA：Unified Paging 与异构批处理（2023-11，MLSys 2024）

论文：*S-LoRA: Serving Thousands of Concurrent LoRA Adapters*（Sheng 等，UC Berkeley / LMSYS 团队）

S-LoRA 把可服务的并发适配器规模从 Punica 的量级进一步推高到"数千个"，其贡献主要体现在三个层面：

**（1）Unified Paging（统一分页）显存管理**：借鉴 vLLM PagedAttention 对 KV Cache 的分页思想，S-LoRA 把 LoRA 适配器权重和 KV Cache 张量统一纳入同一个内存池中以页为单位管理，两者共享同一套分配/回收机制，从而显著减少显存碎片化，让更多适配器和更长的 KV Cache 能够共存于同一张 GPU。

**（2）分离批处理（Separated Batched Computation）**：不同于合并权重的方式，S-LoRA 明确地将基座模型的 $xW_0$ 计算与各请求的 LoRA 增量 $xAB$ 计算分离——前者作为一个统一的大 batch 高效执行，后者通过定制 CUDA 算子实现"异构批处理"（不同 rank、不同适配器的请求可以在同一次算子调用中处理）。虽然这引入了额外的 $xAB$ 计算开销，但由于该计算量远小于 $xW_0$，通过批处理基座计算节省的成本远超新增开销。

**（3）新的张量并行策略**：针对 LoRA 增加的小矩阵计算在多卡张量并行环境下通信开销占比过高的问题，S-LoRA 设计了专门的张量并行方案以降低通信量。

**存储分层**：S-LoRA 将全部适配器保存在主机内存（Host Memory）中，仅将当前批次实际用到的适配器取到 GPU 显存，通过预取等手段掩盖加载延迟；在显存不足时可以借助主机内存扩展可服务的适配器总量。

**效果**：相比 HuggingFace PEFT 和"朴素支持 LoRA 的 vLLM"，S-LoRA 吞吐最高提升 4 倍，可服务的适配器数量提升几个数量级；相比 PEFT，吞吐提升可达约 30 倍。代码已开源（S-LoRA/S-LoRA），并被后续的 Chameleon 等系统直接用作基础平台进行二次开发。

### 5.4 CaraServe：CPU 辅助的冷启动消除与 rank-aware 调度（2024-01）

论文：*CaraServe: CPU-Assisted and Rank-Aware LoRA Serving for Generative LLM Inference*

CaraServe 关注的是 S-LoRA/Punica 尚未充分解决的两个问题：**冷启动延迟**与**异构 rank 请求的调度公平性/SLO 达成率**。

**CPU 辅助消除冷启动**：当请求命中的适配器不在 GPU 上时，系统必须先把适配器权重从主机内存加载到显存，这段等待时间会直接推迟首 token 生成。CaraServe 的做法是：在把适配器权重加载到 GPU 的同时，**先用 CPU 对该适配器执行 prefill 阶段的部分计算**（因为适配器权重本就在主机内存中，CPU 可以立即开始计算，无需等待搬运）；等 GPU 加载完成后，再无缝切换到 GPU 完成剩余 prefill 与后续 decode，从而把"冷启动时间"与"CPU 侧计算时间"重叠掉。为此，CaraServe 设计了专门的 CUDA 算子配合异步内存拷贝和信号量机制，实现 CPU/GPU 两条计算路径的高效同步，将 LoRA 调用开销降到 1 毫秒以下。

**Rank-aware 调度**：在多租户场景下，不同用户的适配器 rank 往往不同，如果简单地把不同 rank 的请求混合批处理，会造成资源浪费或效率下降。CaraServe 引入了基于性能建模的 rank 感知调度算法，专门针对异构 rank 的批次进行优化编排，以提升 SLO（服务等级目标）达成率。

**效果**：论文报告 prefill 延迟最高降低约 1.4 倍，并可在保持约 99% SLO 达成率的同时完成异构负载调度。

### 5.5 dLoRA：动态 merge/unmerge 与跨副本迁移（OSDI 2024）

论文：*dLoRA: Dynamically Orchestrating Requests and Adapters for LoRA LLM Serving*（Wu 等，北京大学）

dLoRA 提出了与 S-LoRA"始终分离计算"不同的思路：它认为**是否合并权重应当动态决定**，而不是一刀切。其核心洞察有两点：

1. 当请求分布高度偏斜（例如某个适配器占绝对多数流量）时，把该适配器**合并（merge）**进基座模型、以纯粹的稠密矩阵乘法方式批处理，反而比始终保持分离计算更高效；只有在请求类型混合、多个适配器均衡出现时，"不合并 + 跨适配器批处理"才是更优选择。因此 dLoRA 支持在运行时动态地 merge / unmerge 适配器。
2. 由于 LLM 请求具有自回归特性，输入输出长度高度可变，即使请求被均匀分配到各个 worker 副本，副本间的实际负载也会出现明显不均衡。为此 dLoRA 支持**动态迁移**——把适配器及其未完成的请求从繁忙副本迁移到空闲副本。

**系统架构**：dLoRA 在副本内部采用"跨适配器批处理"（cross-adapter batching）技术处理不同适配器的请求；副本之间通过负载均衡器结合迁移机制解决负载不均问题；同时有专门的显存管理器动态调整适配器权重与请求中间状态（KV Cache 等）之间的显存分配比例。

**效果**：论文称相比同期的 S-LoRA，dLoRA 平均延迟最多降低 1.8 倍，凸显了"是否合并权重"这一决策本身也应当是自适应的，而非静态假设。

### 5.6 Chameleon：适配器缓存与多级队列调度（2024-11, MICRO 2025）

论文：*Chameleon: Adaptive Caching and Scheduling for Many-Adapter LLM Inference Environments*（UIUC / IBM Research）

Chameleon 在 S-LoRA 开源平台之上进一步引入两项机制，专门针对生产环境中真实存在的两个次生问题：

**（1）适配器缓存（Adapter Cache）**：作者观察到，许多真实生产环境下 GPU 显存并未被 KV Cache 和已加载适配器完全占满，存在"空闲显存"。Chameleon 设计了一个透明、自适应、无干扰的适配器缓存层，利用这部分空闲显存缓存高频访问的适配器权重，将其加载过程移出关键路径，减少 PCIe 带宽争用。论文还发现简单的 LRU 淘汰策略并不适合该场景（不同 rank 的适配器"缓存未命中"的代价不同），因此设计了结合访问近因性、频率与重新加载代价的**成本感知淘汰策略**。

**（2）多级队列调度器（Adapter-aware MLQ Scheduler）**：现有调度器（例如 Shortest-Job-First）在处理适配器异构场景时，容易让长请求因"饥饿避免机制"而拖累短请求的尾延迟。Chameleon 设计了非抢占式的多级队列调度器：将请求按预测长度分类进入不同队列，按队列优先级动态分配可接纳的 token 配额，并在某些队列请求不足时把空闲资源重新分配给其他队列，从而既给短请求提供"快速通道"，又保证长请求不被无限期饿死。

**效果**：基于真实生产 trace 的评测显示，相比 S-LoRA 基线，Chameleon 将 P99 首 token 延迟（TTFT）降低约 80.7%，P50 延迟降低约 48.1%，整体吞吐提升约 1.5 倍。

### 5.7 面向异构 rank 的算子优化（MixLoRA / Dynamic SGMV 等）

Punica 提出的 SGMV 算子有一个重要限制：**要求同一批次内所有请求的 LoRA rank 相同**，一旦 rank 不同，系统只能退化为串行处理，这在真实多租户场景（不同客户可能训练不同 rank 的适配器）中会显著拉低吞吐。

- **MixLoRA**（2025，ICPP）针对这一限制提出了改进的多租户框架，试图在异构 rank 条件下依然保持高效批处理，突破 SGMV 对同 rank 的硬性要求。
- **Dynamic Operator Optimization for Efficient Multi-Tenant LoRA Model Serving**（AAAI 2025）提出一种自动化的动态算子优化方法，根据具体上下文（batch 内 rank 分布、序列长度等）对 SGMV 算子进行自适应调整，在多种批大小和请求模式下取得比原始 SGMV 实现更优的延迟表现。

这类工作代表了 LoRA Serving 领域从"系统架构层面"进一步深入到"底层算子层面"的优化趋势。

### 5.8 工业界实现：vLLM、LoRAX（Predibase）、TGI / Triton

学术论文中的核心思想（SGMV、统一分页、动态调度等）已经较为完整地落地到主流开源推理框架中：

**vLLM**：自 v0.3.0 起引入 Multi-LoRA 支持，直接借鉴了 Punica 的 SGMV 核函数设计；使用方式上，可通过 `--enable-lora` 启动参数在服务端加载多个适配器，并通过请求级别的 `LoRARequest` 指定使用哪个适配器；vLLM 还支持运行时动态加载/卸载适配器（`/v1/load_lora_adapter` 接口）、限定 LoRA 作用的目标模块（`--lora-target-modules`）等能力。2026 年初，vLLM 社区与云厂商合作进一步扩展了对 MoE 架构模型（如 GPT-OSS、Qwen3 MoE、DeepSeek、Llama MoE 系列）的 Multi-LoRA 支持，针对 MoE 场景下 LoRA 核函数的网格维度和稀疏性问题做了专门的算子调优。Triton Inference Server 也基于 vLLM backend 提供了 Multi-LoRA 部署教程。

**LoRAX（Predibase）**：一个专门面向"单 GPU 服务上千个微调模型"场景的开源框架（LoRA eXchange），构建在 HuggingFace TGI（text-generation-inference）之上，并借鉴了 Punica 的 SGMV 核函数。其核心能力包括：
- **动态适配器加载（Dynamic Adapter Loading）**：请求中直接指定 HuggingFace/Predibase/文件系统上的任意 LoRA 适配器路径，即时（just-in-time）加载而不阻塞其他并发请求；
- **异构连续批处理（Heterogeneous Continuous Batching）**：将来自不同适配器的请求打包进同一批次，使延迟和吞吐几乎不随并发适配器数量增长而显著下降；
- **分层权重缓存（Tiered Weight Caching）**：在 GPU 显存和 CPU/磁盘之间异步预取和卸载适配器权重，配合批处理调度优化系统整体吞吐；
- 支持在 AWQ 4-bit 量化基座模型之上挂载多个 LoRA 适配器，将量化与多租户 LoRA 服务结合。

Predibase 团队在《LoRA Land: 310 Fine-tuned LLMs that Rival GPT-4》技术报告中，用 LoRAX 在单张 A100（80GB）上同时服务 25 个基于 Mistral-7B 微调的 LoRA 模型，展示了多 LoRA 服务在实际业务中相较单一通用大模型的性价比优势；公开案例显示，某呼叫中心分析公司使用该方案同时服务 60 余个适配器，平均响应时间仍保持在 2 秒以内。

**其他工程实践**：Cloudflare Workers AI 等云服务商也基于 Punica 的 SGMV 思路构建了自己的 LoRA 微调模型推理服务；HuggingFace PEFT 库则代表了未经过专门多租户优化、更偏训练/单模型推理场景的对照基线。

---

## 6. 技术方案对比总结

| 系统 / 方案 | 发表时间 | 核心技术 | 主要解决的问题 | 局限性 |
|---|---|---|---|---|
| 朴素 Merge/Switch（PEFT 默认） | - | 训练后合并权重，逐适配器串行推理 | 单适配器场景延迟最低 | 无法跨适配器批处理，并发吞吐差 |
| **Punica** | 2023-10 | SGMV 批量算子 | 让不同 LoRA 请求共享基座权重并批处理 | 要求同批 rank 相同，rank 异构时退化为串行 |
| **S-LoRA** | 2023-11 | Unified Paging + 分离批处理 + 定制张量并行 | 单机/多机服务上千级适配器，显存碎片化 | 未充分利用主机侧空闲计算资源掩盖冷启动 |
| **CaraServe** | 2024-01 | CPU 辅助 prefill + rank-aware 调度 | 冷启动延迟、异构 rank 的 SLO 达成 | 依赖 CPU 算力，对 CPU 较弱的机型收益有限 |
| **dLoRA** | 2024-07（OSDI） | 动态 merge/unmerge + 跨副本迁移 | 是否合并权重的自适应决策、副本间负载不均 | 决策与迁移引入额外调度复杂度 |
| **Chameleon** | 2024-11 | 适配器缓存 + 多级队列调度 | 利用空闲显存降低加载延迟、缓解长尾延迟 | 构建于 S-LoRA 之上，非独立底层引擎 |
| **MixLoRA / Dynamic SGMV** | 2025 | 异构 rank 批处理算子优化 | 突破 SGMV 同 rank 限制 | 仍处于较新阶段，工业落地有限 |
| **vLLM Multi-LoRA** | 持续演进 | 集成 SGMV 内核 + PagedAttention 生态 | 生产级、社区驱动的通用方案，已扩展至 MoE 模型 | 高级调度（迁移、缓存分层）不如专用研究系统精细 |
| **LoRAX（Predibase）** | 2023-11 起 | 动态加载 + 异构连续批处理 + 分层缓存 + 量化基座 | 商业化、开箱即用的千级适配器服务 | 生态相对独立，与 vLLM 社区演进路径有所不同 |

---

## 7. 系统设计要点提炼

综合上述论文与工程实践，一个成熟的 LoRA Serving 系统通常需要在以下几个维度做出设计决策：

1. **计算范式**：是否始终保持"基座计算与 LoRA 增量分离"（S-LoRA 路线），还是允许动态 merge/unmerge（dLoRA 路线）。二者的取舍取决于请求分布是否偏斜。
2. **显存管理**：是否将适配器权重与 KV Cache 纳入统一的分页式内存池管理（S-LoRA 的 Unified Paging），以及是否引入独立的适配器缓存层（Chameleon）来利用空闲显存。
3. **冷启动优化**：是否通过 CPU/GPU 协同计算掩盖适配器加载延迟（CaraServe），或通过预测性预取降低命中失败概率。
4. **批处理算子**：核心是 SGMV 及其变体，需要考虑同 rank/异构 rank 场景下的效率差异，以及是否要为 IO-bound（小 batch、多适配器）和 compute-bound（大 batch、少适配器）场景分别设计执行路径。
5. **调度策略**：请求级 / 迭代级调度粒度，是否支持跨 GPU 副本的适配器与请求迁移，以及如何在保证短请求低延迟的同时避免长请求"饥饿"。
6. **与基座模型量化的协同**：是否支持在 4-bit/8-bit 量化基座模型上挂载全精度或量化后的 LoRA 适配器（QLoRA 思路在 Serving 侧的延伸），以在显存与精度之间取得平衡。

---

## 8. 未来趋势与开放问题

1. **异构 rank / 异构结构适配器的高效批处理**：目前主流 SGMV 类算子仍偏好同 rank 假设，MixLoRA、Dynamic Operator Optimization 等工作显示这是一个活跃的研究方向。
2. **MoE 架构与多 LoRA 的结合**：随着 MoE 大模型（如 DeepSeek、Qwen3 MoE、GPT-OSS 等）成为主流，如何为 MoE 结构设计原生的多 LoRA 批处理核函数（如 vLLM 近期新增的 fused_moe_lora 系列算子）成为新的工程重点。
3. **缓存与调度的联合优化**：Chameleon 展示了适配器缓存与请求调度联合设计可以带来显著的尾延迟改善，未来更细粒度的预测性预取（predictive prefetching）、跨节点缓存共享等方向仍有较大空间。
4. **训练侧与服务侧的协同设计**：更多变体（如 VeRA、NoRA、LoRA+ 等）在压缩适配器体积或提升训练效果的同时，也在客观上为 Serving 系统提供了新的优化空间（例如更小的适配器意味着可缓存的适配器数量更多），训练方法与服务系统的协同优化尚有较大潜力。
5. **多机、多集群级别的全局调度**：当前大部分系统仍以单机/单 GPU 集群为主要场景，面向跨地域、跨集群的全局 LoRA 路由与负载均衡研究相对较少。

---

## 9. 参考资料

**核心论文**
- Hu, E. J. et al. *LoRA: Low-Rank Adaptation of Large Language Models*. arXiv:2106.09685, 2021.
- Dettmers, T. et al. *QLoRA: Efficient Finetuning of Quantized LLMs*. arXiv:2305.14314, 2023.
- Chen, L. et al. *Punica: Multi-Tenant LoRA Serving*. arXiv:2310.18547, 2023.
- Sheng, Y. et al. *S-LoRA: Serving Thousands of Concurrent LoRA Adapters*. arXiv:2311.03285, MLSys 2024.
- Li, S. et al. *CaraServe: CPU-Assisted and Rank-Aware LoRA Serving for Generative LLM Inference*. arXiv:2401.11240, 2024.
- Wu, B. et al. *dLoRA: Dynamically Orchestrating Requests and Adapters for LoRA LLM Serving*. OSDI 2024.
- Iliakopoulou, N. et al. *Chameleon: Adaptive Caching and Scheduling for Many-Adapter LLM Inference Environments*. arXiv:2411.17741, MICRO 2025.
- Hayou, S. et al. *LoRA+: Efficient Low Rank Adaptation of Large Models*. arXiv:2402.12354, 2024.
- Wu, T. et al. *Mixture-of-Subspaces in Low-Rank Adaptation (MoSLoRA)*. arXiv:2406.11909, 2024.
- Zhao, J. et al. *LoRA Land: 310 Fine-tuned LLMs that Rival GPT-4, A Technical Report*. arXiv:2405.00732, 2024.
- MixLoRA: An Efficient Multi-Tenant Framework for Concurrently Serving Diverse LoRA Models. ICPP 2025.
- Dynamic Operator Optimization for Efficient Multi-Tenant LoRA Model Serving. AAAI 2025.

**工程与产品文档**
- LMSYS Org Blog: *Recipe for Serving Thousands of Concurrent LoRA Adapters* (S-LoRA 介绍), 2023.
- vLLM 官方文档：LoRA Adapters / MultiLoRA Inference。
- vLLM Blog: *Efficiently serve dozens of fine-tuned models with vLLM on Amazon SageMaker AI and Amazon Bedrock*, 2026.
- GitHub: predibase/lorax（LoRAX 项目）。
- Predibase Blog: *LoRAX: The Open Source Framework for Serving 100s of Fine-Tuned LLMs*。
- Cloudflare Blog: *Running fine-tuned models on Workers AI with LoRAs*。
- NVIDIA Triton Inference Server 文档：Multi-LoRA vLLM Backend 教程。
- GitHub: S-LoRA/S-LoRA、LLMServe/dLoRA-artifact。