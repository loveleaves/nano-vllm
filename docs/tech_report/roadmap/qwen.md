# Qwen（通义千问）系列大模型技术深度分析报告

> **报告版本**：2026年7月（更新版）
> **研究范围**：Qwen1 → Qwen2 → Qwen2.5 → Qwen3 → Qwen3.5 → Qwen3.6 → Qwen3.7 全系列，含多模态、专域及推理模型
> **数据来源**：官方技术报告、arXiv论文、Alibaba Cloud博客、Hugging Face模型卡、Wikipedia、路透社/CNBC/南华早报等新闻报道
> **本次更新说明**：在原报告（截至2026年6月）基础上，补充了Qwen3.5（原生多模态、Gated Delta Networks混合架构）、Qwen3.6、Qwen3.7-Plus的技术细节，以及阿里AI组织架构重组（Token Hub成立）、核心技术负责人变动等最新动态

---

## 报告概览

这份深度技术报告覆盖了Qwen系列从2023年至2026年7月的完整技术全景，共18个章节：

**核心发现亮点：**

**版本演进轴**：从Qwen1（~3T token预训练）→ Qwen2（~7T）→ Qwen2.5（18T token）→ Qwen3（36T token，119种语言）→ **Qwen3.5（原生多模态，201种语言，架构切换为Gated Delta Networks混合线性注意力）**，每一代在数据规模、架构效率与多语言覆盖上持续突破。

**最大架构创新（Qwen3→Qwen3.5演进）**：Qwen3将思维模式与非思维模式统一到同一框架；**Qwen3.5则在此基础上进一步升级为"Gated DeltaNet（线性注意力）+ Gated Attention + 稀疏MoE"的混合架构（源自Qwen3-Next），并首次实现"早期融合（Early Fusion）"的原生多模态预训练**，即模型从零开始在文本、图像、视频交织的token流上训练，而非后期拼接视觉编码器。

**MoE旗舰迭代**：Qwen3旗舰模型Qwen3-235B-A22B（235B总参数，22B激活）→ **Qwen3.5-397B-A17B（397B总参数，17B激活，512专家中10路由+1共享）**，参数量增加但激活参数减少，配合混合线性注意力，在32K/256K上下文下解码吞吐量分别达到Qwen3-Max的**8.6倍/19倍**。

**推理与Agent能力持续进化**：Qwen3-235B-A22B在AIME'24达85.7分、AIME'25达81.5分；**Qwen3.7-Plus进一步引入"深度推理、自主编程、工具调用、验证测试、自主迭代"五大能力，标志Qwen系列正式从"回答问题"转向"自主执行任务"的Agent范式**。

**组织与生态重大变化**：2026年3月，Qwen技术负责人林俊旸（Lin Junyang）离职，阿里随即成立跨部门的"**Token Hub**"（通义大模型事业群），由CEO吴泳铭直接统领，整合通义大模型、MaaS、Qwen、悟空、AI Innovation等业务线，标志阿里将AI提升至集团最高战略优先级。

---

## 1. 项目概况与战略定位

### 1.1 项目背景

Qwen（通义千问）是阿里巴巴云智能集团（Alibaba Cloud）自主研发的大语言模型系列，名称源自中文"千问"，寓意能够回应广泛的用户查询需求。该项目于2023年4月以"通义千问"名义内测启动，同年9月通过监管审批后正式向公众开放，是阿里巴巴集团在人工智能领域的核心战略布局。

Qwen的诞生背景与全球大模型竞赛直接相关：ChatGPT的出现（2022年末）激发了全球对LLM的热情，LLaMA系列的开源进一步燃起了开源社区的参与热情（Qwen1的早期架构即借鉴了Meta LLaMA的设计）。在此背景下，Qwen定位于：

- **中文核心能力的技术领导者**：在中英双语及多语言处理上具备世界级竞争力
- **开源社区的重要贡献者**：以开放权重模型的形式持续向学术界和产业界输出，Hugging Face上Qwen系列衍生模型已超过20万个变体
- **阿里云AI服务的基础底座**：支撑阿里巴巴全系生态（电商、金融、云服务等）
- **（2026年新增）AI Agent与消费生态入口**：2026年1月起，Qwen App开始与阿里生态（饿了么、淘宝、飞猪等）打通，向"能替用户跑腿办事"的超级助手方向演进

### 1.2 战略定位

| 维度 | 定位描述 |
|------|----------|
| 技术路线 | 自主研发，全栈闭环（数据→架构→训练→推理→部署） |
| 开源策略 | 核心版本开放权重，旗舰/最新版本转向"闭源优先"（详见12.4节） |
| 竞争对标 | GPT-5（OpenAI）、Claude Opus（Anthropic）、Gemini 2.5/3（Google）、DeepSeek、Kimi（Moonshot）、GLM（Zhipu） |
| 应用场景 | 企业AI、开发者工具、研究用途、端侧部署、消费级Agent应用 |
| 全球化布局 | 从中英双语扩展至119种语言（Qwen3）→ **201种语言/方言（Qwen3.5）** |
| 组织架构 | 2026年3月起纳入阿里新设的"Token Hub"（通义大模型事业群），由CEO吴泳铭统领 |

---

## 2. 版本演进路线图

### 2.1 主干版本时间线（更新至2026年7月）

```
2023年4月    Tongyi Qianwen（通义千问）内测启动
2023年8月    Qwen-VL（首个视觉语言模型）
2023年9月    Qwen-1.0（首个公开LLM，7B/14B，通过监管审批公开）
2023年11月   Qwen-1.8B / Qwen-72B（超小与旗舰型号扩充）
2024年2月    Qwen1.5（全面升级，0.5B～110B）
2024年3月    Qwen1.5-MoE（首个MoE模型，A2.7B）
2024年6月    Qwen2（重大架构升级，0.5B～72B）
2024年9月    Qwen2.5（18万亿token预训练，0.5B～72B，含Coder/Math专域版）
2024年11月   QwQ-32B-Preview（首个推理专用模型）
2025年1月    Qwen2.5-Max（闭源旗舰）、Qwen2.5-VL（视觉语言旗舰升级）
2025年1月    Qwen2.5-1M（百万token超长上下文）
2025年3月    QwQ-32B（正式版，强化学习加强推理）、Qwen2.5-Omni（全模态7B/3B）
2025年4月    Qwen3（36万亿token，统一思维框架，0.6B～235B，Apache 2.0全系开源）
2025年7月    Qwen3-Coder（480B-A35B旗舰编程模型，SWE-bench对标Claude Sonnet 4）
2025年9月    Qwen3-Next-80B-A3B（首个Gated DeltaNet混合注意力架构，256K原生上下文）
2025年9月    Qwen3-Max（超1T参数闭源旗舰）
2025年9月    Qwen3-VL（视觉语言MoE旗舰，235B）、Qwen3-Omni（全模态：文本+图像+音频+视频）
2026年2月    Qwen3-Coder-Next（小尺寸混合架构编程模型）
2026年2月    **Qwen3.5发布**（397B-A17B，原生多模态早期融合，201种语言，Gated Delta Networks架构）
2026年3月    **林俊旸（Qwen技术负责人）离职**；阿里成立"Token Hub"事业群
2026年4月    Qwen3.5全尺寸系列（0.8B～122B-A10B）、Qwen3.5-Omni（闭源）
2026年4月    **Qwen3.6发布**（Qwen3.6-Plus闭源、Qwen3.6-35B-A3B开源，聚焦真实世界Agent）
2026年6月    **Qwen3.7-Plus发布**（多模态Agent，深度推理+自主编程+工具调用+自主迭代五大能力）
```

### 2.2 关键版本技术跨越对比

| 版本 | 预训练数据 | 最大参数量 | 语言覆盖 | 上下文长度 | 重要特性 |
|------|-----------|-----------|---------|-----------|---------|
| Qwen1 | ~3T | 72B | 中/英为主 | 8K | 基础LLM能力 |
| Qwen2 | ~7T | 72B | 29种语言 | 128K | GQA、YARN、DCA |
| Qwen2.5 | 18T | 72B（开源）/Max（闭源） | 29种语言 | 1M（Turbo） | DPO+GRPO后训练 |
| Qwen3 | 36T | 235B（A22B激活）/Max超1T | 119种语言 | 128K/256K+ | 统一思维模式、MoE |
| **Qwen3-Next** | ~15T（增量训练） | 80B（A3B激活） | 沿用Qwen3 | 262K原生/1M+外推 | 首个Gated DeltaNet混合注意力，10倍推理吞吐 |
| **Qwen3.5** | 增量多模态预训练 | 397B（A17B激活） | **201种语言** | 262K原生/1.01M外推 | 原生多模态早期融合，250K词表，混合线性注意力 |
| **Qwen3.6** | 增量Agent强化 | 35B-A3B（开源旗舰） | 沿用Qwen3.5 | 262K+ | 聚焦真实世界Agent、思维保留（Thinking Preservation） |
| **Qwen3.7** | 增量多模态Agent强化 | Max/Plus（闭源） | 沿用Qwen3.5 | 262K+ | 深度推理+自主编程+验证测试+自主迭代 |

---

## 3. 核心架构技术

### 3.1 基础网络架构

Qwen全系列基于**Transformer Decoder-Only**架构起步，采用自回归的下一词预测（Next-Token Prediction）训练目标；**自Qwen3-Next（2025年9月）起，架构演进为"混合注意力"（线性注意力 + 标准注意力交替堆叠）**，这是Qwen系列迄今最大的底层架构变革。以下为各关键组件的技术选型与演进：

#### 3.1.1 注意力机制

**Grouped Query Attention（GQA）**

从Qwen2开始全面引入GQA，相较于标准多头注意力（MHA）和多查询注意力（MQA），GQA在KV Cache效率与模型表达能力之间取得最优平衡：

- Query head数量保持与MHA一致（如72B模型64个Q头）
- Key/Value head数量显著减少（如8个KV头）
- KV Cache显存占用降低8倍，推理吞吐量大幅提升
- 性能损失可忽略不计，是工程落地的最优选择

**QK-Norm（Qwen3引入）**

Qwen3在注意力机制中引入QK-Norm，对Query和Key向量在内积之前进行LayerNorm归一化，显著提升超大规模训练的数值稳定性。同时移除了Qwen2中的QKV-Bias，进一步简化模型结构。

**Gated DeltaNet + Gated Attention 混合架构（Qwen3-Next / Qwen3.5起，架构重大升级）**

这是Qwen系列自GQA之后最重要的架构创新，首次在Qwen3-Next-80B-A3B中引入，并成为Qwen3.5、Qwen3.6、Qwen3.7的底层基础：

- **设计动机**：标准Transformer注意力对序列中每一对token都要计算相似度，复杂度为O(n²)，长上下文下KV Cache显存和计算开销急剧膨胀
- **Gated DeltaNet（GDN，线性注意力）**：每层维护一个**固定尺寸**（与head维度平方成正比，如128×128）的状态矩阵，而非随序列增长的注意力图；新token通过"Delta Rule"（误差纠正式更新机制）增量更新该状态，推理时直接查询当前状态得到输出。计算复杂度降为O(n·d²)，即相对序列长度线性、仅在固定的小head维度上为二次
- **Gated Attention（标准注意力，少量保留）**：混合架构并非完全弃用标准注意力，而是按固定比例交替堆叠。以Qwen3.5-397B-A17B为例，其60层结构为：**15个模块 × [3×(Gated DeltaNet → MoE) → 1×(Gated Attention → MoE)]**，即每4层中3层为线性注意力、1层为标准全局注意力，兼顾长程依赖捕捉能力与推理效率
- **注意力输出门控机制**：帮助消除"注意力汇聚"（Attention Sink）和"异常激活值"（Massive Activation）问题，配合Zero-Centered、带权重衰减的RMSNorm，进一步提升超大规模训练的数值稳定性
- **实测收益**：Qwen3-Next-80B-A3B相比Qwen3-32B训练成本降低约90%，长上下文推理吞吐提升约10倍；Qwen3.5-397B-A17B在32K/256K上下文下解码吞吐量分别达到Qwen3-Max的8.6倍/19倍，部署显存占用降低约60%
- **多token预测（MTP，Multi-Token Prediction）**：同步引入，进一步提升预训练效果与推理速度

> **技术小结**：GQA解决的是"KV Cache太大"，Gated DeltaNet混合架构解决的是"注意力计算本身随长度平方增长"，二者叠加使Qwen在超长上下文（256K原生、1M+外推）场景下的工程可用性大幅提升，是Qwen3.5系列吞吐量数量级提升的核心原因。

#### 3.1.2 位置编码

**Rotary Positional Embedding（RoPE）**

Qwen系列全面采用RoPE进行位置信息编码，主要优势：
- 无需额外位置参数，通过旋转矩阵将相对位置融入注意力计算
- 天然支持长度外推（Length Extrapolation）
- 配合ABF技术调整RoPE基频（从10,000扩展至1,000,000+），支持超长上下文

**多模态分支的MRoPE变体**

视觉语言模型（Qwen-VL系列）引入Multimodal RoPE（MRoPE），为文本、图像块、视频帧分别分配独立的位置维度，支持任意分辨率输入下的位置表示。Qwen3-Omni进一步演进为Time-aligned Multimodal RoPE（TM-RoPE），实现音视频流的精确时间对齐；该机制延续至Qwen3.5-Omni。

#### 3.1.3 前馈网络

采用**SwiGLU**激活函数（Swish + Gated Linear Unit），相比ReLU/GELU提供更好的非线性表达能力，已成为主流LLM的标准选择。

前馈网络隐层维度通常为模型维度的约2.67倍（采用2/3系数以与标准FFN参数量对齐）。

#### 3.1.4 归一化

采用**RMSNorm**（Root Mean Square Normalization）并以**Pre-Normalization**方式部署（在子层输入处归一化，而非输出处）。Qwen3-Next起进一步引入**Zero-Centered、带权重衰减的LayerNorm**变体，专门用于修复混合注意力架构下部分层归一化权重异常增大的问题，提升训练稳定性。

#### 3.1.5 词表与分词器

- 采用**Byte-Level BPE**（字节级字节对编码）分词器
- 词表大小：151,643（Qwen2/2.5）→ 151,669（Qwen3）→ **248,320（Qwen3.5，约250K）**
- **Qwen3.5词表大幅扩充的原因**：语言覆盖从119种扩展至201种，扩大后的词表使多数语言的编码/解码效率提升10%～60%
- 专门优化中文分词颗粒度，支持完整的Unicode字符覆盖
- 特殊token包含完备的功能标记（tool call、think tags等）

### 3.2 模型规格矩阵

#### Qwen3 Dense模型

| 模型 | 参数量 | 层数 | 隐层维度 | Q头/KV头 |
|------|--------|------|---------|---------|
| Qwen3-0.6B | 0.6B | 28 | 1024 | 16/8 |
| Qwen3-1.7B | 1.7B | 28 | 2048 | 16/8 |
| Qwen3-4B | 4B | 36 | 2560 | 32/8 |
| Qwen3-8B | 8B | 36 | 4096 | 32/8 |
| Qwen3-14B | 14B | 40 | 5120 | 40/8 |
| Qwen3-32B | 32B | 64 | 5120 | 64/8 |

#### Qwen3 MoE模型

| 模型 | 总参数 | 激活参数 | 专家数 | 激活专家数 |
|------|--------|---------|--------|---------|
| Qwen3-30B-A3B | 30B | 3B | 128 | 8 |
| Qwen3-235B-A22B | 235B | 22B | 128 | 8 |

#### Qwen3-Next / Qwen3.5 混合MoE模型（新增）

| 模型 | 总参数 | 激活参数 | 专家配置 | 层结构 | 上下文 |
|------|--------|---------|---------|--------|--------|
| Qwen3-Next-80B-A3B | 80B | ~3.9B | 512专家（10路由+1共享） | Gated DeltaNet + Gated Attention 混合 | 262,144原生 |
| Qwen3.5-0.8B/2B/4B/9B | 0.8B～9B | 稠密 | — | 混合架构下探至端侧 | 262,144原生（全系列） |
| Qwen3.5-27B | 27B | 稠密 | — | 强指令遵循 | 262,144原生 |
| Qwen3.5-35B-A3B | 35B | 3B | 精简MoE | 平衡质量与速度 | 262,144原生 |
| Qwen3.5-122B-A10B | 122B | 10B | 中型MoE | 复杂分析场景 | 262,144原生 |
| **Qwen3.5-397B-A17B（旗舰）** | **397B** | **17B** | **512专家（10路由+1共享）** | 60层：15×[3×(GDN→MoE)→1×(Attention→MoE)] | 262,144原生／约1,010,000外推 |

---

## 4. 预训练技术体系

### 4.1 数据规模演进

预训练数据规模是Qwen系列最显著的迭代轴：

```
Qwen1     : ~3 万亿 tokens
Qwen2     : ~7 万亿 tokens
Qwen2.5   : 18 万亿 tokens  (+157%)
Qwen3     : 36 万亿 tokens  (+100%，覆盖语言从29→119种)
Qwen3-Next: ~15 万亿 tokens（增量预训练，验证混合架构有效性）
Qwen3.5   : 万亿级多模态交织token（早期融合，文本/图像/视频统一训练，语言从119→201种）
```

### 4.2 数据构成与质量控制

Qwen3预训练语料的多维度构成：

**数据来源类型**：
- **网络文本**：经多轮质量过滤的CommonCrawl等网页数据
- **文档解析**：使用Qwen2.5-VL模型对PDF等结构化文档进行OCR与文本提取，再经Qwen2.5精炼质量
- **代码数据**：使用Qwen2.5-Coder生成的合成代码数据，包含代码片段、注释、测试用例
- **数学数据**：使用Qwen2.5-Math生成的合成数学内容，包含教材、题目问答对
- **多语言数据**：Qwen3覆盖119种语言和方言；**Qwen3.5进一步扩大至201种语言和方言**
- **书籍与学术**：高质量的书籍、论文等长篇专业内容

**数据质量流程**：
- 多轮去重（MinHash LSH、精确去重）
- 领域分类与配比调节
- 毒性、隐私、版权过滤
- 以现有Qwen模型自举生成合成数据（Self-play数据飞轮）

### 4.3 三阶段预训练策略（Qwen3）

Qwen3采用精心设计的**三阶段渐进式预训练**：

**阶段1：通用能力建立（S1）**
- 数据规模：约30万亿token
- 上下文长度：4,096 tokens
- 目标：建立广博的语言能力和世界知识基础

**阶段2：推理能力强化（S2）**
- 数据规模：约5万亿token（增量）
- 上下文长度：4,096 tokens
- 目标：显著增强STEM、编程、逻辑推理能力
- 使用加速的学习率衰减策略

**阶段3：长上下文适配（S3）**
- 数据规模：数千亿token（增量）
- 上下文长度：从4K扩展至32,768 tokens
- 目标：使模型具备处理长文档的能力

### 4.4 Qwen3.5的"早期融合"原生多模态预训练（重大新增）

与Qwen-VL/Qwen2-VL/Qwen3-VL"先训练文本LLM、再拼接视觉编码器"的**后期融合（Late Fusion）**路线不同，Qwen3.5首次采用**早期融合（Early Fusion）**范式：

- **训练目标**：从预训练起点开始，就在文本、图像、视频交织的多模态token流上联合训练，而非先建立纯文本能力再"外挂"视觉模块
- **异构基础设施**：Qwen3.5通过解耦视觉与语言组件的并行策略（Heterogeneous Infrastructure），避免统一并行方案带来的效率损失，使超大规模原生多模态训练在工程上可行
- **效果**：Qwen3.5-397B-A17B在跨代际对比中，与纯文本的Qwen3取得相当能力的同时，在视觉理解、视频分析、Agent操作UI等任务上全面超越"后期融合"路线的Qwen3-VL同等规模版本
- **行业意义**：这一转变呼应了GPT-5、Gemini系列的技术路线，标志"多模态原生"正取代"文本优先+多模态适配"成为下一代旗舰模型的默认范式

### 4.5 扩展律与超参数优化

Qwen3在预训练中系统性研究了**扩展律（Scaling Laws）**：通过在三阶段预训练全流程上拟合扩展律，预测最优学习率和批次大小；对Dense和MoE模型分别建立扩展律模型，显著减少超参数搜索的计算开销。

### 4.6 长上下文预训练演进

- **Qwen2.5-Turbo**：四阶段渐进式上下文扩展（32K→64K→128K→256K），Qwen2.5-1M单独发布，通过YARN扩展至100万token
- **Qwen3-Next / Qwen3.5**：得益于Gated DeltaNet混合架构，原生上下文即达262,144 tokens，无需额外分阶段扩展训练即可外推至约1,010,000 tokens

---

## 5. 后训练与对齐技术

### 5.1 后训练数据规模

Qwen2.5后训练数据总量达**100万样本**，覆盖三个阶段：监督微调（SFT）、直接偏好优化（DPO）、群体相对策略优化（GRPO）。

### 5.2 监督微调（SFT）

- 涵盖单轮和多轮指令跟随数据
- 数据类型：通用问答、工具调用、代码执行、结构化输出
- 质量控制：拒绝采样（Rejection Sampling）过滤低质量样本
- 执行反馈：代码执行结果作为正确性信号

### 5.3 离线强化学习：DPO

**Direct Preference Optimization**作为离线RL阶段：使用SFT模型对查询集进行重新采样，通过质量检查的响应为正例，未通过的为负例，采用标准DPO损失优化策略，无需显式奖励模型。

### 5.4 在线强化学习：GRPO

**Group Relative Policy Optimization（GRPO）**是Qwen2.5+系列最核心的RL创新：

**算法原理**：
- 对每个查询采样G个响应（通常G=8）
- 通过组内相对奖励估计优势函数（无需单独的Critic网络）
- 优势函数 $\hat{A}_{g,i} = \tilde{r}_g = \frac{r_g - \text{mean}(\{r_1,...,r_G\})}{\text{std}(\{r_1,...,r_G\})}$
- 比PPO节省约50%显存，训练更稳定

**奖励建模**：使用训练好的奖励模型（RM）评分；数学/代码任务使用基于规则的奖励；多维度奖励融合：回答质量+格式合规+安全合规。

### 5.5 Qwen3四阶段后训练

Qwen3将后训练升级为精心设计的**四阶段流程**，直接催生了"统一思维模式"能力：

**Stage 1：CoT冷启动** — 使用高质量长链思维数据进行SFT
**Stage 2：推理强化RL** — 采用DAPO（Decoupled RLHF with Advantage and Policy Optimization），专注数学、代码、科学问题
**Stage 3：思维模式融合** — 混合思维/非思维数据联合训练，实现`/think`与`/no_think`动态切换
**Stage 4：通用RL** — 全面提升指令跟随、格式控制、工具使用等通用能力

### 5.6 GSPO：混合架构下的RL训练稳定性方案（新增）

Qwen3-Next及后续Qwen3.5/3.6引入**GSPO（Group Sequence Policy Optimization）**，专门解决"混合注意力（线性+标准）+ 高稀疏度MoE"组合在RL训练中的稳定性与效率挑战。相比GRPO，GSPO在序列层面而非token层面进行策略优化，缓解了混合架构下奖励信号在超长上下文中传播不稳定的问题，是Qwen3-Next-80B-A3B-Thinking能够在多项基准上超越同级别Gemini-2.5-Flash-Thinking的关键训练技术之一。

### 5.7 Qwen3.5的"百万级Agent环境"强化学习（新增）

Qwen3.5及以后版本的后训练引入**规模化的Agent环境强化学习**：在数百万个复杂度递增的任务分布上进行RL训练，目标是让模型在真实世界工具调用、多步骤任务执行中具备更强的泛化能力，而非仅在静态问答基准上优化。这一训练范式是Qwen3.6/3.7强调"真实世界Agent"能力的直接基础。

### 5.8 思维保留（Thinking Preservation，Qwen3.6新特性）

Qwen3.6引入"思维保留"机制：在多轮对话/迭代开发场景中，模型可以跨对话历史保留此前的思维链上下文，避免每轮都重新展开完整推理过程，从而降低迭代开发中的计算开销，并使多轮协作更连贯。**需要注意的是，Qwen3.5起官方已不再支持Qwen3时代的`/think`与`/no_think`软切换标记**，模式控制方式发生了变化，开发者需查阅对应版本文档确认最新调用方式。

---

## 6. 推理能力与思维模式

### 6.1 推理模型发展历程

| 模型 | 发布时间 | 特点 |
|------|----------|------|
| QwQ-32B-Preview | 2024年11月 | 首个Qwen推理专用模型，独立模型 |
| QwQ-32B | 2025年3月 | 正式版，强化学习加强推理 |
| QVQ-72B-Preview | 2024年12月 | 视觉推理专用模型 |
| Qwen3全系 | 2025年4月 | 将推理能力集成至统一模型框架 |
| Qwen3-Next-Thinking | 2025年9月 | 混合架构下的高效推理，超越Gemini-2.5-Flash-Thinking |
| Qwen3.7-Plus | 2026年6月 | 引入"深度推理"作为多模态Agent五大能力之一 |

### 6.2 统一思维框架（Qwen3核心创新）

Qwen3最重要的架构创新是将**思维模式**和**非思维模式**统一在同一模型中，无需在不同模型之间切换。

**工作原理**：
- **思维模式（Thinking Mode）**：模型在`<think>`...`</think>`标签内进行显式的链式推理，再给出最终答案
- **非思维模式（Non-thinking Mode）**：直接生成响应，适合简单查询和对话场景
- **动态切换**：Qwen3时代通过`/think`或`/no_think`系统提示切换模式；**Qwen3.5起该软切换标记不再officially支持**，转而通过独立的Instruct/Thinking模型变体或新的调用参数控制（详见各版本模型卡）

**思维预算（Thinking Budget）**：用户可精细控制推理token数量的上限，研究表明增加思维预算在多数任务上带来单调提升，实现推理深度与计算成本之间的灵活权衡。

### 6.3 Qwen3旗舰模型推理性能

Qwen3-235B-A22B在主要推理基准上的成绩：AIME'24 **85.7分**，AIME'25 **81.5分**，LiveCodeBench v5 **70.7分**，CodeForces Rating **2,056**，BFCL v3（工具调用）**70.8分**。

### 6.4 Qwen3.7-Plus的五大Agent能力（重大新增）

2026年6月发布的Qwen3.7-Plus在图像/视频理解基础上，明确提出五项面向"自主执行任务"的核心能力，标志Qwen系列正式进入多模态混合Agent技术阶段：

1. **深度推理（Deep Reasoning）**：逐步拆解复杂问题
2. **自主编程（Self-Programming）**：模型自行编写并修订代码
3. **工具调用（Tool Invocation）**：调用外部函数/API
4. **验证与测试（Verification & Testing）**：执行输出结果并核验正确性
5. **自主迭代（Autonomous Iteration）**：循环执行直至任务完成

其多模态版本Qwen3.7-Plus-Preview在LM Arena Vision Arena排名第16位，使阿里在视觉类模型的实验室排名中位列第5；文本版Qwen3.7-Max在Artificial Analysis Intelligence Index上取得56.6分，为发布时中国模型最高分。Bailian（阿里云百炼）平台同步引入**Agentic RL机制**，利用真实世界执行反馈持续优化模型准确性。

---

## 7. 多模态技术体系

### 7.1 视觉语言模型（VL系列）

#### Qwen-VL（2023）→ Qwen2-VL（2024）→ Qwen2.5-VL（2025）→ Qwen3-VL（2025）→ **Qwen3.5原生多模态（2026）**

**视觉编码器演进**：
- Qwen-VL：OpenCLIP ViT + 3阶段训练（图文预训练→多任务精调→指令微调）
- Qwen2-VL：引入**Naive Dynamic Resolution**，原生支持任意分辨率输入
- Qwen2.5-VL：增强视觉定位能力（bounding box/point），强化结构化文档理解
- Qwen3-VL：基于SigLIP-2进行视觉编码器持续训练，支持MoE骨干网络
- **Qwen3.5**：不再是"LLM+视觉编码器"的拼接式VLM，而是**从预训练起点即原生多模态**的统一基础模型（详见4.4节），在同等规模下全面超越Qwen3-VL

**Qwen3-VL架构**：
- 三模块架构：视觉编码器 + MLP视觉语言合并器 + Qwen3 LLM骨干
- 四阶段预训练：S0视觉语言对齐（67B token）→ S1多模态预训练（~1T）→ S2长上下文（~1T，32K）→ S3思维推理
- 支持图像、视频的长上下文理解（256K token窗口，1M token外推）
- Needle-in-a-Haystack测试：256K上下文100%准确率，1M token外推达99.5%

### 7.2 音频语言模型

**Qwen-Audio（2023）**：采用类Whisper的音频编码器，支持多种音频理解任务（ASR、音频问答、情感识别），与LLM共享训练目标，实现音频-文本统一理解。

### 7.3 全模态模型（Omni系列）

**Qwen2.5-Omni（2025年3月）→ Qwen3-Omni（2025年9月）→ Qwen3.5-Omni（2026年4月，闭源）**

**Thinker-Talker架构**：
- **Thinker**：大型MoE Transformer，负责跨模态理解与推理（"大脑"）
- **Talker**：紧凑型MoE语音生成器，负责流式语音合成（"嘴"）

**Qwen3-Omni的五大升级**：
1. Thinker和Talker均升级为MoE架构，支持高并发推理
2. 引入自研AuT（Audio Transformer）音频编码器，在2000万小时监督音频上从零训练
3. Talker解耦设计：不再消费Thinker的高层文本表示，仅依赖音频/视觉多模态特征
4. 支持streaming生成（首包延迟234ms）
5. TM-RoPE位置编码实现音视频流精确时间对齐

**Qwen3.5-Omni（新增）**：
- Thinker与Talker均**升级为Qwen3.5引入的Hybrid MoE架构**，即在语音/视频生成路径中同样引入Gated Delta Net（GDN）模块
- GDN模块专门针对长音视频序列建模优化，显著降低长上下文推理下的KV Cache I/O开销，提升生成吞吐量与并发服务能力
- 延续Qwen3-Omni的分块预填充（Chunked-Prefilling）机制，进一步压缩首包延迟（TTFT）
- 2026年3月起，Qwen3.5-Omni与Qwen3.6-Plus均以**闭源**形式发布，仅通过官方网站与阿里云平台提供访问，未再开放权重（详见12.4节生态策略转向）

**能力覆盖**：文本理解与生成、图像识别与推理、视频分析、语音识别与合成，端到端，一个模型完成。

---

## 8. 专域模型矩阵

### 8.1 Qwen2.5-Coder → Qwen3-Coder → Qwen3-Coder-Next

**Qwen2.5-Coder**：在超过1万亿token的代码数据上专项预训练，支持90+种编程语言，工具集成推理（可调用Python解释器验证代码），Fill-in-the-Middle（FIM）训练，最大32B参数。

**Qwen3-Coder（2025年7月，重大升级）**：
- 旗舰版**Qwen3-Coder-480B-A35B**：480B总参数/35B激活参数，训练语料达7.5万亿token（其中70%为代码）
- 在SWE-bench Verified等Agentic Coding基准上达到开源模型最先进水平，**性能对标Anthropic Claude Sonnet 4**
- 轻量版**Qwen3-Coder-30B-A3B**（Flash）：面向本地部署与低延迟场景

**Qwen3-Coder-Next（2026年2月）**：采用Qwen3-Next的小尺寸混合架构（Gated DeltaNet + MoE），在保持较小模型体量的同时推动Agentic Coding（仓库级推理、前端工作流处理）能力逼近大模型水准，是"小模型高效Agent编程"路线的代表作。

### 8.2 Qwen2.5-Math

**定位**：数学推理专用模型

**技术特点**：专项数学语料超1万亿token（含大量合成数学数据），工具集成推理（TIR），链式思维（CoT）数据高质量合成与筛选，配套奖励模型用于数学答案验证。MATH、GSM8K、AMC、AIME等数学基准达到领先水平。

### 8.3 QwQ（Qwen with Questions）

**定位**：开放权重的慢思考（Slow-thinking）推理模型，32B参数规模，类似OpenAI o1的推理范式，是Qwen3统一思维框架的先驱实验。

### 8.4 QVQ（Qwen Visual Questions）

**定位**：视觉多模态推理专用模型，72B参数规模，将QwQ的慢思考能力扩展至视觉理解。

---

## 9. 混合专家（MoE）技术路线

### 9.1 MoE发展历程（更新）

```
Qwen1.5-MoE-A2.7B（2024年3月）：首次MoE实验，14.3B总参数，2.7B激活
Qwen2.5-Turbo/Plus（2024年）：商业闭源MoE产品
Qwen3-30B-A3B（2025年4月）：轻量MoE，30B总参，3B激活
Qwen3-235B-A22B（2025年4月）：旗舰MoE，235B总参，22B激活
Qwen3-Coder-480B-A35B（2025年7月）：编程旗舰MoE，480B总参，35B激活
Qwen3-Next-80B-A3B（2025年9月）：首个混合注意力+高稀疏MoE，512专家（10路由+1共享）
Qwen3.5-397B-A17B（2026年2月）：原生多模态混合MoE旗舰，397B总参，17B激活
```

### 9.2 Qwen3 MoE架构设计（基础范式）

**细粒度专家分割**：每层使用128个专家，每个token激活8个（top-k路由），提升专家专业化程度、降低负载不均衡风险。
**无共享专家设计**：不同于Qwen2.5-MoE的共享专家设计，Qwen3 MoE完全依赖路由专家。
**全局批次负载均衡损失**：采用全局批次负载均衡辅助损失，防止"专家坍塌"（Expert Collapse）。

### 9.3 Qwen3-Next/Qwen3.5的高稀疏度MoE演进（新增）

- **专家规模扩大，激活比例进一步降低**：从Qwen3的128专家（8激活）扩展至Qwen3-Next/Qwen3.5的**512专家（10路由+1共享）**，重新引入"共享专家"设计（与Qwen3路线有所回摆），共享专家负责捕获通用模式，路由专家负责专业化分工
- **实证规律**：在保持激活专家数固定的前提下，持续增加总专家参数量能稳定降低训练损失（Pareto改进），这一发现是Qwen3-Next/Qwen3.5持续扩大专家池的理论依据
- **计算效率**：Qwen3.5-397B-A17B仅需激活约4.3%的参数（17B/397B），较Qwen3-235B-A22B的9.4%激活比进一步降低，是吞吐量提升8.6～19倍的架构基础之一

---

## 10. 长上下文与高效推理技术

### 10.1 长上下文关键技术

**YARN（Yet Another RoPE extensioN）**：通过调整RoPE基频和插值方式，将训练时的短上下文外推至更长序列。Qwen2.5-1M使用YARN将上下文从32K扩展至1M token；Qwen3-Next/Qwen3.5则依靠混合架构原生达到262K，外推至约1.01M。

**Dual Chunk Attention（DCA）**：对超长序列分块处理，块内标准注意力，块间稀疏注意力，降低注意力计算复杂度。

**ABF（Adjusted Base Frequency）**：将RoPE基频从10,000调整至1,000,000（甚至10,000,000），配合长上下文训练数据实现无缝上下文窗口扩展。

**Gated DeltaNet（新增，见3.1.1节）**：从架构根本上以固定尺寸状态矩阵替代随长度增长的注意力图，是Qwen3-Next/Qwen3.5系列长上下文效率提升的核心机制，与YARN/DCA/ABF等"事后扩展"技术形成互补而非替代关系。

### 10.2 推理效率优化

**模型量化**：官方支持AWQ量化版本，支持FP8量化（旗舰模型），兼容GPTQ、INT4/INT8等主流量化格式。

**推理框架支持**：
- vLLM：官方推荐高吞吐量推理框架，已支持Qwen3-Next混合架构算子
- SGLang：支持批量推理优化
- llama.cpp：支持本地CPU/GPU混合推理
- Transformers（HuggingFace）：Qwen3-Next代码已合并至主分支
- **注意**：Gated DeltaNet层需要专门的kernel实现，社区反馈初期部分平台上的推理速度可能尚未完全反映理论加速比，需关注各推理框架的持续优化

**强到弱蒸馏（Strong-to-Weak Distillation）**：使用大型Qwen模型为小型模型生成高质量合成数据，显著提升小型模型（如0.6B、0.8B、1.7B）的性能天花板。

---

## 11. 基准评测与竞争格局

### 11.1 Qwen3主要基准成绩

#### 知识与推理

| 基准 | Qwen3-235B-A22B | Qwen3-32B |
|------|----------------|----------|
| MMLU | ~87+ | ~85+ |
| GPQA Diamond | ~75 | ~68 |
| SuperGPQA | 领先水平 | 竞争力强 |

#### 数学与STEM

| 基准 | Qwen3-235B-A22B | 说明 |
|------|----------------|------|
| AIME'24 | 85.7 | 超越o3-mini等 |
| AIME'25 | 81.5 | 最新竞赛题目 |
| MATH | 接近满分 | 经典数学基准 |

#### 代码能力

| 基准 | Qwen3-235B-A22B | Qwen3-Coder-480B-A35B |
|------|----------------|----------------------|
| LiveCodeBench v5 | 70.7 | 领先水平 |
| CodeForces | 2056（Expert级别） | — |
| SWE-bench Verified | — | 对标Claude Sonnet 4 |

#### 工具与Agent

| 基准 | Qwen3-235B-A22B |
|------|----------------|
| BFCL v3 | 70.8 |

### 11.2 Qwen3.5/3.7代际基准表现（新增）

| 基准/指标 | Qwen3.5-397B-A17B | 说明 |
|------|----------------|------|
| GDPval-AA | 超越Qwen3-235B达361分 | 长文档/代理任务综合基准 |
| SWE-bench Verified（多模态版本） | 78.8% | 逼近闭源旗舰水平 |
| LM Arena（Qwen3.5-Max-Preview） | 1464分，全球第6、中国第1 | 自报数据 |
| Artificial Analysis Intelligence Index（Qwen3.7-Max） | 56.6 | 发布时中国模型最高分 |
| LM Arena Vision Arena（Qwen3.7-Plus-Preview） | 排名第16（实验室排名第5） | 视觉多模态 |

> **重要说明**：以上跨代际及跨厂商对比数据（尤其与GPT-5、Claude Opus等模型的"持平"表述）主要来自阿里官方自评或第三方竞技场（如LM Arena）的自报结果，第三方媒体（如CNBC）明确指出"无法独立验证阿里所提供的对比数据"。读者在引用时应注意基准评测的局限性：训练数据污染、评测集选择偏差、"自报（self-reported）"性质等因素均可能影响可比性。

### 11.3 关键规律：参数效率的突破

Qwen3 Dense模型呈现出显著的**参数效率提升**规律：

> Qwen3-1.7B/4B/8B/14B/32B-Base 性能分别与 Qwen2.5-3B/7B/14B/32B/72B-Base 相当

即Qwen3的模型在同等性能下，参数量减少约一半。**这一趋势在Qwen3.5延续并深化**：Qwen3.5-9B据称在多项基准上超越参数量更大的OpenAI gpt-oss-120B，Qwen3.5-4B在同权重级别中表现突出，进一步印证"小模型高能力"是Qwen系列持续投入的方向。

### 11.4 竞争格局定位（更新至2026年）

| 竞争维度 | 对标模型 | Qwen地位 |
|----------|---------|----------|
| 开源旗舰 | LLaMA系列（Meta）、Mistral Large、GLM（Zhipu） | Qwen3.5-397B-A17B为2026年中"世界第三强开源模型"（据行业评测），处于领先梯队 |
| 推理能力 | GPT-5/o系列（OpenAI）、DeepSeek-R1、Gemini 2.5/3思考模式 | Qwen3系AIME成绩持平或超越，Qwen3-Next-Thinking超越Gemini-2.5-Flash-Thinking |
| 多语言 | Gemini（Google） | 201种语言覆盖（Qwen3.5）在开源模型中处于领先位置 |
| 小模型效率 | Gemma（Google）、Phi（Microsoft）、gpt-oss（OpenAI） | Qwen3.5小模型（4B/9B）性价比与效率突出 |
| 商业API/闭源旗舰 | GPT-5、Claude Opus、Gemini 3 | Qwen3.5-Plus/Qwen3.6-Plus/Qwen3.7-Max为阿里对标闭源旗舰的最新尝试，但已不再随主线开源（详见12.4节） |
| Agent能力 | Anthropic Claude Agent系列、OpenAI Agent工具 | Qwen3.6/3.7聚焦"真实世界Agent"，为中国大模型厂商中较早系统性押注Agent方向的团队之一 |
| 国内竞品 | DeepSeek、Moonshot（Kimi）、Zhipu（GLM） | 2026年农历新年前后，字节跳动、智谱AI等同期密集发布Agent能力升级模型，国内"Agent竞赛"白热化 |

---

## 12. 生态系统与开源策略

### 12.1 开源策略（含2026年重大转向）

Qwen历史上采用**双轨制**：

**开源权重（Open-weight）**：所有Dense模型和部分MoE模型提供完整权重，托管于Hugging Face Hub、ModelScope、Kaggle，超过20万个不同规格/量化版本可供下载，Qwen3-VL-2B-Instruct单模型下载量已突破1800万次。许可证通常为Apache 2.0（商业友好）或Qwen专有许可证。

**闭源API服务**：Qwen-Max、Qwen-Plus、Qwen-Turbo等商业API，通过阿里云百炼（Model Studio）平台提供企业级服务。

**（2026年新趋势）向"闭源优先"摆动**：与此前"核心版本必然开源"的模式不同，**2026年2月起，Qwen3.5的首个模型（397B-A17B）以开源权重发布，但同期的Qwen3.5-Plus（托管版）为闭源**；4月发布的**Qwen3.5-Omni与Qwen3.6-Plus均为闭源**，仅通过官网与阿里云平台提供访问；同月发布的Qwen3.6-35B-A3B则依然遵循Apache 2.0开源。南华早报等媒体将此概括为"中国AI巨头转向闭源模型以驱动营收与性能"的行业性趋势，DeepSeek、Moonshot等同行也出现类似动向。阿里官方回应称将继续坚持开源投入，但旗舰/最新模型的"先闭源、后逐步开放"节奏已较2025年更为明显。

### 12.2 工具生态

**Qwen-Agent框架**：基于Qwen≥3.0构建的Agent应用框架，支持Function Calling、MCP（Model Context Protocol）、Code Interpreter、RAG，提供Chrome扩展等应用层工具，支持并行、多步骤、多轮工具调用。

**Model Context Protocol（MCP）支持**：Qwen3系列原生支持MCP协议，增强与外部工具和系统的互操作性，支持工具调用的流式处理和并行执行。

**Qwen Studio（原Qwen Chat，2026年更名，新增）**：官方交互门户，提供Web/桌面/移动端一体化体验，功能涵盖聊天机器人、图像/视频理解、图像生成、文档处理、网页搜索集成、工具调用与Artifacts（可交互产物生成）。Qwen API通过阿里云百炼（Model Studio）提供，兼容OpenAI与Anthropic的API规范，便于开发者迁移接入。

### 12.3 部署生态

| 框架 | 支持版本 | 适用场景 |
|------|---------|---------|
| Hugging Face Transformers | 全系列（Qwen3-Next代码已合并主分支） | 研究、开发 |
| vLLM | 全系列 | 高吞吐量生产部署 |
| SGLang | Qwen2+系列 | 批量推理 |
| llama.cpp | Qwen2+系列 | 本地/边缘部署 |
| mlx-lm | Qwen2+系列 | Apple Silicon部署 |
| TensorRT-LLM / NVIDIA NIM | 主要版本 | NVIDIA企业部署 |
| Ollama | 主要版本（含Qwen3-Next） | 个人本地部署 |
| Together AI 等第三方推理云 | Qwen3.5等最新版本 | 云端API托管 |

### 12.4 消费生态与商业化布局（新增）

- **2026年1月**：Qwen App与阿里生态打通，率先接入餐饮外卖（饿了么）服务，用户可通过对话式交互直接完成点餐等任务
- **规划**：阿里宣布将逐步把任务分配能力扩展至淘宝、飞猪等更多平台，并支持打电话、文档处理等"跑腿类"代理任务
- **用户规模**：截至2026年5月，Qwen App月活用户已达**2.34亿**
- **API定价（DashScope，供参考）**：最小模型输入价格低至0.01美元/百万token，旗舰Qwen3 Max约0.78美元/百万token，并为新账户提供100万输入+100万输出token的免费额度

---

## 13. 组织架构与治理变化（新增章节）

这是原报告未覆盖、但对理解Qwen未来走向至关重要的新维度。

### 13.1 核心技术负责人离职

2026年3月，长期担任Qwen模型技术负责人的**林俊旸（Lin Junyang）**在Qwen3.5与Qwen3.5-Plus发布后离职，是2026年内阿里已知的第三位重要高管离职。这一消息一度引发外界对阿里是否会削弱开源与基础研究投入的担忧（VentureBeat以"阿里是否正在削弱其强大的Qwen团队"为题进行报道）。阿里官方回应强调将继续坚持开源方向的投入。

### 13.2 "Token Hub"事业群成立

2026年3月，阿里巴巴宣布成立新的AI业务单元——**Token Hub（通义大模型事业群）**，统筹管理公司AI相关工作：

- **领导架构**：由阿里巴巴集团CEO**吴泳铭（Eddie Wu）**亲自统领，**周靖人（Zhou Jingren）**任首席AI架构师，**吴泽明（Wu Zeming）**任CTO
- **整合范围**：涵盖通义大模型事业部（Tongyi Large Model Business Unit，原通义实验室）、MaaS业务线（Model-as-a-Service）、Qwen、悟空（Wukong，阿里文生图/视频品牌）、AI Innovation等多条业务线
- **组织意义**：原"通义实验室"重组后专注于Qwen系列模型的研发，整体架构调整被解读为阿里将AI提升至集团最高优先级、强化跨部门协同的信号

### 13.3 治理与安全新挑战

- **思维模式带来的新安全议题**：模型内部推理过程可能产生有害的中间步骤，如何在保留思维链透明性的同时进行安全审查是开放问题
- **多语言对齐挑战随语言数扩张而加剧**：Qwen3.5支持201种语言，但低资源语言的对齐质量与安全过滤能力仍是行业公认的短板
- **开源与合规的地缘政治维度**：美中经济与安全审查委员会（USCC）在2026年3月的报告中指出，以Qwen为代表的中国开源AI策略，是中国突破算力约束、提升真实世界数据整理能力的关键手段，这一评估也令Qwen系列的开源节奏成为中美科技政策关注的焦点之一

---

## 14. 关键技术创新总结（更新）

### 14.1 架构创新

| 创新点 | 首次引入版本 | 技术价值 |
|--------|------------|---------|
| GQA | Qwen2 | KV Cache效率↑，推理吞吐量↑ |
| SwiGLU | Qwen1 | 非线性表达能力↑ |
| RMSNorm + Pre-Norm | Qwen1 | 训练稳定性↑ |
| QK-Norm | Qwen3 | 超大规模训练数值稳定性↑ |
| 细粒度MoE（128专家） | Qwen3 | 专家专业化↑，负载均衡↑ |
| 全局批次负载均衡 | Qwen3 | MoE训练质量↑ |
| MRoPE/TM-RoPE | Qwen-VL/Qwen3-Omni | 多模态位置表示↑ |
| **Gated DeltaNet混合注意力** | **Qwen3-Next（2025年9月）** | **长上下文推理复杂度由O(n²)降至O(n·d²)，吞吐量数量级提升** |
| **512专家高稀疏MoE（10路由+1共享）** | **Qwen3-Next/Qwen3.5** | **激活比例降至4.3%，进一步降低推理成本** |
| **早期融合原生多模态** | **Qwen3.5（2026年2月）** | **从后期拼接视觉编码器转为从零联合训练，多模态理解能力质变** |
| **多token预测（MTP）** | **Qwen3-Next** | **预训练效果↑，推理速度↑** |

### 14.2 训练创新

| 创新点 | 相关版本 | 技术价值 |
|--------|---------|---------|
| 三阶段预训练 | Qwen2.5/Qwen3 | 通用+推理+长上下文能力分层建立 |
| 36T token + 119语言语料 | Qwen3 | 数据规模与多样性的新标杆 |
| Qwen模型自举合成数据 | Qwen3 | 数学、代码数据的数量与质量↑ |
| 扩展律指导超参数 | Qwen3 | 训练效率↑，搜索成本↓ |
| GRPO | Qwen2.5+ | 无Critic的稳定在线RL↑ |
| 四阶段后训练 | Qwen3 | 统一思维模式的系统性实现 |
| 强到弱蒸馏 | Qwen2.5/3/3.5 | 小模型性能↑ |
| **GSPO** | **Qwen3-Next起** | **混合架构+高稀疏MoE场景下RL训练稳定性↑** |
| **百万级Agent环境RL** | **Qwen3.5起** | **真实世界任务泛化能力↑** |
| **201语言语料+250K词表** | **Qwen3.5** | **多语言编解码效率↑10%-60%** |

### 14.3 能力创新

| 创新点 | 相关版本 | 技术价值 |
|--------|---------|---------|
| 统一思维/非思维模式 | Qwen3 | 无需切换模型，全场景覆盖 |
| 思维预算控制 | Qwen3 | 推理深度与计算成本可控权衡 |
| Thinker-Talker架构 | Qwen2.5/3/3.5-Omni | 全模态端到端流式生成 |
| 动态分辨率视觉编码 | Qwen2-VL+ | 任意分辨率图像无感知处理 |
| 1M token上下文 | Qwen2.5-1M / Qwen3.5外推 | 超长文档处理能力突破 |
| **原生多模态Agent能力** | **Qwen3.5/3.6/3.7** | **UI理解与操作、深度推理+自主编程+验证测试+自主迭代五合一** |
| **思维保留（跨对话）** | **Qwen3.6** | **多轮迭代开发场景计算开销↓** |

---

## 15. 发展趋势与未来展望

### 15.1 官方明确的未来方向（基于Qwen3技术报告）

**1. 强化Agent能力** — 增加基于环境反馈的Agent RL计算资源投入，构建能够处理需要推理时间扩展的复杂任务的Agent，深化工具使用、记忆管理、动作规划能力（**这一方向已在Qwen3.5～3.7中被系统性落地，详见6.4、5.7节**）

**2. 推理时间扩展（Inference-time Scaling）** — 进一步探索思维预算与任务性能的关系，构建更完善的"慢思考"能力

**3. 多模态深化** — 已从"后期融合VLM"演进为"早期融合原生多模态"（Qwen3.5），视频理解、长视频分析持续是重点投入方向

**4. 数据飞轮持续强化** — 利用Qwen系列多模态模型持续扩充高质量预训练数据，扩大合成数据在数学、代码、科学领域的比例

### 15.2 技术趋势研判（更新）

**趋势一：混合注意力架构成为长上下文效率的新范式**

Gated DeltaNet等线性注意力与标准注意力的混合方案，正在取代"纯Transformer+外推技巧"的旧范式。预计更多厂商将跟进类似的混合线性/标准注意力设计，以突破O(n²)复杂度对长上下文规模化部署的制约。

**趋势二：原生多模态取代"文本优先+适配"路线**

Qwen3.5的早期融合实践验证了原生多模态训练在效果上的优势。预计Qwen及同行下一代旗舰模型将进一步弱化"先做文本LLM，再做VLM"的分阶段路线，转向从预训练起点即统一建模文本、图像、视频、音频。

**趋势三：Agent能力成为核心竞争轴，评测体系加速迭代**

Qwen3.6/3.7明确将"真实世界Agent"作为技术主线，GDPval-AA等面向真实任务、而非单纯静态问答的评测基准开始受到重视，反映行业评测标准正从"知识/推理分数"转向"任务完成度"。

**趋势四：开源节奏出现分化，商业化压力上升**

2026年以来，Qwen3.5-Plus、Qwen3.5-Omni、Qwen3.6-Plus等最新旗舰/多模态模型陆续转为闭源，仅通过官方渠道提供访问，与Qwen3时代"全系列Apache 2.0"的策略形成对比。这与DeepSeek等同行的类似动向共同构成"中国AI巨头为营收与差异化转向部分闭源"的行业趋势，值得持续关注其是否会演变为长期战略调整。

**趋势五：组织架构与AI优先级持续强化**

Token Hub事业群的成立、CEO直接统领AI业务，表明阿里正将AI置于集团最高战略地位；但核心技术人才流动（如林俊旸离职）也提示，快速迭代节奏下团队稳定性与技术连续性存在一定不确定性。

**趋势六：端侧部署与效率竞赛持续**

Qwen3.5-0.8B、2B、4B等小型号在256K原生上下文、混合架构加持下，进一步兑现"小模型大能力"的效率目标；量化技术（FP8、INT4）与架构优化协同推进，移动设备与边缘计算专用模型版本预计将持续增加。

---

## 16. 技术风险与挑战

### 16.1 技术层面的挑战

**长上下文的"有效利用"问题**：支持1M+ token ≠ 能有效利用其中的所有信息，真实的多跳推理与归纳能力仍有差距，长上下文训练的计算成本极高。

**混合架构的工程成熟度**：Gated DeltaNet等新型算子在部分推理框架/硬件平台上仍处于kernel优化早期阶段，理论加速比与实际部署表现之间可能存在阶段性落差。

**MoE的工程复杂度**：专家间通信（All-to-All）是分布式训练的瓶颈，负载不均衡问题在动态推理场景下更难控制，量化后的MoE模型质量损失高于Dense模型。

**推理一致性**：Thinking模式下的推理过程难以保证逻辑一致性，同一问题多次推理可能得到不同的中间过程，"推理质量"评估仍是开放问题。

**跨代际基准可比性存疑**：厂商自评或竞技场自报数据（如与GPT-5、Claude Opus"持平"的表述）缺乏第三方独立复核，实际使用体验可能与宣传基准存在落差。

### 16.2 竞争与生态挑战

**开源生态的竞争与自身开源节奏的摇摆**：Meta LLaMA系列、Google Gemma系列、DeepSeek、Moonshot（Kimi）等强劲竞争者持续迭代；Qwen自身近期出现的"旗舰闭源化"倾向，也可能影响其在开源社区中的号召力。

**对齐与安全**：思维模式带来内部推理过程可能产生有害中间步骤的新挑战；201种语言的多语言对齐质量在低资源语言上难以保证；随着模型能力（尤其是自主编程、自主迭代等Agent能力）提升，越狱攻击面与滥用风险同步扩大。

**商业可持续性与人才稳定性**：大规模预训练（万亿级多模态token）计算成本极高；核心技术负责人变动为团队连续性带来不确定性；如何在开源贡献、商业变现与集团战略优先级（Token Hub整合）之间保持平衡，是2026年后Qwen面临的核心治理议题。

**地缘政治关注度上升**：美中经济与安全审查委员会等机构已将Qwen等中国开源模型的发展路径纳入战略评估，未来Qwen的开源/闭源决策可能进一步受到跨境合规、出口管制等外部因素影响。

---

## 附录：参考文献

### 核心技术论文

| 论文 | 链接 |
|------|------|
| Qwen Technical Report (2023) | arXiv:2309.16609 |
| Qwen2 Technical Report (2024) | arXiv:2407.10671 |
| Qwen2.5 Technical Report (2024) | arXiv:2412.15115 |
| Qwen3 Technical Report (2025) | arXiv:2505.09388 |
| Qwen-VL Technical Report (2023) | arXiv:2308.12966 |
| Qwen2-VL Technical Report (2024) | arXiv:2409.12191 |
| Qwen2.5-VL Technical Report (2025) | arXiv:2502.13923 |
| Qwen3-VL Technical Report (2025) | arXiv:2511.21631 |
| Qwen2.5-Omni Technical Report (2025) | arXiv:2503.20215 |
| Qwen3-Omni Technical Report (2025) | arXiv:2509.17765 |
| Qwen2.5-Math Technical Report (2024) | arXiv:2409.12122 |
| Qwen2.5-Coder Technical Report (2024) | arXiv:2409.12186 |
| QwQ Blog (2024) | qwenlm.github.io/blog/qwq-32b |
| Qwen2.5-1M Technical Report (2025) | arXiv:2501.15383 |
| **Qwen3-Next: Towards Ultimate Training & Inference Efficiency (2025)** | qwen.ai/blog（2025-09-11） |
| **Qwen3.5: Towards Native Multimodal Agents (2026)** | qwen.ai/blog?id=qwen3.5（2026-02-16） |
| **Qwen3.5-Omni Technical Report (2026)** | arXiv:2604.15804 |
| **Qwen3.6-Plus: Towards Real World Agents (2026)** | qwen.ai/blog?id=qwen3.6 |

### 主要新闻与行业报道（新增）

| 报道 | 来源 | 日期 |
|------|------|------|
| Alibaba unveils Qwen3.5 as China's chatbot race shifts to AI agents | CNBC/Reuters | 2026-02-16/17 |
| Did Alibaba just kneecap its powerful Qwen AI team? | VentureBeat | 2026-03-04 |
| Alibaba forms task force to boost AI development after Qwen chief's exit | Reuters | 2026-03-05 |
| Alibaba Consolidates AI Operations Under New Business Group | The Wall Street Journal | 2026-03-17 |
| Chinese AI giants pivot toward proprietary models to drive revenue, performance | South China Morning Post | 2026-04-02 |
| Alibaba's Qwen Team Launches Qwen3.7-Plus | MarkTechPost | 2026-06-02 |
| How China's Open AI Strategy Reinforces Its Industrial Dominance | U.S.-China Economic and Security Review Commission | 2026-03-23 |

### 代码与模型仓库

- **GitHub**：https://github.com/QwenLM （含QwenLM/Qwen3.6等子仓库）
- **Hugging Face**：https://huggingface.co/Qwen
- **ModelScope**：https://modelscope.cn/organization/qwen
- **官方博客**：https://qwen.ai/blog （原qwenlm.github.io/blog）
- **官方交互平台**：https://chat.qwen.ai （Qwen Studio）
- **阿里云百炼（Model Studio）**：企业级API服务入口