# vLLM Structured Outputs 技术报告

> 基于 vLLM 官方文档（latest/v0.9.2）及相关学术资料综合分析  
> 报告日期：2026-06

---

## 一、概述

Structured Outputs（结构化输出）是 vLLM 的核心功能之一，旨在将大语言模型（LLM）的生成过程约束到预定格式，如 JSON、正则表达式、上下文无关文法（CFG）等，从而消除解析失败风险，确保下游系统的可靠性。

与 prompt engineering 方式（"请输出 JSON 格式"）相比，结构化输出通过对解码过程本身施加约束，在数学上保证输出合法性，是生产级 LLM 应用的必备能力。

---

## 二、技术路线全景

### 2.1 方法论分类

结构化输出的实现路线整体分为两大类：

| 类别 | 思路 | 代表方案 |
|------|------|---------|
| **后处理修复（Post-processing）** | 生成后尝试解析，失败则重试或修复 | 早期 prompt-only 方案 |
| **约束解码（Constrained Decoding）** | 在每步 token 采样前，屏蔽所有违反约束的 token | Outlines、XGrammar、lm-format-enforcer、Guidance |

vLLM 采用**约束解码**路线，在 logits 层进行 mask，从根本上杜绝格式错误。

### 2.2 vLLM 后端（Backend）演进历程

```
vLLM v0.x（早期）
├── outlines          （基于 FSM，最早引入）
├── lm-format-enforcer（基于 Python regex）
└── xgrammar          （后续引入，逐渐成为默认）

vLLM v1.x（当前 latest）
├── xgrammar          ← 默认，推荐，适合大多数场景
└── guidance          （llguidance，Microsoft 出品）
     └── auto 模式    ← 系统根据请求自动路由最优后端
```

> **注**：v1 引擎重构后，outlines 和 lm-format-enforcer 已被移除，统一为 xgrammar + guidance 双后端架构，并引入智能 `auto` 路由模式。

---

## 三、核心后端深度分析

### 3.1 Outlines（历史参考）

**原理**：将 JSON Schema / Regex 编译成有限状态机（FSM），每步解码时根据当前状态计算合法 token 集合，生成二值 mask 施加到 logits。

**核心缺陷**：

- FSM 仅能表示正则语言，无法处理任意递归结构（如多层嵌套 JSON）
- 每步运行时须遍历全词表（vocab size 可达 10 万+），解码延迟可达数秒/步
- 当 LLM 生成的 token 跨越词法 token 边界时，状态判断可能出错（cross-token 问题）
- schema 缓存能力有限，冷启动开销大

### 3.2 lm-format-enforcer

**原理**：使用 Python `re` 模块匹配正则，结合 token trie 树加速 mask 生成。

**特点与局限**：使用 Python-style regex 语法（与其他后端的 Rust-style regex 有差异），不支持完整 CFG，仅适用于相对简单的格式约束场景。

### 3.3 XGrammar（当前主推后端）

**来源**：MLC-AI 团队，论文 *XGrammar: Flexible and Efficient Structured Generation Engine for Large Language Models*（2024）

#### 3.3.1 核心技术：下推自动机（PDA）

XGrammar 使用**下推自动机（Pushdown Automaton, PDA）**替代 FSM。PDA 可以理解为"一组 FSM 的集合，每个 FSM 表示一个上下文无关文法规则"，其递归特性天然支持无限嵌套结构（任意深度的 JSON、递归文法等）。

```
FSM（Outlines）         PDA（XGrammar）
  ┌───────┐               ┌───────┐
  │ State │──token──▶...  │ State │──token──▶...
  └───────┘               └───────┘
  仅支持正则语言           ├── Stack：记录递归层级
                           └── 支持完整 CFG
```

#### 3.3.2 自适应 Token Mask Cache

这是 XGrammar 性能领先的核心机制：

**预处理阶段（Offline）**：
- 分析词表中每个 token 是否"上下文无关"（即 mask 结果不依赖 PDA 栈状态）
- 对上下文无关 token 预先计算并持久化缓存 mask
- 这部分 token 通常占词表的绝大多数

**运行时阶段（Online）**：
- 上下文无关 token：直接查缓存，O(1) 返回
- 上下文敏感 token（依赖当前栈状态）：按需计算，但数量极少

此分级策略使每步 mask 生成开销从毫秒级压缩至微秒级。

#### 3.3.3 批量约束解码

XGrammar 为 batch 中每个请求独立维护 PDA 状态，支持不同请求使用完全不同的 schema，不干扰 vLLM 的 continuous batching 调度，吞吐量损失极小。

#### 3.3.4 与竞品能力对比

| 特性 | Outlines | lm-format-enforcer | XGrammar |
|------|:--------:|:-----------------:|:--------:|
| 支持完整 CFG | ✗ | ✗ | ✓ |
| Cross-token 正确性 | ⚠️ 有缺陷 | ⚠️ 有缺陷 | ✓ |
| 每步解码延迟 | 高（秒级） | 中 | 低（μs 级）|
| 多级 Caching | 有限 | 无 | ✓ 自适应 |
| Batch 支持 | 有限 | 有限 | ✓ 原生 |
| 语法支持 | Rust regex | Python regex | EBNF CFG + regex |
| vLLM v1 支持 | ✗ 已移除 | ✗ 已移除 | ✓ |

### 3.4 Guidance / llguidance（Microsoft）

**来源**：Microsoft guidance-ai 项目，底层为高性能 Rust 实现的 `llguidance` 库。

**技术特点**：
- Rust-style regex，编译与执行均高效
- 在**每请求 schema 唯一**（无 cache 复用）的场景下，TTFT（首 token 时间）优于 XGrammar
- 提供灵活的结构声明语法，支持更丰富的约束表达

**适用场景**：动态 schema 场景（schema 几乎不重复），或对首 token 延迟敏感的实时应用。

---

## 四、vLLM 实现架构

### 4.1 整体数据流

```
用户请求
  │  guided_json / guided_regex / guided_grammar / guided_choice / structural_tag
  ▼
GuidedDecodingParams（采样参数封装层）
  │
  ▼
Backend 路由（auto / xgrammar / guidance / xgrammar:no-fallback）
  │
  ▼
Grammar Compilation（Schema/Regex/CFG → PDA / Automaton）
  │  ← vLLM V1：异步后台编译，非阻塞
  ▼
┌─────────────────────────────────────────────┐
│              推理循环（每步）                │
│                                             │
│  GPU: Forward Pass → Logits                 │
│  CPU: Token Mask 计算（与 GPU 并行）         │
│        ├─ Cache Hit  → 直接返回 mask        │
│        └─ Cache Miss → PDA 状态推进 → mask  │
│                                             │
│  masked_logits → Sampling → Next Token      │
│  → PDA 状态更新                             │
└─────────────────────────────────────────────┘
  │
  ▼
完整合法输出
```

### 4.2 vLLM V1 引擎的关键架构改进

**V0 的问题**：Grammar 编译（schema → automaton）发生在请求处理的关键路径上，会**阻塞整个推理引擎**，在高并发场景下造成严重排队延迟。

**V1 的解决方案**：

| 改进点 | V0 行为 | V1 行为 |
|-------|---------|---------|
| Grammar 编译 | 同步阻塞，占用引擎线程 | 异步后台线程，非阻塞 |
| 初始化时机 | 阻塞 prefill 开始 | prefill 与编译并行 |
| Cache 架构 | 请求级 | 跨请求共享 grammar cache |
| 后端支持 | outlines/lm-format-enforcer/xgrammar | xgrammar/guidance（更精简） |

### 4.3 auto 模式路由逻辑

`--guided-decoding-backend auto`（默认）会根据请求特征自动选择后端：

```
请求到达
  │
  ├─ guided_choice → xgrammar（简单枚举，编译极快）
  ├─ guided_regex  → xgrammar 或 guidance（视复杂度）
  ├─ guided_json   → xgrammar（schema cache 优先）
  └─ guided_grammar → xgrammar（CFG 专项能力）

未来版本将引入更精细的路由策略（基于 schema 复杂度、请求并发度等）
```

---

## 五、API 接口完整说明

### 5.1 在线服务（OpenAI 兼容 API）

通过 `extra_body` 传入约束参数，或使用标准 `response_format` 字段：

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="-")
model = client.models.list().data[0].id

# ① 选项约束（guided_choice）
completion = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "Classify this sentiment: vLLM is wonderful!"}],
    extra_body={"guided_choice": ["positive", "negative"]},
)

# ② 正则约束（guided_regex）
completion = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "Generate an email for Alan Turing at Enigma."}],
    extra_body={"guided_regex": r"\w+@\w+\.com\n", "stop": ["\n"]},
)

# ③ JSON Schema 约束（guided_json + Pydantic）
from pydantic import BaseModel
from enum import Enum

class CarType(str, Enum):
    sedan = "sedan"
    suv = "SUV"
    truck = "Truck"

class CarDescription(BaseModel):
    brand: str
    model: str
    car_type: CarType

completion = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "Generate a JSON for the most iconic 90s car."}],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "car-description",
            "schema": CarDescription.model_json_schema(),
        },
    },
)

# ④ CFG 文法约束（guided_grammar，EBNF 格式）
simplified_sql_grammar = """
    root             ::= select_statement
    select_statement ::= "SELECT " column " FROM " table " WHERE " condition
    column           ::= "col_1" | "col_2"
    table            ::= "table_1" | "table_2"
    condition        ::= column " = " number
    number           ::= "1" | "2"
"""
completion = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "Generate a SQL query for username and email."}],
    extra_body={"guided_grammar": simplified_sql_grammar},
)

# ⑤ 指定后端 + 禁止 fallback
completion = client.chat.completions.create(
    model=model,
    messages=[...],
    extra_body={
        "guided_json": CarDescription.model_json_schema(),
        "guided_decoding_backend": "xgrammar:no-fallback",
    },
)
```

### 5.2 离线推理（Python API）

```python
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

llm = LLM(model="Qwen/Qwen2.5-7B-Instruct", guided_decoding_backend="xgrammar")

# JSON Schema 约束
guided_decoding = GuidedDecodingParams(json=CarDescription.model_json_schema())
sampling_params = SamplingParams(
    temperature=0.7,
    max_tokens=512,
    guided_decoding=guided_decoding,
)
outputs = llm.generate(prompts, sampling_params)

# 正则约束
guided_decoding = GuidedDecodingParams(regex=r"\d{4}-\d{2}-\d{2}")
sampling_params = SamplingParams(guided_decoding=guided_decoding)

# 选项约束
guided_decoding = GuidedDecodingParams(choice=["yes", "no", "maybe"])
```

### 5.3 Experimental Automatic Parsing（OpenAI API）

vLLM 跟进了 OpenAI 的 `parse()` 接口，支持自动将输出解析为 Pydantic 对象：

```python
from openai import OpenAI
from pydantic import BaseModel

class Step(BaseModel):
    explanation: str
    output: str

class MathResponse(BaseModel):
    steps: list[Step]
    final_answer: str

client = OpenAI(base_url="http://localhost:8000/v1", api_key="-")
completion = client.beta.chat.completions.parse(
    model=model,
    messages=[{"role": "user", "content": "Solve 8x + 31 = 2."}],
    response_format=MathResponse,
)
message = completion.choices[0].message
print(message.parsed)  # 直接得到 MathResponse 实例
```

---

## 六、支持的约束类型汇总

| 约束类型 | 参数 | 后端支持 | 典型应用场景 |
|---------|------|---------|------------|
| 选项列表 | `guided_choice` | 全部 | 分类、情感分析、A/B 决策 |
| 正则表达式 | `guided_regex` | 全部（语法略有差异）| 邮箱、电话、日期、编号提取 |
| JSON Schema | `guided_json` | xgrammar、guidance | API 结构化响应、信息抽取 |
| 标准 response_format | `response_format.json_schema` | xgrammar、guidance | OpenAI 兼容客户端 |
| CFG/EBNF 文法 | `guided_grammar` | xgrammar、guidance | SQL 生成、DSL 构造、代码骨架 |
| 结构标签 | `structural_tag` | xgrammar | 混合自然语言 + 结构化片段 |
| 后端选择 | `guided_decoding_backend` | — | 性能调优、调试 |

---

## 七、推理模型（Reasoning）与结构化输出

vLLM 对 DeepSeek-R1 等推理模型提供原生支持，可在思维链之后生成结构化输出：

```
<think>
  让我先分析这道题...
  步骤一：...
  步骤二：...
</think>
{"answer": "42", "confidence": 0.95}   ← 仅此部分受 JSON 约束
```

vLLM 自动分离推理 token 与答案 token 的约束边界：
- `<think>...</think>` 段：完全自由生成，保留推理能力
- 答案段：施加 `guided_json` 或其他格式约束

使用方式与普通结构化输出完全相同，无需额外配置，vLLM 根据模型的 reasoning parser 配置自动处理。

---

## 八、性能特征与后端选型指南

### 8.1 性能基准（综合 Red Hat & SqueezeBits 测评，2025）

**场景 A：固定 schema 复用（< 100 种模板）**
- XGrammar 的 grammar cache 命中率接近 100%
- TPOT（每输出 token 时间）仅比无约束推理高出 3%~8%
- **推荐：xgrammar**

**场景 B：每请求 schema 完全唯一**
- XGrammar 每次需重新编译，TTFT 较高
- Guidance（llguidance）的 Rust 实现编译更快，TTFT 领先
- **推荐：guidance**

**场景 C：长输出生成（> 500 tokens）**
- XGrammar 低 TPOT 优势随输出长度线性放大
- **推荐：xgrammar**

**场景 D：生产环境默认配置**
- **推荐：auto**，让 vLLM 智能路由

### 8.2 后端选型决策树

```
是否需要完整 CFG 支持？
  │
  ├─ 是 → xgrammar（唯一支持）
  │
  └─ 否
       │
       ├─ schema 是否频繁复用？
       │    ├─ 是 → xgrammar（cache 优势显著）
       │    └─ 否 → guidance（TTFT 更优）
       │
       └─ 不确定 → auto（让系统决定）
```

### 8.3 生产调优建议

**Schema 设计**：
- 将高频输出格式设计为常量 schema，充分利用 grammar cache
- 避免在每次请求中动态生成略有不同的 schema（cache 失效）

**后端配置**：
- 开发/测试：使用 `auto` 默认值
- 生产调试：使用 `xgrammar:no-fallback` 快速暴露兼容性问题，避免隐式降级
- 对延迟极敏感的场景：benchmark 后固定为 `xgrammar` 或 `guidance`

**XGrammar 线程配置**：
- XGrammar 支持多线程 mask 计算，但过度并行反而劣化性能
- 建议从默认值开始，通过实测 TPOT 曲线找最优线程数

**Prompt 工程**：
- 即使格式已被强制约束，在 prompt 中描述期望的字段和格式仍能提升输出语义质量

---

## 九、生态竞品对比

| 框架 | 默认约束后端 | 架构特点 | 生产成熟度 |
|------|------------|---------|----------|
| **vLLM** | XGrammar / Guidance (auto) | 高吞吐，OpenAI 兼容，v1 非阻塞编译 | ⭐⭐⭐⭐⭐ |
| **SGLang** | XGrammar / LLGuidance | RadixAttention + 结构化输出，类似路线 | ⭐⭐⭐⭐ |
| **llama.cpp** | 内置 GGML CFG | C++ 实现，单机低资源，社区广泛 | ⭐⭐⭐⭐ |
| **TGI（HF）** | Outlines | 与 HF 生态深度集成 | ⭐⭐⭐⭐ |
| **LightLLM** | Pre³（实验性） | DPDA 确定化加速，学术前沿 | ⭐⭐ |

---

## 十、前沿研究动态

### 10.1 Pre³：确定性 PDA 加速（2025）

将 PDA 转化为**确定性 PDA（DPDA）**，消除运行时的非确定性分支搜索，进一步压缩每步 mask 计算时间。在 JSON 生成任务中相比 XGrammar 实现了额外加速。

### 10.2 Speculative Decoding × Structured Output

结合推测解码（draft model 生成多个候选 token）与约束解码的联合优化，核心挑战是对 draft token 进行批量格式验证。目前 vLLM 的 Speculative Decoding 和 Structured Outputs 功能已可共存，但联合优化仍是研究热点。

### 10.3 Reward-Guided Speculative Decoding

在约束解码框架内引入奖励信号，在保证格式合法的前提下引导模型输出语义更优的结果，探索约束解码与 RLHF 的结合。

### 10.4 Constrained Adaptive Rejection Sampling（CARS）

从概率论角度重新审视约束解码，提出基于自适应拒绝采样的方法，理论上可在保持格式约束的同时维持更接近原始模型分布的采样，减少约束对输出多样性的损害。

---

## 十一、局限性与注意事项

| 问题 | 说明 | 缓解方案 |
|------|------|---------|
| 复杂 schema 首次编译慢 | 大型嵌套 JSON Schema 编译可能耗时数百 ms | 预热请求 / 服务启动时预编译常用 schema |
| 正则语法差异 | xgrammar/guidance 用 Rust regex，lm-format-enforcer 用 Python re | 迁移时逐条验证语法兼容性 |
| 模型质量上限 | 约束保证格式合法，不保证语义正确 | 弱模型在强约束下需降低约束严格度 |
| 循环陷阱 | 弱模型在严格约束下可能重复生成同一 token | 设置 `repetition_penalty` 或 `max_tokens` 保底 |
| v0 → v1 迁移 | v1 移除了 outlines/lm-format-enforcer | 升级前测试所有 guided_* 请求的兼容性 |
| structural_tag 实验性 | 仅部分后端支持，API 可能变动 | 生产环境谨慎使用，关注 changelog |

---

## 十二、总结

### 技术演进脉络

```
早期（2023）       中期（2024）          当前（2025）
Outlines           + lm-format-enforcer  XGrammar（主）
FSM + regex        + xgrammar（引入）    + llguidance（辅）
                                         + auto 路由
                                         + V1 异步编译
                                         + reasoning 集成
```

### 核心结论

vLLM Structured Outputs 的技术路线已从 FSM 时代演进至 PDA 时代，XGrammar 凭借自适应多级缓存和批量约束解码，将格式约束的性能开销降至近乎零。

**当前最佳实践**：
- 默认使用 `auto` 模式，无需手动选型
- schema 复用场景（API 服务）选 `xgrammar`，性能最优
- 动态唯一 schema 场景选 `guidance`，TTFT 更优
- 推理模型（DeepSeek-R1 等）无需特殊配置，原生支持
- 生产环境建议 `no-fallback` 模式，快速定位问题

随着 XGrammar、llguidance 持续迭代，以及 vLLM V1 引擎的成熟，结构化输出将成为所有 LLM 生产应用的零成本标准基础设施。

---

## 附录：快速参考

### A. 服务端启动参数

```bash
# 默认（auto 路由）
vllm serve Qwen/Qwen2.5-7B-Instruct

# 指定 xgrammar
vllm serve Qwen/Qwen2.5-7B-Instruct --guided-decoding-backend xgrammar

# 指定 guidance
vllm serve Qwen/Qwen2.5-7B-Instruct --guided-decoding-backend guidance
```

### B. GuidedDecodingParams 字段速查

```python
GuidedDecodingParams(
    json=...,        # dict（JSON Schema）或 Pydantic BaseModel 的 schema
    regex=...,       # str，Rust-style 正则表达式
    choice=...,      # list[str]，枚举选项
    grammar=...,     # str，EBNF 上下文无关文法
    backend=...,     # "xgrammar" | "guidance" | "auto"
    whitespace_pattern=...,  # 覆盖 JSON 解码的默认空白模式
)
```

### C. 参考资料

- vLLM 官方文档 Structured Outputs：https://docs.vllm.ai/en/latest/features/structured_outputs/
- XGrammar 论文：https://arxiv.org/abs/2411.15100
- llguidance（Guidance-AI）：https://github.com/guidance-ai/llguidance
- Red Hat Developer Blog（2025-06）：Structured outputs in vLLM
- SqueezeBits 技术博客（2025-09）：Guided Decoding Performance on vLLM and SGLang
- Pre³ 论文：https://arxiv.org/abs/2506.03887