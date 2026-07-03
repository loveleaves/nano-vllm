# vLLM Tool Calling 技术调研报告

> 报告日期：2026-06
> 参考来源：vLLM 官方文档（latest/dev preview）、GitHub Issues、学术论文

## 核心内容概要

1. 概述 — Tool Calling 在 vLLM 中的定位
2. 基本交互流程 — 从请求到响应的完整链路
3. tool_choice 三种模式 — Named / Required / Auto 的机制差异
4. Tool Parser 体系 — 20+ 个内置 Parser 的格式分类与对比表
5. 约束解码机制 — XGrammar/Outlines/LLGuidance 原理与 Strict Mode
6. Chat Template 机制 — Jinja2 模板如何注入工具定义
7. 流式输出支持 — 增量 delta 解析的实现要点
8. 推理模型集成 — DeepSeek-R1/Qwen3 思维链 + 工具调用联合
9. MCP 集成 — 原生 Model Context Protocol 支持
10. 插件化扩展 — 自定义 ToolParser 完整代码结构
11. 性能与工程实践 — 开销分析与已知问题清单
12. 技术路线对比 — 四代演进路线 + vLLM vs SGLang vs TGI 横向对比
13. 总结与选型建议 — 模型优先级与部署配置建议

---

## 一、概述

Tool Calling（工具调用/函数调用）是当前大语言模型（LLM）走向 Agent 化的核心能力之一，允许模型在推理过程中决策调用外部工具（API、函数、数据库等），获取结果后继续生成响应。vLLM 作为高性能 LLM 推理框架，提供了完善的 Tool Calling 支持，兼容 OpenAI API 规范，并通过插件化架构支持多种模型的工具调用格式。

本报告从技术路线、实现方法、模型支持矩阵、约束解码机制、扩展机制等维度，对 vLLM 的 Tool Calling 体系进行系统性调研与分析。

---

## 二、Tool Calling 的基本流程

### 2.1 标准交互流程

```
用户请求（含 tools 定义）
        ↓
vLLM 服务端（将工具定义注入 Prompt）
        ↓
LLM 生成包含工具调用信息的文本
        ↓
Tool Parser（解析模型输出 → 结构化 ToolCall 对象）
        ↓
返回 OpenAI 兼容的 ChatCompletion 响应（含 tool_calls 字段）
        ↓
客户端执行工具，将结果作为 tool role 消息回传
        ↓
LLM 继续生成最终响应
```

### 2.2 关键 API 参数

vLLM 服务端通过以下核心参数开启工具调用支持：

| 参数 | 说明 |
|------|------|
| `--enable-auto-tool-choice` | 启用自动工具调用（Automatic Function Calling） |
| `--tool-call-parser <name>` | 指定工具调用解析器（如 hermes、mistral、llama3_json 等） |
| `--chat-template <path>` | 指定 Jinja2 格式的聊天模板，处理 tool role 消息 |
| `--tool-parser-plugin <path>` | 加载自定义 ToolParser 插件 |

---

## 三、Tool Calling 模式（tool_choice）

vLLM 支持 OpenAI 兼容的三种工具调用控制模式：

### 3.1 Named Function Calling（指定函数）

通过 `tool_choice={"type": "function", "function": {"name": "xxx"}}` 强制模型调用特定工具。

实现机制：底层使用**约束解码（Guided Decoding）**，将工具参数的 JSON Schema 注入解码约束，保证输出可解析。该模式默认启用，无需额外配置，适用于所有支持的模型。

### 3.2 Required Function Calling（强制调用）

通过 `tool_choice="required"` 保证模型一定生成一个或多个工具调用。

实现机制：同样依赖约束解码，使用带 `anyOf` 的 JSON Schema 覆盖全部工具定义。目前 V0 引擎依赖 `outlines` 后端，V1 引擎支持正在路线图中。

```python
response = client.chat.completions.create(
    model="...",
    messages=[...],
    tools=[...],
    tool_choice="required"  # 强制调用至少一个工具
)
```

### 3.3 None / Auto（禁用 / 自动）

- `tool_choice="none"`：即使请求中包含工具定义，模型也不会触发工具调用。
- `tool_choice="auto"`（默认）：模型根据上下文自主决定是否调用工具，需启用 `--enable-auto-tool-choice`。

---

## 四、核心技术路线：Tool Parser 体系

vLLM 的工具调用核心是 **Tool Parser 插件体系**，每种模型系列对应一个解析器，将模型生成的原始文本解析为结构化的 `tool_calls` 对象。

### 4.1 架构设计

```
AbstractToolParser（抽象基类）
├── extract_tool_calls(model_output: str) → List[ToolCall]           # 非流式
├── extract_tool_calls_streaming(delta_text, delta_token_ids) → DeltaMessage  # 流式
└── adjust_request(request) → request                                # 可选：注入约束解码
```

所有 ToolParser 通过 `@ToolParserManager.register_module(["name"])` 注册，对应 `--tool-call-parser` 参数值。

### 4.2 内置 Tool Parser 一览（截至 2026 年 6 月）

| Parser 名称 | 支持模型 | 调用格式 | 并行调用 |
|-------------|----------|----------|----------|
| `hermes` | NousResearch Hermes-2-Pro 及后续系列 | XML 标签（`<tool_call>...</tool_call>`） | ✅ |
| `mistral` | Mistral 系列（v0.3+） | `[TOOL_CALLS][{...}]` 特殊 token | ⚠️（7B 不稳定）|
| `llama3_json` | Llama 3.1/3.2 系列 | JSON，`<|python_tag|>` 格式 | ❌ (Llama 3.x) / ✅ (Llama 4) |
| `llama4_pythonic` | Llama 4 系列 | Python 函数调用语法 | ✅ |
| `granite` | IBM Granite | 自定义 JSON 格式 | ✅ |
| `internlm` | InternLM 系列 | 自定义格式 | ⚠️（不稳定）|
| `jamba` | Jamba 系列 | 自定义格式 | - |
| `xlam` | xLAM 系列 | 自定义格式 | - |
| `deepseek_v3` | DeepSeek-V3 | 自定义 DSML token | - |
| `deepseek_v31` | DeepSeek-V3.1 | 自定义格式 | - |
| `openai` | OpenAI OSS（gpt-oss-20b/120b）| OpenAI 标准格式 | - |
| `kimi_k2` | Kimi-K2-Instruct | 自定义格式 | - |
| `hunyuan_a13b` | Hunyuan-A13B | 自定义格式 | - |
| `cohere_command3` | Command A Reasoning | 自定义格式 | - |
| `pythonic` | Llama 3.2 小模型、ToolACE 等 | Python 函数调用语法 | - |
| `qwen3_xml` | Qwen3-Coder | XML 标签格式 | - |
| `glm45` / `glm47` | GLM-4.5 / GLM-4.7 | 自定义标签 | - |
| `functiongemma` | FunctionGemma | 自定义格式 | - |
| `olmo3` / `gigachat3` / `apertus` | 对应各自模型 | 各自格式 | - |

### 4.3 主要格式类型分析

**JSON 格式（最常见）**

模型输出标准 JSON 结构，Parser 通过正则或状态机解析：
```json
{"name": "get_weather", "arguments": {"city": "Beijing"}}
```

**XML/标签格式（Hermes、Qwen3-Coder）**

```xml
<tool_call>{"name": "search", "arguments": {"query": "vLLM"}}</tool_call>
```

**特殊 Token 格式（Mistral）**

```
[TOOL_CALLS][{"name": "add", "arguments": {"a": 3.5, "b": 4}}]
```

**Pythonic 格式（Llama 4、部分小模型）**

```python
get_weather(city="Beijing", unit="celsius")
```
该格式更接近自然代码，但解析难度更高，对小模型稳定性挑战大。

---

## 五、约束解码机制（Constrained Decoding）

### 5.1 核心原理

约束解码（也称引导解码/结构化输出）在推理时动态计算并应用 token 掩码，确保模型在每个解码步骤只能选择当前上下文语法合法的 token，从根本上保证输出格式合法性。

```
当前解码状态（FSM）
        ↓
计算合法 token 集合（mask）
        ↓
对 logits 应用 mask（-inf 屏蔽非法 token）
        ↓
Softmax + 采样 → 下一个 token
```

### 5.2 vLLM 中的后端实现

vLLM 支持多种约束解码后端，主要包括：

**XGrammar**（当前 V1 引擎默认后端）
- 基于有限状态机（FSM）编译 EBNF 语法
- 低开销，编译时预处理
- 支持 JSON Schema、正则表达式、上下文无关文法

**Outlines**（V0 引擎后端，部分功能）
- 支持 `anyOf` 类型 JSON Schema（`tool_choice=required` 场景）
- V1 引擎的 `anyOf` 支持仍在路线图中

**LLGuidance**（可选后端）
- 微软开发，性能与 XGrammar 相近

### 5.3 Strict Mode（严格模式）

通过 `strict: true` 字段精细控制是否对工具参数进行 Schema 约束：

| tool_choice 模式 | strict 字段 | 约束行为 |
|-----------------|-------------|----------|
| `"required"` / named | 忽略 | 始终启用约束解码 |
| `"auto"` | `true`（至少一个工具）| 启用约束解码 |
| `"auto"` | 未设置 / `false` | 自由生成，解析原始文本 |

环境变量 `VLLM_ENFORCE_STRICT_TOOL_CALLING=true`（默认开启）与 `strict: true` 配合生效。

**最佳实践**：定义工具时使用 OpenAI strict-schema 风格：
- 每个 object 中设置 `additionalProperties: false`
- 所有 `properties` 字段标记为 `required`

---

## 六、Chat Template 机制

Tool Calling 依赖 Jinja2 格式的 Chat Template 将工具定义和历史工具调用结果注入到模型输入的 Prompt 中。

### 6.1 vLLM 提供的内置模板

| 模板文件 | 适用场景 |
|----------|----------|
| `tool_chat_template_hermes.jinja` | Hermes 模型标准模板 |
| `tool_chat_template_mistral.jinja` | Mistral 官方模板（vLLM 适配版，截断 tool_call_id 至 9 位） |
| `tool_chat_template_mistral_parallel.jinja` | Mistral 增强版（含系统提示，并行调用更稳定） |
| `tool_chat_template_llama3.1_json.jinja` | Llama 3.1 调整版 |
| `tool_chat_template_llama3.2_json.jinja` | Llama 3.2（含图像支持） |
| `tool_chat_template_llama4_pythonic.jinja` | Llama 4 Pythonic 格式（推荐） |
| `tool_chat_template_granite.jinja` | IBM Granite（官方模板改进版） |

### 6.2 工具注入行为

当请求中包含 `tools` 字段时，vLLM 无论 `tool_choice` 如何设置，默认都会将工具定义注入到 Prompt 中。模板负责将工具 Schema（JSON Schema 格式）格式化为模型可理解的文本形式。

---

## 七、流式输出（Streaming）支持

Tool Calling 支持流式返回，ToolParser 需同时实现：
- `extract_tool_calls`（完整响应解析）
- `extract_tool_calls_streaming`（增量 delta 解析）

流式场景下，Parser 需维护内部状态，处理分片到达的 JSON/XML 片段，逐步构造完整的 `tool_calls` 对象并通过 SSE（Server-Sent Events）推送给客户端。

---

## 八、与推理模型（Reasoning Model）的集成

随着 DeepSeek-R1、Qwen3 等推理模型的流行，vLLM 新增了对"思维链 + 工具调用"的联合支持：

- 模型生成 `<think>...</think>` 块（推理过程），然后生成工具调用
- vLLM 的 `reasoning-parser` 与 `tool-call-parser` 协同工作，分别提取推理内容和工具调用
- 响应中增加 `reasoning_content` 字段（非 OpenAI 标准）

示例配置：
```bash
vllm serve tencent/Hunyuan-A13B-Instruct \
  --tool-call-parser hunyuan_a13b \
  --reasoning-parser hunyuan_a13b
```

---

## 九、MCP（Model Context Protocol）集成

vLLM 最新版本引入了对 MCP 的原生支持（`vllm/entrypoints/mcp/` 模块），提供标准化的工具上下文协议接口：

- vLLM 可作为 MCP 服务端暴露工具能力
- 支持通过 MCP 工具服务器动态发现和调用工具
- 与 OpenAI Responses API 兼容，通过 `openai_responses_client_with_mcp_tools.py` 示例展示

---

## 十、插件化扩展：自定义 Tool Parser

vLLM 提供了完整的 ToolParser 插件机制，支持为任意模型添加工具调用支持。

### 10.1 插件结构

```python
from vllm.entrypoints.openai.tool_parsers import ToolParser, ToolParserManager
from vllm.transformers_utils.tokenizer import AnyTokenizer

@ToolParserManager.register_module(["my_model"])
class MyModelToolParser(ToolParser):
    def __init__(self, tokenizer: AnyTokenizer):
        super().__init__(tokenizer)
        # 配置初始化，如设置 skip_special_tokens=False

    def extract_tool_calls(self, model_output: str, request) -> ExtractedToolCallInformation:
        # 解析完整响应
        ...

    def extract_tool_calls_streaming(
        self, previous_text, current_text, delta_text,
        previous_token_ids, current_token_ids, delta_token_ids, request
    ) -> Union[DeltaMessage, None]:
        # 流式增量解析
        ...
```

### 10.2 使用自定义插件

```bash
vllm serve <model> \
  --enable-auto-tool-choice \
  --tool-parser-plugin /path/to/my_plugin.py \
  --tool-call-parser my_model
```

---

## 十一、性能与工程实践

### 11.1 性能基准

vLLM 提供专用的工具调用性能基准工具（`vllm benchmark --tool-calling`），可测量：
- 工具调用首 token 延迟（TTFT）
- 吞吐量（tokens/s）
- 约束解码的额外开销

约束解码整体引入 2-10% 的性能开销（取决于 Schema 复杂度和后端选择），但 XGrammar 通过编译时预计算大幅降低了运行时开销。

### 11.2 已知问题与局限

| 问题 | 影响范围 | 说明 |
|------|----------|------|
| Mistral tool_call_id 限制 | Mistral 系列 | tokenizer 要求恰好 9 位 ID，vLLM 自动截断 |
| Hermes 2 Theta 质量退化 | Hermes Theta | merge 步骤导致工具调用能力下降 |
| Llama 3.x 不支持并行调用 | Llama 3.1/3.2 | Llama 4 已修复 |
| 小模型格式错误率高 | Pythonic 格式 | Llama 3.2 1B/3B 频繁生成格式错误 |
| V1 引擎 anyOf 支持缺失 | `tool_choice=required` | 目前仍依赖 V0 + outlines |
| XML 格式约束问题 | Qwen3-Coder | 当前仍强制 JSON 文法，影响 XML 模型 |

---

## 十二、技术路线对比与趋势

### 12.1 工具调用格式演进

```
第一代（ReAct / 文本解析）
→ 模型输出自然语言描述的动作
→ 启发式解析，容易出错

第二代（JSON Schema Function Calling）
→ GPT-4 引领，结构化 JSON 输出
→ vLLM 通过 ToolParser 解析原始文本

第三代（约束解码 + JSON Schema）
→ 引导解码（xGrammar/Outlines）保证格式正确
→ vLLM strict mode 实现

第四代（Pythonic / 代码化）
→ Llama 4、部分新模型采用 Python 函数调用语法
→ 更接近代码执行，减少 schema 转换开销
→ MCP 标准化工具接口
```

### 12.2 与其他框架的对比

| 特性 | vLLM | SGLang | TGI |
|------|------|--------|-----|
| OpenAI Tool Calling 兼容 | ✅ | ✅ | ✅ |
| 约束解码后端 | XGrammar/Outlines/LLGuidance | XGrammar | - |
| 多模型 Parser 支持 | 丰富（20+） | 较少 | 基础 |
| 插件化扩展 | ✅ | 有限 | 有限 |
| Streaming 支持 | ✅ | ✅ | ✅ |
| MCP 集成 | ✅（新增） | 部分 | ❌ |
| 推理模型集成 | ✅ | ✅ | 有限 |

---

## 十三、总结与建议

### 核心结论

vLLM 的 Tool Calling 实现采用了**"模型特化 Parser + 通用约束解码"**的双层架构：
- 上层：模型特化的 ToolParser 处理各厂商格式差异
- 下层：XGrammar/Outlines 提供格式保障的约束解码

这一设计在灵活性（支持任意格式）和正确性（约束解码保障）之间取得了良好平衡。

### 选型建议

**模型选择优先级**（工具调用质量综合考量）：
1. Hermes 系列（Hermes-2-Pro+）：格式稳定，并行调用支持好
2. Llama 4 系列（Pythonic 格式）：Meta 官方最新，并行支持
3. Mistral Large/Nemo：commercial grade 工具调用能力
4. Qwen3-Coder（XML 格式，注意当前约束解码兼容性问题）
5. DeepSeek-V3 系列：代码能力强，工具调用质量高

**部署配置建议**：
- 生产环境建议启用 `strict: true` + 约束解码，保障格式正确性
- 并行工具调用场景优先选择 Hermes 或 Llama 4
- 推理+工具调用场景选择 Hunyuan-A13B 或 Cohere Command A Reasoning
- 自定义模型建议参考 Hermes ToolParser 实现，贡献 PR 至 vLLM 社区