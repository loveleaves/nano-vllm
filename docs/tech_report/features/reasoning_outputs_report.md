# vLLM Reasoning Outputs 技术报告

> 基于 vLLM 官方文档（stable / latest）及 GitHub 源码分析
> 报告日期：2026-06

---

## 核心内容概要

**1. 背景与概述** — 说明为何需要 Reasoning Outputs 功能，以及 vLLM 从 v0.7.x 开始引入该特性的动因。

**2–3. 核心概念与模型矩阵** — 解释 `reasoning` 字段的含义与整体数据流，并列出当前支持的 13+ 模型，标注各自的 Parser 名称、结构化输出/工具调用兼容性、默认思考模式开关。

**4. 技术架构** — 分析两条核心技术路线：
- **路线一**：基于特殊标签的后处理解析（主流，模型无关）
- **路线二**：基于 Token ID 的边界感知（用于结构化输出集成）

**5. ReasoningParser 详解** — 深入分析抽象接口设计、DeepSeek R1 的流式/非流式解析逻辑、Granite 的特殊文本分隔符机制，以及懒加载注册系统。

**6. Reasoner 与结构化输出集成** — 说明 `Reasoner` 抽象层如何让 xgrammar 等引擎感知推理边界，避免在 `<think>` 阶段错误施加 JSON Schema 约束。

**7. Thinking Budget 控制** — 讲解服务端强制截断机制、`thinking_token_budget` 采样参数、`reasoning_end_str` 过渡短语设计。

**8. 三级优先级开关体系** — 模型默认 → 服务器级 `--default-chat-template-kwargs` → 请求级覆盖，以及 `reasoning_effort` 的自动注入机制。

**9–11. API 使用、工具调用协同、扩展新模型** — 完整的代码示例与步骤指南。

**12–13. 限制与演进路线** — 总结当前约束，并分析从 v0.7 到现在的架构演进趋势（统一接口、推理感知采样、多模态扩展）及潜在改进方向。

---

## 1. 背景与概述

随着 DeepSeek-R1、Qwen3、Gemma 4 等**推理增强型大语言模型（Reasoning LLM）**的兴起，模型在生成最终答案前会产生大量中间推理过程（Chain-of-Thought），这些内容通常被特殊标签（如 `<think>...</think>`）包裹。

vLLM 作为高吞吐、低延迟的 LLM 推理引擎，从 v0.7.x 开始引入了 **Reasoning Outputs** 功能，使其能够：

- 将模型输出中的推理过程与最终答案**自动分离**
- 通过兼容 OpenAI API 的接口将推理内容单独暴露为 `reasoning` 字段
- 支持流式（Streaming）和非流式两种场景
- 与结构化输出（Structured Outputs）、工具调用（Tool Calling）等特性协同工作

---

## 2. 核心概念

### 2.1 Reasoning Field

推理模型在输出中额外返回一个 `reasoning` 字段（旧版本曾使用 `reasoning_content`，已于新版迁移为 `reasoning`），结构如下：

```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "reasoning": "<模型推理过程>",
      "content": "<最终答案>"
    }
  }]
}
```

`reasoning` 字段由 vLLM 服务端通过 **ReasoningParser** 从原始模型输出中提取，客户端无需感知模型内部的特殊标签。

### 2.2 整体数据流

```
用户请求
  │
  ▼
vLLM Chat Completion Endpoint (/v1/chat/completions)
  │
  ├─ Chat Template 渲染（插入 enable_thinking 等 kwargs）
  │
  ▼
LLM 推理引擎（PagedAttention + Continuous Batching）
  │
  ▼
原始 Token 序列（包含 <think>...</think> 等特殊标签）
  │
  ▼
ReasoningParser（按模型类型分发）
  │
  ├─ extract_reasoning()         非流式
  └─ extract_reasoning_streaming() 流式
  │
  ▼
分离后的 reasoning + content
  │
  ▼
Chat Completion Response（reasoning 字段 + content 字段）
```

---

## 3. 支持的模型矩阵

vLLM（截至 2026 年 6 月 stable 版本）支持以下推理模型：

| 模型系列 | Parser 名称 | 结构化输出 | 工具调用 | 思考默认启用 |
|---|---|---|---|---|
| Cohere Command A Reasoning | `cohere_command3` | json, regex | ✅ | 是 |
| DeepSeek R1 系列 | `deepseek_r1` | json, regex | ❌ | 是 |
| Gemma 4 系列 | `gemma4` | json, regex | ✅ | **否**（需显式开启） |
| DeepSeek-V3.1 | `deepseek_v3` | json, regex | ❌（仅非思考模式） | **否** |
| ERNIE-4.5-VL 系列 | `ernie45` | json, regex | ❌ | 是 |
| ERNIE-4.5-21B-A3B-Thinking | `ernie45` | json, regex | ✅ | 是 |
| GLM-4.5 系列 | `glm45` | json, regex | ✅ | 是 |
| Holo2 系列 | `holo2` | json, regex | ✅ | **是**（需显式关闭） |
| Hunyuan A13B 系列 | `hunyuan_a13b` | json, regex | ✅ | 是 |
| IBM Granite 3.2 | `granite` | ❌ | ❌ | **否** |
| MiniMax-M2 | `minimax_m2_append_think` | json, regex | ✅ | 是 |
| Qwen3 系列 | `qwen3` | json, regex | ✅ | **是**（可显式关闭） |
| QwQ-32B | `deepseek_r1` | json, regex | ✅ | 是 |

**关键差异总结：**

- 多数模型默认开启推理（`<think>` 标签自动生效）
- Gemma 4、DeepSeek-V3.1、IBM Granite 3.2 需要通过 `chat_template_kwargs` 显式启用
- IBM Granite 3.2 当前不支持结构化输出与工具调用（受限于其推理分隔符实现方式）

---

## 4. 技术架构与实现路线

### 4.1 整体架构分层

```
┌─────────────────────────────────────────────────────────┐
│                   vLLM Server Layer                      │
│  CLI: --reasoning-parser <name>                          │
│        --reasoning-config <json>                         │
│        --default-chat-template-kwargs <json>             │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│              Chat Completion Serving Layer               │
│  vllm/entrypoints/openai/chat_completion/serving.py      │
│  - 构建 chat template，注入 enable_thinking              │
│  - 调用 ReasoningParser 进行内容分离                    │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│              ReasoningParser 抽象层                      │
│  vllm/reasoning/abs_reasoning_parsers.py                 │
│  - ReasoningParser（抽象基类）                          │
│  - ReasoningParserManager（注册与工厂）                 │
└──────────────────────┬──────────────────────────────────┘
                       │
          ┌────────────┴────────────┐
          │                         │
┌─────────▼──────────┐   ┌─────────▼──────────┐
│ 非流式解析          │   │ 流式解析            │
│ extract_reasoning() │   │ extract_reasoning_  │
│                    │   │ streaming()         │
└────────────────────┘   └────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│               Reasoner 抽象层（结构化输出集成）          │
│  vllm/reasoning/abs_reasoning_parsers.py                 │
│  - Reasoner（用于 xgrammar 等结构化输出引擎感知边界）   │
└─────────────────────────────────────────────────────────┘
```

### 4.2 两大核心技术路线

vLLM 在 Reasoning Outputs 上采用了两条并行技术路线：

**路线一：基于特殊标签的后处理解析（Post-processing Parser）**

适用于所有模型，通过解析模型输出中的特殊标签（如 `<think>...</think>`）在**服务层**分离推理内容。这是主流路线，具有：

- 与模型架构无关，纯文本后处理
- 支持流式与非流式
- 不依赖模型训练细节

**路线二：基于 Token ID 的边界感知（Token-level Boundary Detection）**

主要用于**结构化输出（Structured Outputs）集成**场景。`Reasoner` 类通过识别 `end_token_id`（如 `</think>` 对应的 token ID），告知 xgrammar 等结构化输出引擎何时推理结束、何时应用语法约束，避免在推理过程中错误地施加格式限制。

---

## 5. ReasoningParser 机制详解

### 5.1 抽象基类接口

`vllm/reasoning/abs_reasoning_parsers.py` 定义了所有 Parser 必须实现的接口：

```python
class ReasoningParser:
    def extract_reasoning(
        self,
        model_output: str,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> tuple[str | None, str | None]:
        """非流式：从完整输出中提取 (reasoning, content)"""

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        """流式：处理增量 token，返回 delta 消息"""
```

流式解析需要维护**状态**（当前已解析了多少推理内容、是否已过渡到 content 阶段），因此 Parser 是有状态的实例。

### 5.2 DeepSeek R1 Parser 实现原理

DeepSeek R1 系列（包括 QwQ-32B）使用 `<think>` / `</think>` 作为边界标签。其解析逻辑：

**非流式模式：**
```
原始输出: "<think>...推理过程...</think>最终答案"
         ↓ 正则或字符串分割
reasoning = "...推理过程..."
content   = "最终答案"
```

**流式模式：**
通过 `previous_token_ids` 与 `delta_token_ids` 追踪当前处于推理阶段还是答案阶段，每个 delta 分配到对应字段。Qwen3 Parser 特别优化了使用 token ID 的处理速度。

### 5.3 Granite Parser 的特殊性

IBM Granite 系列不使用 XML 风格标签，而是用自然语言分隔符：

```
"Here is my thought process:"  → 推理开始
"Here is my response:"          → 答案开始（支持 "Here's my response:" 变体）
```

由于边界识别依赖字符串匹配而非专用 token，Granite 当前不支持结构化输出集成（`Reasoner` 层无法提供精确的 `end_token_id`）。

### 5.4 Parser 注册机制

vLLM 使用懒加载的注册方式，避免不必要的模块导入：

```python
ReasoningParserManager.register_lazy_module(
    name="deepseek_r1",
    module_path="vllm.reasoning.deepseek_r1_reasoning_parser",
    class_name="DeepSeekR1ReasoningParser",
)
```

启动时通过 `--reasoning-parser deepseek_r1` 触发对应模块的加载与实例化。

---

## 6. Reasoner 抽象与结构化输出集成

### 6.1 问题背景

推理模型在思考阶段输出的内容**不应受到结构化输出约束**（如 JSON Schema 或正则表达式），约束只应作用于 `</think>` 之后的最终答案部分。

如果不加区分地对全部 token 施加语法约束，会导致：
- 推理内容被错误裁剪
- 生成质量显著下降

### 6.2 Reasoner 接口

```python
@dataclass
class Reasoner:
    def is_reasoning_end(self, input_ids: list[int]) -> bool:
        """判断推理是否已结束（用于非流式）"""

    def is_reasoning_end_streaming(
        self,
        input_ids: list[int],
        delta_ids: list[int]
    ) -> bool:
        """判断推理是否已在此次增量中结束（用于流式）"""
```

### 6.3 DeepSeek Reasoner 示例

```python
@dataclass
class DeepSeekReasoner(Reasoner):
    start_token_id: int  # <think> 的 token ID
    end_token_id: int    # </think> 的 token ID

    @classmethod
    def from_tokenizer(cls, tokenizer) -> "Reasoner":
        return cls(
            start_token_id=tokenizer.encode("<think>")[0],
            end_token_id=tokenizer.encode("</think>")[0],
        )

    def is_reasoning_end(self, input_ids: list[int]) -> bool:
        return self.end_token_id in input_ids
```

xgrammar 引擎通过调用 `is_reasoning_end()` 知道何时开始应用 JSON Schema 约束，实现推理与结构化输出的**无缝切换**。

### 6.4 Qwen3 Coder 的特殊配置

Qwen3 Coder 在同时启用推理与结构化输出时需要额外的服务端参数：

```bash
--structured-outputs-config.enable_in_reasoning=True
```

这是因为当推理内容未被单独解析时，xgrammar 会误将推理内容视为结构化输出的一部分。

---

## 7. 思考预算控制（Thinking Budget）

### 7.1 功能原理

支持预算控制的模型（Qwen3、DeepSeek、Nemotron3 等）允许限制推理 token 数量上限。vLLM 的实现机制：

1. Token 计数从 `reasoning_start_str`（如 `<think>`）开始
2. 当推理 token 计数达到 `thinking_token_budget` 时
3. vLLM **强制注入** `reasoning_end_str`（如 `</think>`），终止推理阶段
4. 模型随即切换到答案生成

这是一种**服务端强制截断**机制，而非模型原生的 token budget 功能（部分模型如 Qwen3 原生支持，vLLM 此处提供统一接口）。

### 7.2 配置参数

| 参数 | 类型 | 说明 |
|---|---|---|
| `--reasoning-parser` | CLI | 指定解析器 |
| `--reasoning-config` | CLI JSON | 指定边界字符串 |
| `thinking_token_budget` | 采样参数 | 每请求推理 token 上限 |

`reasoning_end_str` 支持过渡短语，例如：

```
"I have to give the solution based on the reasoning directly now.</think>"
```

这使得推理终止更自然，避免模型产生截断感。

### 7.3 离线推理示例

```python
from vllm import LLM, SamplingParams
from vllm.config import ReasoningConfig

llm = LLM(
    model="Qwen/Qwen3-0.6B",
    reasoning_config=ReasoningConfig(
        reasoning_start_str="<think>",
        reasoning_end_str="I have to give the solution based on the thinking directly now.</think>",
    ),
)

sampling_params = SamplingParams(thinking_token_budget=100)
outputs = llm.chat(messages, sampling_params=sampling_params)
```

---

## 8. 思考模式开关管理

### 8.1 三级优先级体系

vLLM 实现了细粒度的思考模式控制，优先级从高到低：

```
请求级 chat_template_kwargs  >  服务器默认 kwargs  >  模型默认行为
```

### 8.2 服务器级默认配置

```bash
# 对于默认开启思考的模型（如 Qwen3），在服务器级关闭
vllm serve Qwen/Qwen3-8B \
    --reasoning-parser qwen3 \
    --default-chat-template-kwargs '{"enable_thinking": false}'

# 对于默认关闭思考的模型（如 IBM Granite），在服务器级开启
vllm serve ibm-granite/granite-3.2-2b-instruct \
    --reasoning-parser granite \
    --default-chat-template-kwargs '{"thinking": true}'
```

### 8.3 请求级覆盖

```python
response = client.chat.completions.create(
    model=model,
    messages=messages,
    extra_body={"chat_template_kwargs": {"enable_thinking": True}}
)
```

### 8.4 reasoning_effort 自动注入机制

`reasoning_effort` 参数（兼容 OpenAI API）会**自动**将 `enable_thinking` 注入 chat template kwargs：

| `reasoning_effort` 值 | 自动注入 `enable_thinking` |
|---|---|
| `"low"` / `"medium"` / `"high"` | `true` |
| `"none"` | `false` |
| 未设置 | 不注入（保持模型默认） |

若用户已显式设置 `enable_thinking`，则以用户设置为准。

---

## 9. API 使用方式

### 9.1 在线服务（Online Serving）

**启动服务：**
```bash
vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --reasoning-parser deepseek_r1
```

**非流式请求：**
```python
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")
response = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "9.11 和 9.8 哪个大？"}]
)

print("推理过程:", response.choices[0].message.reasoning)
print("最终答案:", response.choices[0].message.content)
```

**流式请求：**
```python
stream = client.chat.completions.create(
    model=model,
    messages=messages,
    stream=True,
)

for chunk in stream:
    reasoning = getattr(chunk.choices[0].delta, "reasoning", None) or None
    content = getattr(chunk.choices[0].delta, "content", None) or None
    # 注意：OpenAI Python 客户端不原生支持 reasoning 流式属性，
    # 需用 getattr 安全访问
```

### 9.2 支持的 API Endpoints

| Endpoint | 说明 | 支持 reasoning |
|---|---|---|
| `/v1/chat/completions` | OpenAI 兼容聊天接口 | ✅ |
| `/v1/messages` | Anthropic Messages API | ✅ |
| `/v1/responses` | Responses API | ✅ |
| `/v1/completions` | 纯文本补全接口 | ❌ |

**注意**：reasoning 内容仅在在线服务（online serving）的上述接口中可用，离线推理（offline inference）的 `LLM.generate()` 接口不直接返回分离的 reasoning 字段，但可通过 `ReasoningConfig` 控制推理行为。

---

## 10. 工具调用与 Reasoning 的协同

### 10.1 工作机制

当同时启用推理解析和工具调用时：

- 推理内容（`<think>...</think>` 内）**不参与工具调用解析**
- 工具函数调用仅从 `content` 字段中提取
- `reasoning` 字段仍正常返回推理过程

### 10.2 示例

```python
response = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "旧金山天气怎么样？"}],
    tools=tools,
    tool_choice="auto",
)

print("推理过程:", response.choices[0].message.reasoning)
print("工具调用:", response.choices[0].message.tool_calls)
```

### 10.3 兼容性注意事项

- DeepSeek R1 系列：不支持工具调用（`❌`）
- DeepSeek-V3.1：工具调用仅在**非思考模式**下可用
- IBM Granite 3.2：不支持工具调用
- Qwen3、GLM-4.5、Holo2 等：完整支持推理 + 工具调用

---

## 11. 扩展：支持新模型

### 11.1 实现步骤

**Step 1: 创建 ReasoningParser 子类**

```python
from vllm.reasoning import ReasoningParser, ReasoningParserManager

class MyModelParser(ReasoningParser):
    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        self._in_reasoning = False  # 流式状态追踪

    def extract_reasoning(self, model_output, request):
        # 非流式：用正则或字符串分割提取
        # 返回 (reasoning_str, content_str)
        ...

    def extract_reasoning_streaming(
        self, previous_text, current_text, delta_text,
        previous_token_ids, current_token_ids, delta_token_ids
    ):
        # 流式：返回 DeltaMessage，设置 reasoning 或 content
        ...

# 注册
ReasoningParserManager.register_lazy_module(
    name="my_model",
    module_path="my_package.my_model_parser",
    class_name="MyModelParser",
)
```

**Step 2: 实现 Reasoner（如需支持结构化输出）**

```python
from dataclasses import dataclass
from vllm.reasoning import Reasoner

@dataclass
class MyModelReasoner(Reasoner):
    end_token_id: int

    @classmethod
    def from_tokenizer(cls, tokenizer):
        return cls(end_token_id=tokenizer.encode("</think>")[0])

    def is_reasoning_end(self, input_ids):
        return self.end_token_id in input_ids

    def is_reasoning_end_streaming(self, input_ids, delta_ids):
        return self.end_token_id in delta_ids
```

**Step 3: 启动服务**

```bash
vllm serve <model_tag> --reasoning-parser my_model
```

---

## 12. 当前限制

| 限制项 | 说明 |
|---|---|
| 仅支持在线服务接口 | `/v1/chat/completions`、`/v1/messages`、`/v1/responses`；离线 `generate()` 不直接返回分离字段 |
| IBM Granite 3.2 不支持结构化输出 | 其文本标签边界无法提供精确 token ID |
| DeepSeek R1 系列不支持工具调用 | 模型训练限制 |
| Qwen3 Coder 的特殊配置需求 | 需额外传入 `--structured-outputs-config.enable_in_reasoning=True` |
| OpenAI 客户端流式支持不完整 | 需使用 `getattr` 访问 `reasoning` 字段，或改用 `requests` 库 |
| reasoning_effort 参数行为差异 | 对未声明 `enable_thinking` 的模型（如 DeepSeek R1），注入的 kwarg 会被静默过滤 |

---

## 13. 演进路线分析

### 13.1 架构演进（从 v0.7 到当前 stable）

| 版本阶段 | 关键变化 |
|---|---|
| v0.7.x 初期 | 仅支持 DeepSeek R1，`reasoning_content` 字段，需要 `--enable-reasoning` + `--reasoning-parser` 双标志 |
| v0.8～v0.9 | 字段名迁移为 `reasoning`；移除 `--enable-reasoning`，仅保留 `--reasoning-parser`；扩展支持 Qwen3、Gemma 4 等更多模型 |
| stable（当前） | 完整生态：13+ 模型支持、Thinking Budget、服务器级默认 kwargs、`reasoning_effort` 自动注入、Anthropic/Responses API 支持 |

### 13.2 技术路线趋势

**趋势一：统一控制接口**

从模型各自的 chat template 参数（`enable_thinking`、`thinking`）向标准 OpenAI `reasoning_effort` 参数收敛，通过自动注入机制屏蔽底层差异。

**趋势二：推理感知的结构化输出**

`Reasoner` 抽象层的引入使结构化输出引擎（xgrammar）能够感知推理边界，是推理模型与受限采样（constrained decoding）深度集成的基础。

**趋势三：Token Budget 精细化控制**

`thinking_token_budget` 采样参数的引入，使开发者可以在**推理质量**与**推理成本**之间进行细粒度权衡，是面向生产部署的重要特性。

**趋势四：多模态与推理融合**

Gemma 4（视觉-语言模型）的推理支持、ERNIE-4.5-VL 系列的加入，表明推理能力正在向多模态模型扩展。

### 13.3 潜在改进方向

- **离线推理中直接返回 reasoning 字段**：当前离线推理无法直接获取分离的 reasoning，需要用户自行解析
- **推理内容的 KV Cache 优化**：大型推理过程会占用大量 KV Cache，前缀缓存（Automatic Prefix Caching）对推理场景的适配值得关注
- **推理内容的 token 计费**：当前 usage 统计是否将推理 token 计入 `completion_tokens` 尚需明确
- **多轮对话中 reasoning 字段的处理**：推理内容是否应作为历史上下文传回模型

---

## 14. 总结

vLLM 的 Reasoning Outputs 功能构建了一套完整的、可扩展的推理内容处理框架：

**核心设计亮点：**

1. **关注点分离**：推理内容解析（ReasoningParser）与结构化输出约束（Reasoner）分层设计，各司其职
2. **可插拔架构**：通过 `ReasoningParserManager` 的懒加载注册机制，新模型支持无需修改核心代码
3. **三级优先级控制**：模型默认 → 服务器默认 → 请求级覆盖，灵活适配不同部署场景
4. **OpenAI API 兼容**：通过 `reasoning_effort` 等标准参数提供统一接口，降低迁移成本
5. **流式与非流式统一**：两种接口模式均有完整实现，流式解析通过状态机管理增量 token

**适用建议：**

- 生产部署推理模型时，建议使用 `--default-chat-template-kwargs` 在服务器级统一配置思考模式，避免客户端逐请求配置的复杂性
- 需要控制推理成本时，优先使用 `thinking_token_budget` 结合合适的 `reasoning_end_str` 过渡短语
- 同时需要结构化输出和推理的场景，确认所选模型的 Parser 已实现 `Reasoner` 接口（见支持矩阵）
- 扩展新模型时，优先实现 token ID 级别的边界检测，以获得最完整的功能支持

---

*参考资料：*
- *vLLM Docs (stable): https://docs.vllm.ai/en/stable/features/reasoning_outputs/*
- *vLLM Docs (latest): https://docs.vllm.ai/en/latest/features/reasoning_outputs/*
- *GitHub 源码: https://github.com/vllm-project/vllm/blob/main/docs/features/reasoning_outputs.md*
- *vllm.reasoning API: https://docs.vllm.ai/en/v0.9.2/api/vllm/reasoning/*