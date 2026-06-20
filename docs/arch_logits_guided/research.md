# Logits Processor 框架 + 引导解码（R+S）对齐 — V1 现状调研

> 对照基准：本机 `/home/cb/work/vllm/vllm` @ tag `v0.15.1`。
> 第一梯队单机功能之 ①②（见 `docs/nano_vs_vllm-架构对比` 的功能分析）：把硬编码的采样前
> logits 修改收敛为可插拔框架，并在其上落地结构化/引导输出。

## 背景：Logits Processor 框架（R） + 引导/结构化输出（S）

**R — 什么是 Logits Processor 框架**：采样前对 logits 的各种修改（惩罚、禁止词、logit_bias、
强制最小长度…）若都硬编码在 Sampler 里，则难扩展、难组合。框架把每种修改抽象成一个可插拔的
`LogitsProcessor`：Sampler 持一个**有序列表**，采样前逐个 `apply(logits, metadata)`；每个处理器
从批级元数据读自己的配置、对相关行生效、无配置时快速 no-op。新增约束 = 加一个处理器。

**S — 什么是引导 / 结构化输出**：让模型输出**严格满足某种格式**（只能是给定候选之一、合法 JSON、
匹配正则/语法）。

**核心思想——逐步 token 掩码 + FSM**：为每个受约束请求维护一个**文法状态机（FSM）**；每步采样前，
由一个 LogitsProcessor 把"当前状态下不允许的 token"的 logit 置 `-inf`（掩码），于是只可能采到合法
token；采样后用该 token **推进 FSM**。文法走到"完成"时只允许 EOS，请求自然停止。

**作用 / 收益**：R 给出统一可插拔的 logits 修改底座（惩罚/bad_words 收编其中 + 新增 logit_bias/
min_tokens）；S 在其上实现受约束生成——服务场景（结构化抽取、工具调用入参）的刚需。nano 自带
零依赖的 ChoiceGrammar（候选集约束）演示完整机制，真 xgrammar/正则后端可按同一 Grammar 接口接入。

## V1 组件

| 组件 | 文件 | 职责 | nano 起点 |
|---|---|---|---|
| Logits Processor 框架 | `v1/sample/logits_processor/` | 可插拔 LogitsProcessor 列表，Sampler 逐个 apply | ❌ Sampler 里硬编码 penalties + bad_words |
| 结构化输出 | `v1/structured_output/`（backend_xgrammar / outlines / guidance / lm_format_enforcer） | 文法 → 每步 token 掩码 | ❌ 无 |
| 引导请求状态 | `v1/structured_output/request.py` | 每请求 FSM 状态，随 token 推进 | ❌ 无 |

## V1 关键机制

- **框架**：`LogitsProcessor` 列表（penalties / min_tokens / logit_bias / min_p / 引导 …），
  Sampler 在采样前依次作用于 logits；每个处理器从 batch 级元数据读配置、对相关行生效。
- **引导**：每个受约束请求持一个文法 FSM；逐步由 logits processor 把"当前状态不允许的
  token"置 -inf；采样后推进 FSM；文法完成后只允许 EOS → 请求停止。
- **后端**：xgrammar（JSON/正则/EBNF，编译为 token mask）、outlines（正则→FSM）等重型依赖。

## 与 nano 的差距（本轮范围）

| V1 特性 | 是否对齐 | 说明 |
|---|---|---|
| LogitsProcessor 框架（ABC + 有序链 + batch 元数据驱动） | ✅ | `sample/logits_processor/` |
| 内置 penalties / bad_words 收编进框架 | ✅ | 包装现有 ops |
| logit_bias / min_tokens | ✅ | 新增，OpenAI 兼容 |
| 引导解码（文法 → 逐步 token 掩码 → 采样后推进 FSM） | ✅ | `sample/guided/`：ChoiceGrammar（候选集约束，trie） |
| xgrammar / outlines / 正则 / JSON / EBNF 后端 | ❌→自包含 | 实现 choice 后端演示机制，零额外依赖、纯 CPU 可测；真后端可按 Grammar 接口接入 |
| 跨进程引导状态 | ❌ | 文法对象挂 Sequence、引擎进程内推进 → **仅 UniProc** |
