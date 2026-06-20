# Logits Processor 框架 + 引导解码（R+S）对齐 — V1 现状调研

> 对照基准：本机 `/home/cb/work/vllm/vllm` @ tag `v0.15.1`。
> 第一梯队单机功能之 ①②（见 `docs/nano_vs_vllm-架构对比` 的功能分析）：把硬编码的采样前
> logits 修改收敛为可插拔框架，并在其上落地结构化/引导输出。

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
