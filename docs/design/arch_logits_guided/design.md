# Logits Processor 框架 + 引导解码（R+S）— 详细设计

> 基于 `research.md`。R：把 Sampler 里硬编码的 penalties/bad_words 收编为可插拔
> LogitsProcessor 链，并补 logit_bias / min_tokens。S：在该框架上落地引导解码
> （文法 → 逐步 token 掩码），自包含 ChoiceGrammar 后端。

## 范围决策

| V1 特性 | 是否对齐 | 说明 |
|---|---|---|
| LogitsProcessor ABC + 有序链 + 默认装配 | ✅ | `apply(logits, metadata)`，元数据驱动、快速 no-op |
| penalties / bad_words / logit_bias / min_tokens | ✅ | 前二包装现有 ops，后二新增 |
| 引导：Grammar 接口 + ChoiceGrammar + Guided 处理器 | ✅ | 候选集约束（trie/FSM） |
| 重型文法后端（xgrammar/outlines）、JSON/正则/EBNF | ❌ | 自包含 choice 演示机制 |
| 跨进程引导状态 | ❌ | 仅 UniProc（文法不随 Sequence 序列化） |

## Architecture

```
sample/
├── logits_processor/
│   ├── interface.py   # LogitsProcessor(ABC) + build_logits_processors()（有序链）
│   └── builtin.py     # Penalties / BadWords（包装 ops）/ LogitBias / MinTokens
├── guided/
│   ├── grammar.py     # Grammar(ABC) + ChoiceGrammar(trie) + build_grammar(tokenizer)
│   └── processor.py   # GuidedDecodingLogitsProcessor（按行 Grammar 掩码非法 token）
├── metadata.py        # +logit_bias / min_tokens / eos_token_id / grammars
└── sampler.py         # forward: logits.float → 逐 processor.apply → sample → 推进 grammar
```

### 处理器链（顺序敏感）

`build_logits_processors()` →
`[Penalties, BadWords, LogitBias, MinTokens, GuidedDecoding]`

引导置于最后，使文法约束对其它处理器有最终决定权。每个处理器从 SamplingMetadata 读自己
的配置（`logit_bias` / `min_tokens` / `grammars` …），整批无配置即原样返回（零开销）。

### 数据流（引导解码）

```
Processor(tokenizer): guided_choice → build_grammar → ChoiceGrammar → EngineCoreRequest.grammar
   ▼
EngineCore.add_request: seq.grammar = request.grammar
   ▼
ModelRunner.prepare_sample: {row: seq.grammar} → SamplingMetadata.grammars
   ▼
Sampler.forward:
   GuidedDecodingLogitsProcessor.apply → 每行 allowed=grammar.allowed_token_ids()，
       非 allowed 的 logit 置 -inf
   sample()
   _advance_grammars: 每受约束行 grammar.accept(sampled_token)   # 推进 FSM
```

### ChoiceGrammar（FSM）

状态 = 仍可行候选 `(tokens, pos)` 集合：
- `allowed_token_ids()`：可延伸候选的下一 token ∪（有候选已完成时 ∪ {EOS}）；完成后 = {EOS}。
- `accept(tok)`：tok==EOS 且有完成候选 → 标记完成；否则保留下一 token 命中的候选。
- 支持候选互为前缀（`["a","ab"]`：完成与延伸并存）。
- 文法完成后只允许 EOS → 采到 EOS 即正常停止。

### 关键设计点

- **零回归**：Sampler 默认链复刻原有 penalties→bad_words 行为；其余处理器无配置即 no-op，
  greedy / 纯温度路径不受影响。
- **min_tokens / 引导需 output 历史**：prepare_sample 在二者启用时确保 `output_token_ids`
  就绪（与 bad_words 同款分支）。
- **引导仅 UniProc**：grammar 是挂在 Sequence 上的 Python 对象，跨步在引擎进程内原地推进；
  MultiProcExecutor 下 worker 吃反序列化 Sequence（grammar 置 None），故不生效——与
  penalties/async/swap 的隔离边界一致。
- **OpenAI 接入**：`_SamplingMixin` 加 `logit_bias`（{str:float}→{int:float}）/ `min_tokens`
  / `guided_choice`，经 to_sampling_params 透传。
