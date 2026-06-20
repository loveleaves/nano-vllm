# nano-vllm 采样层对齐 V1 — 详细设计

> 基于 `research.md`。目标：把 nano 单函数 `Sampler(logits, temperatures)` 升级为
> V1 风格的**结构化采样层** `layers/sample/`——SamplingMetadata + Sampler + ops
> （真·greedy、top-k/top-p、presence/frequency/repetition 惩罚、logprobs）。

## 范围决策（与 V1 的取舍）

| V1 特性 | 是否对齐 | 说明 |
|---|---|---|
| `SamplingMetadata` 结构化批配置 | ✅ | temperature/top-p/top-k/penalties/历史 token/logprobs |
| 真·greedy（temperature=0 → argmax） | ✅ | 旧版禁止 temperature≤0；现 0=greedy |
| top-k / top-p | ✅ | 原生 sort 路径，逐行阈值 |
| presence / frequency / repetition 惩罚 | ✅ | repetition 纯 torch（V1 走 CUDA 自定义 op） |
| logprobs（top-k + 采样 token + 排名） | ✅ | 取惩罚/温度前原始 logits |
| 逐行 greedy/random 混合批 | ✅ | `where(temp<eps, greedy, random)` |
| generators(逐请求种子) / bad_words / allowed_ids / min-p / logitsprocs / rejection / flashinfer | ❌ | nano 不引入 |

## 包结构

```
nanovllm/layers/sample/
├── __init__.py          # 导出 Sampler / SamplingMetadata / SamplerOutput / LogprobsTensors
├── metadata.py          # SamplingMetadata（按行对齐的批配置）
├── outputs.py           # SamplerOutput / LogprobsTensors
├── sampler.py           # Sampler(nn.Module)：forward + sample
└── ops/
    ├── topk_topp.py      # apply_top_k_top_p / apply_top_k_only / random_sample / TopKTopPSampler
    ├── penalties.py      # apply_all_penalties + bin_counts/mask + repetition(纯 torch)
    └── logprobs.py       # compute_logprobs / gather_logprobs
nanovllm/layers/sampler.py   # 向后兼容垫片 → re-export
```

## 调用链与关键点

```
ModelRunner.run（rank0）:
  sampling_metadata = prepare_sample(ordered)      # 行序序列 → SamplingMetadata
  sampler_output    = self.sampler(logits, sampling_metadata)
  row_tokens        = sampler_output.sampled_token_ids.tolist()
  → 按 seq_id 映射回入参 seqs 顺序返回（沿用 InputBatch 行序→req_id 对齐）

Sampler.forward:
  ① max_num_logprobs 非空 → 先存原始 logits 的 logprobs
  ② float32
  ③ not no_penalties → apply_all_penalties（rep 看 prompt∪output，freq/pres 看 output）
  ④ sample：all_random→纯随机；all_greedy→argmax；混合→where(temp<eps, argmax, 温度+topk/topp 随机)
  ⑤ 请求 logprobs → gather_logprobs（采样 token + top-k + 排名）
```

### prepare_sample 的批级跳过（性能）

`prepare_sample` 把整批配置塌缩成标志：`all_greedy`/`all_random`（按 temperature<eps）、
`top_p=None`（全为 1.0）、`top_k=None`（全关闭，关闭行填 vocab_size）、`no_penalties`
（全 freq/pres=0 且 rep=1）。常见的纯 greedy / 纯温度采样因此走 argmax / softmax+gumbel
最短路径，惩罚与 top-k/p 分支完全不构造张量。

### 配置透传

`SamplingParams` 增加 `top_p / top_k / presence_penalty / frequency_penalty /
repetition_penalty / logprobs`，并放开 `temperature>=0`（0=greedy）。这些经 `Sequence`
字段携带，仅 rank0 `prepare_sample` 读取——**不入 `Sequence.__getstate__`**（rank>0 不采样），
故 TP 序列化零变化。

## 边界（本轮不做）

- **logprobs 端到端输出通路**：Sampler 已产出 `LogprobsTensors`，但未串到
  `EngineCore→OutputProcessor→RequestOutput`（涉及 RPC 载荷与文本输出契约的较大改动）。
  本轮只对齐采样层内部能力 + 单测验证；`run()` 返回仍为 `list[int]`，保持 TP/引擎契约不变。
- 逐请求随机种子（generators）：nano 用全批 Gumbel，无可复现 per-seq 种子。
