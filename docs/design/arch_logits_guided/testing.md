# Logits Processor 框架 + 引导解码（R+S）— 测试

> 单测：`tests/test_logits_processor.py`（15 例，`-m unit`，免 GPU）。

## 运行

```bash
source .venv/bin/activate
pytest tests/test_logits_processor.py -m unit -v
pytest -m unit -q          # 全量 332 passed
```

## 覆盖矩阵

| 用例 | 验证点 |
|---|---|
| `test_default_chain_order` | 默认链顺序 Penalties→BadWords→LogitBias→MinTokens→Guided |
| `test_bias_added` / `test_noop_when_empty` | logit_bias 加偏置 / 空时 no-op |
| `test_bias_flips_greedy` | 偏置改变 greedy 结果 |
| `test_suppress_eos_below_threshold` | min_tokens 未达 → EOS 置 -inf |
| `test_allow_eos_at_threshold` | min_tokens 达到 → 不抑制 |
| `test_penalties_noop` / `test_bad_words_noop` | 包装器无配置时原样返回 |
| `test_basic_choice_flow` | ChoiceGrammar allowed/accept/complete 全流程 |
| `test_prefix_overlap` | 候选互为前缀（完成 + 延伸并存） |
| `test_invalid_token_empties` | 非法 token → 无可行候选 |
| `test_build_grammar_with_tokenizer` | tokenizer 编码候选 + None 透传 |
| `test_mask_to_allowed` | Guided 处理器把非 allowed 置 -inf |
| `test_sampler_constrains_and_advances` | Sampler 受约束选 token 并推进 grammar |
| `test_sampler_two_step_choice` | 两步逐 token 约束（[1,2] 候选） |

## 结果

- `tests/test_logits_processor.py`：15 passed
- 全量 `-m unit`：**332 passed**，无回归（原 penalties/bad_words 行为经 test_sampler 验证不变）。

## 手动 GPU 联调

```python
from nanovllm import LLM, SamplingParams
llm = LLM("~/model/Qwen3-1.7B", enforce_eager=True)
# 引导：输出须为 yes/no 之一
out = llm.generate(["Is the sky blue? Answer:"],
                   SamplingParams(temperature=0, guided_choice=["yes", "no"]))
print(out[0]["text"])   # 预期严格为 "yes" 或 "no"
```
预期：受约束输出严格落在候选集内；logit_bias / min_tokens 同 OpenAI 语义。
