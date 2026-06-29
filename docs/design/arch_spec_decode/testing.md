# 投机解码（T）— 测试

> 单测：`tests/test_spec_decode.py`（13 例，`-m unit`，免 GPU）。

## 运行

```bash
source .venv/bin/activate
pytest tests/test_spec_decode.py -m unit -v
pytest -m unit -q          # 全量 332 passed
```

## 覆盖矩阵

| 用例 | 验证点 |
|---|---|
| `test_basic_match` | n-gram 命中 → 提议更早出现处之后的 k 个 token |
| `test_prefers_longer_ngram` | 优先更长 n-gram |
| `test_k_limit` | 提议数不超过 k |
| `test_no_match_returns_empty` | 无匹配 → [] |
| `test_short_sequence` | 序列过短 → [] |
| `test_all_accepted_plus_bonus` | 草案全对 → 接受全部 + 奖励 token |
| `test_mismatch_correction` | 分歧处用目标 token 修正并停止 |
| `test_first_token_mismatch` | 首位即分歧 → 仅 1 修正 token |
| `test_empty_draft_single_token` | 无草案 → 退化单 token |
| `test_length_assert` | target 长度须 = draft+1 |
| `test_step_accepts_correct_draft` | 编排：草案全中 + 奖励 |
| **`test_greedy_equivalence`** | **投机贪心序列逐 token == 自回归贪心，且步数 < token 数（加速）** |
| `test_fallback_no_draft` | 非重复序列退化为单 token，仍正确 |

## 引擎集成测试（`tests/test_spec_decode_engine.py`，6 例）

`object.__new__(EngineCore)` + 真 Scheduler（纯 Python 块管理）+ 假 Executor
（execute_model 基准步 + verify_spec 验证），测 `_step_spec` 多 token 扩展（CPU，免 GPU）。

| 用例 | 验证点 |
|---|---|
| `test_accept_all_plus_bonus` | 草案全中 → [base, d0, d1, bonus]；`num_cached == num_tokens-1` 不变式 |
| `test_partial_accept` | 分歧处修正后停止 |
| `test_finish_by_max_tokens_mid_spec` | spec 中途达 max_tokens → LENGTH 截断 + 出请求表 |
| `test_finish_by_eos_mid_spec` | spec 中途遇 EOS → STOP |
| `test_no_draft_falls_back` | 无草案 → 退化普通 decode（1 token） |
| `test_period3_equivalence` | 真 NgramProposer + period-3 目标：**完整 completion == 参考贪心**，步数 < token 数 |

## 结果

- `tests/test_spec_decode.py`（算法/组件）：13 passed
- `tests/test_spec_decode_engine.py`（引擎集成）：6 passed
- 全量 `-m unit`：**338 passed**，无回归（spec 走独立 `_step_spec`，门控默认关闭）。

## 说明

GPU `ModelRunner.verify_spec`（多位置并行前向 + 全位置 lm_head + argmax）为唯一未在 CPU
覆盖的部分，由真实模型 GPU 联调验证；引擎编排 / 块回滚 / num_cached 不变式 / 终止判定 /
贪心等价均由上述 CPU 测试锚定。

## 手动 GPU 联调

```python
from nanovllm import LLM, SamplingParams
llm = LLM("~/model/Qwen3-1.7B", enforce_eager=True, speculative_num_tokens=4)
out = llm.generate(["def fibonacci(n):"], SamplingParams(temperature=0, max_tokens=128))
print(out[0]["text"])   # 与 speculative_num_tokens=0 逐 token 一致（贪心等价），代码类文本加速明显
```
