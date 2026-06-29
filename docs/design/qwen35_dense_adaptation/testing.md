# Qwen3.5 dense 适配 · 测试

## 单元测试 `tests/test_qwen35.py`（9 例，CPU fp32，微型随机配置）

| 测试 | 验证点 |
|---|---|
| `test_rmsnorm_zero_weight_is_plain_normalize` | Qwen35RMSNorm weight=0 → 纯归一化 |
| `test_rmsnorm_one_plus_weight` | `(1+w)` 缩放语义 |
| `test_partial_rope_passes_through_tail` | 部分 RoPE：前 rotary_dim 维旋转、尾部直通 |
| `test_recurrent_step_shapes_and_decay` | 门控 Δ-rule 单步形状 + 衰减（g→-∞、β=0 → out≈0） |
| **`test_gdn_prefix_consistency`** | **核心不变量**：整段处理 == 先 T-1 再续算 1（conv+recurrent 状态续算一致） |
| `test_gdn_slot_isolation` | 不同 slot 的序列状态互不串扰 |
| **`test_all_linear_model_prefix_consistency`** | **模型级**：整段 prefill 末位 logits == 分步（prefill+decode）末位（覆盖 state_slots 下发链路） |
| `test_forward_output_shapes` | 前向 / compute_logits 形状 |
| `test_tie_word_embeddings` | lm_head 与 embed_tokens 共享存储 |

前缀一致性是线性注意力正确性的最强本地不变量：递归状态 + 卷积左 context 必须使「分步续算」
严格等价「整段处理」，否则连续批 / chunked prefill / decode 会发散。

> 注：未加载真实权重的微型模块用 `_randomize`（小正态 + A_log/dt_bias 置零）初始化，避免
> `torch.empty` 垃圾值经 `exp` 溢出成 NaN。

## 真实权重端到端（CPU）`example_qwen35.py`
`Qwen3.5-2B`，device=cpu、fp32、`max_num_seqs=4`、greedy、max_tokens=48：

```
Prompt: introduce yourself
→ "Hello! I'm Qwen3.5, the latest large language model developed by Tongyi Lab.
   I'm designed to assist with a wide range of tasks..."

Prompt: What is the capital of France?
→ "The capital of France is **Paris**. Located in the northwestern part of the
   country, Paris is not only the political center but also a major global hub..."
```
输出连贯且事实正确 → 端到端验证：混合层分发、分离投影权重加载、部分 RoPE、输出门、
prefill→decode 递归状态续算、tie embedding、MTP/视觉权重跳过、config.json 回退加载均正确。

## 回归
全量 `pytest tests/`：**355 passed, 4 skipped**（原 346 + 新增 9，4 例 GPU-only 跳过）。
对 `rotary_embedding` / `model_runner` / `config` / `loader` / `context` 的改动未影响既有模型
（Qwen3）与各引擎特性。

## 已知未覆盖
- TP>1 的线性注意力（设计上断言 TP=1）。
- GPU 路径未在本环境实跑（无 GPU）；改动均按 `is_cuda` 参数化，逻辑与 CPU 路径同构。
- 线性注意力叠加投机/async/swap 抢占（设计上不支持）。
