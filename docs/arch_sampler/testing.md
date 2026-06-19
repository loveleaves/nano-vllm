# 采样层对齐 — 测试设计

## 测试矩阵

| 文件 | 覆盖点 | 依赖 |
|---|---|---|
| `test_sampler.py`（重写） | greedy=argmax/int64；混合 greedy+random 行；随机形状/范围/温度对 argmax 一致率；apply_top_k_only/top_p 掩码；top_k 限制采样集合；repetition(prompt∪output)/frequency/presence 数值；no_penalties 跳过；logprobs 形状/采样列对齐/greedy 排名=1 | CPU |
| `test_config.py`（改） | temperature=0 合法（greedy）；top_p 越界报错 | CPU |
| `test_sequence.py` / 引擎链 | 不变（采样配置不入 __getstate__） | CPU |
| `example.py` 派生 E2E 校验 | temperature=0 真·greedy **两次生成 token 完全一致（确定性）**；top_k+top_p+三类惩罚混合路径产出连贯、不崩 | GPU |

## 关键校验

- **真·greedy 确定性**：`temperature=0` 两次 `generate` 的 `token_ids` 逐 token 相等
  （旧版 temperature=0.01 近似 greedy 无此保证）。
- **数值精确**：repetition rep=2、logit=1 → 0.5；freq=0.25×2 + pres=0.5 → -1.0，CPU 上精确比对。
- **批级跳过**：no_penalties / top_p=None / top_k=None 路径产出等于未施加对应变换。

## 回归

- 全量套件：**230 passed, 4 skipped**（test_sampler 重写 8→12；test_config +1；其余不变）。
- `example.py` 两 prompt 仍连贯；greedy 确定性 + 惩罚路径 E2E 验证采样层端到端可用。

## 限制

- logprobs 仅在 Sampler 层验证（产出 `LogprobsTensors`），未串到 RequestOutput（见 design.md 边界）。
- 无逐请求种子，随机采样不保证 per-seq 可复现。
