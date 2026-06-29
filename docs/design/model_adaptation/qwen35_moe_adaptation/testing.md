# Qwen3.5-35B-A3B MoE 适配 测试文档

## 测试环境

- GPU：8GB 显存（WSL2）
- venv：`/home/cb/work/vllm/nano-vllm/.venv`
- 模型：`/home/cb/model/Qwen3.5-35B-A3B-3L/`（减层 config：3 层 GDN，软链接原始权重）
- 分支：`phase3_model_adapt`

## 测试用例清单

| 测试文件 | 测试类型 | 测试点 | 结果 |
|---------|---------|--------|------|
| test_qwen35_moe.py::TestExpertWeights | Unit | 专家权重形状/参数命名 | ✅ |
| test_qwen35_moe.py::TestSharedExpertMLP | Unit | 共享专家前向形状 | ✅ |
| test_qwen35_moe.py::TestQwen35MoEFFN | Unit | MoE 输出形状/top-k 路由/确定性 | ✅ |
| test_qwen35_moe.py::TestGDNWithNvGtNk | Unit | GDN nv>nk（35B 场景）prefill/decode/参数维度 | ✅ |
| test_qwen35_moe.py::TestQwen35MoEDecoderLayer | Integration | 解码层（GDN+MoE）前向 | ✅ |
| test_qwen35_moe.py::TestQwen35MoEModel | Integration | 3 层模型前向 + logits + 参数路径对齐 | ✅ |
| test_qwen35.py（全部） | Regression | 2B 模型回归（GDN nk→nv 修改无破坏） | ✅ |
| tests/（其余 129 项） | Regression | 全套件回归 | ✅ |
| e2e_qwen35_moe_3l.py | E2E | 3L 模型 GPU 加载 + 生成 | （待填） |

## 验收标准对照

| 验收标准（来自 PRD）| 测试方法 | 实测值 | 是否达标 |
|-------------------|---------|--------|---------|
| config.json 正确解析为 qwen3_5_moe_text | Config 加载 + model_runner 分支 | 解析成功 | ✅ |
| 8GB GPU 加载不 OOM | E2E 运行 | （待填） | （待填） |
| prefill + decode 不报错 | Unit + E2E | CPU 通过 | （待填 GPU） |
| 输出非全空白/乱码 | E2E 生成检查 | （待填） | （待填） |
| top-8 路由正确 | test_top_k_routing | 每 token 恰好 8 个互异专家 | ✅ |

## 已知局限

1. **仅 3 层 GDN**：8GB 显存上限决定，输出质量无法与全 40 层相比，仅验证架构正确性。
2. **MoE dispatch 为 Python 循环**：每步遍历 256 专家（空专家跳过），未做 kernel 融合，性能非目标。
3. **无 full_attention 层**：3 层配置全为 linear_attention，KV cache 路径（num_kv_layers=0 提前返回）已覆盖，但 full_attention+MoE 组合仅在 CPU 单元测试覆盖。
4. **测试发现的存量 flaky**：`test_qwen35.py` 的 `_make_gdn` 此前用未初始化权重（torch.empty），偶发 `A_log` 垃圾值导致状态恒零，已修复为显式初始化。

## 修复记录

- **GDN nk→nv 维度 Bug**（本次发现并修复）：`in_proj_b/in_proj_a/A_log/dt_bias` 实际权重为 nv-sized（35B: nv=32≠nk=16），原代码用 nk。2B 因 nk=nv=16 未暴露。同步修正 `_recurrent_step` 不再扩展 g/beta（已是 nv-sized），只扩展 k/q。
