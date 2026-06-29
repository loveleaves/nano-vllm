# CPU 执行后端 — 测试（轮次 U）

## 单元测试：`tests/test_cpu_backend.py`（8 例，纯 CPU，纳入 `pytest -m unit`）

### 1. Config 门控（6 例，`TestCPUConfigGating`）
- `test_cpu_forces_eager`：`device="cpu", enforce_eager=False` → 构造后 `enforce_eager` 被强制为 True。
- `test_cpu_rejects_mp`：`device="cpu"` + `distributed_executor_backend="mp"` → AssertionError。
- `test_cpu_rejects_tp`：`device="cpu"` + `tensor_parallel_size=2` → AssertionError。
- `test_cpu_rejects_swap`：`device="cpu"` + `num_swap_blocks=8` → AssertionError。
- `test_invalid_device`：`device="tpu"` → AssertionError。
- `test_cuda_default_unchanged`：默认 `device=="cuda"` 且 `enforce_eager==False`（GPU 路径不被影响）。

### 2. 端到端：微型 Qwen3 on CPU（2 例）
在 `tmp_path` 下搭一个**真**微型 Qwen3（`config.json` model_type=qwen3 + 2 层/hidden 32/vocab 100
的 `model.safetensors`，权重以 **HF 命名**落盘 → 经 `weight_loader` 还原 qkv/gate_up），
用 `device="cpu"` 起 `EngineCore` 跑完整 prefill+decode：
- `test_cpu_end_to_end_runs_without_gpu`：贪心生成 8 个 token，断言数量与取值范围——证明
  **无 GPU、无 flash-attn、无 Triton** 也能跑通（KV cache 分配、SDPA 注意力、naive KV 写入、采样）。
- `test_cpu_greedy_deterministic`：同权重两次构建 + 贪心生成 → 逐 token 一致（CPU 后端可复现）。

> 权重落盘用 HF 命名而非 nano 命名的原因：nano 的融合参数名 `qkv_proj` **包含** 子串 `v_proj`，
> 若直接用 nano 名落盘，`load_model` 的 packed 映射会把整块 qkv 误当作 v 分片加载而尺寸不符。
> 故拆成 `q_proj/k_proj/v_proj`、`gate_proj/up_proj` 的 HF 名，正好也覆盖真实 weight_loader 路径。

## 回归

`pytest -m unit` → **346 passed, 4 deselected**（U 前 338 + 本轮 8）。

`sampler.py` 的 `copy=True` 改动影响所有采样路径，全量绿证明无回归（GPU 上 bf16→fp32 本就拷贝，
语义/开销不变）。GPU 默认路径（`device="cuda"`）的所有断言全部裹在 `is_cuda` 分支内，逐字节不变。

## 未覆盖（需真实环境）

- 真实模型（如 Qwen3-0.6B）在 CPU 上的输出连贯性 / 与 GPU 的数值一致性：走 `example_cpu.py` 手测
  （`NANO_VLLM_MODEL=/path python example_cpu.py`）。
- CPU 上的吞吐：CPU 算力/带宽远低于 GPU，仅适合小模型 / 短序列功能验证。
