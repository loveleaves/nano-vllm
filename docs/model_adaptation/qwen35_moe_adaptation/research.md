# Qwen3.5-35B-A3B MoE 适配 nano-vllm 技术调研报告

## 摘要

Qwen3.5-35B-A3B 是 Qwen3.5-2B 的 MoE 变体，模型结构高度相似（相同的 GDN 线性注意力 + 全注意力混合架构），关键差异在于：
1. FFN 由 dense MLP 替换为 **MoE（256 专家，top-8 激活 + 1 共享专家）**
2. GDN 的 `in_proj_b/in_proj_a/A_log/dt_bias` 维度为 `nv=32` 而非 `nk=16`（2B 模型 nk=nv=16 故未暴露）
3. 全量 35B 权重 ≈ 70GB BF16，8GB 显存仅能容纳 3 层（约 6.65GB）

---

## 参照实现对比

### 1. vllm `qwen3_next.py`（主要参照）

**文件路径**：`/home/cb/work/vllm/vllm/vllm/model_executor/models/qwen3_next.py`

**MoE 块实现**（`Qwen3NextSparseMoeBlock`）：
- `gate`：`ReplicatedLinear(hidden, num_experts)` — 路由器，输出 logit
- `shared_expert_gate`：`ReplicatedLinear(hidden, 1)` — 共享专家的标量门控
- `shared_expert`：`Qwen3NextMLP`（标准 SwiGLU）— 始终激活的 shared expert
- `experts`：`FusedMoE`（CUDA kernel 融合实现）— 256 routed experts

**路由逻辑**：
```
router_logits = gate(hidden)         # [T, 256]
final = FusedMoE(hidden, router_logits)  # top-8 + shared_expert 内嵌
```
vllm 用 FusedMoE kernel 一次性完成 topk 选择 + expert 计算 + 共享专家合并，nano-vllm 无此 kernel，需 Python 朴素实现。

**权重命名**（与 safetensors 实际 key 完全对应）：
```
mlp.experts.gate_up_proj  [256, 1024, 2048]  # 已 packed
mlp.experts.down_proj     [256, 2048, 512]
mlp.gate.weight           [256, 2048]
mlp.shared_expert.gate_proj.weight  [512, 2048]  # 需 packed_modules_mapping 合并
mlp.shared_expert.up_proj.weight    [512, 2048]
mlp.shared_expert.down_proj.weight  [2048, 512]
mlp.shared_expert_gate.weight       [1, 2048]
```

**注意**：vllm 没有单独的 `Qwen3_5MoE` 文件，35B 复用 `qwen3_next.py`（该文件同时支持 Qwen3-Next 系列）。

---

### 2. 当前 nano-vllm `qwen35.py`（现有代码）

已有的 GDN (`GatedDeltaNet`) 和全注意力 (`Qwen35Attention`) 实现基本正确，可直接复用。

**发现的 Bug（35B 才暴露）**：
```
GatedDeltaNet.__init__:
  self.in_proj_b = ColumnParallelLinear(hidden, nk, ...)   # 应为 nv
  self.in_proj_a = ColumnParallelLinear(hidden, nk, ...)   # 应为 nv
  self.A_log     = nn.Parameter(torch.empty(nk))           # 应为 nv
  self.dt_bias   = nn.Parameter(torch.empty(nk))           # 应为 nv
```
实际权重形状（从 35B safetensors 验证）：
- `linear_attn.in_proj_b.weight: [32, 2048]` → 32 = nv，而非 nk=16
- `linear_attn.A_log: [32]` → 32 = nv

在 2B 中 nk=nv=16 故未暴露；35B 中 nk=16, nv=32 故必须修正。

同时，`_recurrent_step` 中对 g/beta 的扩展逻辑需移除（原来 nk=nv 时扩展因子=1 无影响，现在 g/beta 已是 nv-sized，再扩展会出错）：
```python
# 现在的（错误）
g = g.repeat_interleave(ratio)       # 35B: g 已是 [nv=32]，再扩展→[64]！
beta = beta.repeat_interleave(ratio) # 同上错误

# 修正后
# 只扩展 k（和 q），不扩展 g/beta
```

---

### 3. 权重文件实际结构（35B 完整验证）

从 safetensors 实测各关键 key 形状：

| 权重 key | shape | 说明 |
|---------|-------|------|
| `layers.0.linear_attn.in_proj_qkv.weight` | [8192, 2048] | nk×dk×2 + nv×dv = 8192 |
| `layers.0.linear_attn.in_proj_z.weight` | [4096, 2048] | nv×dv = 32×128 |
| `layers.0.linear_attn.in_proj_b.weight` | **[32, 2048]** | nv=32 (非 nk=16) |
| `layers.0.linear_attn.in_proj_a.weight` | **[32, 2048]** | nv=32 |
| `layers.0.linear_attn.A_log` | **[32]** | nv=32 |
| `layers.0.linear_attn.conv1d.weight` | [8192, 1, 4] | conv_dim |
| `layers.0.linear_attn.out_proj.weight` | [2048, 4096] | nv×dv=4096 |
| `layers.0.mlp.experts.gate_up_proj` | **[256, 1024, 2048]** | 已 packed |
| `layers.0.mlp.experts.down_proj` | **[256, 2048, 512]** | 已 packed |
| `layers.0.mlp.gate.weight` | [256, 2048] | 路由器 |
| `layers.0.mlp.shared_expert.gate_proj.weight` | [512, 2048] | 需 mapping |
| `layers.0.mlp.shared_expert.up_proj.weight` | [512, 2048] | 需 mapping |
| `layers.0.mlp.shared_expert.down_proj.weight` | [2048, 512] | |
| `layers.0.mlp.shared_expert_gate.weight` | [1, 2048] | 标量门 |
| `layers.3.self_attn.q_proj.weight` | [8192, 2048] | 含输出门 |
| `layers.3.self_attn.k_proj.weight` | [512, 2048] | num_kv=2 |
| `layers.3.self_attn.v_proj.weight` | [512, 2048] | |
| `layers.3.self_attn.o_proj.weight` | [2048, 4096] | |

---

## 当前代码库分析

### 各模块职责及改动范围

| 文件 | 现状 | 需要改动 |
|------|------|---------|
| `nanovllm/models/qwen35.py` | GDN + Attention + dense MLP | 修正 GDN nk/nv Bug |
| `nanovllm/models/qwen35_moe.py` | 不存在 | **新建**：MoE FFN + 完整 MoE 模型类 |
| `nanovllm/config.py` | 识别 `qwen3_5` → text_config | 补充 `qwen3_5_moe` 同样提取 |
| `nanovllm/engine/model_runner.py` | 注册 `qwen3_5_text` | 补充 `qwen3_5_moe_text` |
| `nanovllm/engine/model_runner.py` | `allocate_kv_cache` | 修复 num_kv_layers=0 时除零 |

### 权重加载兼容性

当前 loader (`utils/loader.py`) 设计足够通用：
- `weight_prefix_to_strip = "model."` → 35B 与 2B 相同，直接复用
- `weight_skip_prefixes = ("model.visual.", "mtp.")` → 35B 与 2B 相同
- `packed_modules_mapping` → 35B 额外需要 `shared_expert.gate_proj` → `shared_expert.gate_up_proj`
- `mlp.experts.gate_up_proj`（已 packed）→ 直接 copy，无需 mapping

---

## 显存预算（BF16 精确计算）

**单层 GDN（nv=32）**：
- linear_attn 权重: ~101MB
- MoE FFN: gate_up [256,1024,2048]=1024MB + down [256,2048,512]=512MB + 其他≈7MB = ~1543MB
- **小计: ~1644MB ≈ 1.6GB**

**单层 full_attention**：  
- 注意力权重: ~88MB
- MoE FFN: ~1543MB
- **小计: ~1631MB ≈ 1.6GB**

**全局固定**：
- Embedding [248320, 2048]: ~970MB
- LM head [248320, 2048] (no tie): ~970MB
- Final norm: 可忽略
- **固定: ~1.94GB**

**层数 → 显存估算**：

| 层配置 | 模型权重 | 是否可行（8GB） |
|--------|---------|---------------|
| 3×GDN | 3×1.6+1.94 ≈ **6.74GB** | ✅ 有~1.26GB余量 |
| 4×GDN | 4×1.6+1.94 ≈ **8.34GB** | ❌ 超出 8GB |
| 3×GDN+1×full | 4×1.6+1.94 ≈ **8.34GB** | ❌ 超出 8GB |

**结论：8GB 显存最多支持 3 层 GDN（无全注意力层）。**

---

## 结论与选型建议

### 实现方案

1. **新建 `qwen35_moe.py`**：
   - 复用 qwen35.py 的 `Qwen35RMSNorm`、`GatedDeltaNet`（修正后）、`Qwen35Attention`
   - 新增 `ExpertWeights`（存放 packed gate_up/down）、`SharedExpertMLP`、`Qwen35MoEFFN`
   - 用 Python 循环实现 top-8 routing（无需 CUDA kernel）
   - `Qwen35MoEForCausalLM` 包含 `weight_prefix_to_strip`、`weight_skip_prefixes`、`packed_modules_mapping`

2. **修正 `qwen35.py` 的 GDN**：
   - `in_proj_b/in_proj_a/A_log/dt_bias` 改用 `nv` 维度
   - `_recurrent_step` 移除 g/beta 扩展（只扩展 k）
   - 对 2B 模型无影响（nk=nv=16）

3. **减层 config.json**：
   - `num_hidden_layers: 3`，`layer_types: ["linear_attention","linear_attention","linear_attention"]`
   - 放置于新目录 `/home/cb/model/Qwen3.5-35B-A3B-3L/` 并软链到原权重文件

4. **基础设施修复**：
   - `config.py`：补充 `qwen3_5_moe` 顶层 model_type 的 text_config 提取
   - `model_runner.py`：注册 `qwen3_5_moe_text`；修复 `allocate_kv_cache` 在 num_kv_layers=0 时的除零问题

### 风险

| 风险 | 概率 | 缓解 |
|------|------|------|
| 显存实际略超估算 | 中 | 可将 gpu_memory_utilization 调低至 0.85 |
| 3 层模型输出质量很差 | 高 | 符合预期，目的是验证架构而非质量 |
| GDN nv 修正破坏 2B | 低 | nk=nv=16 维度不变，只改变代码读取字段 |
