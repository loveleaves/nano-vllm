# Qwen3.5-2B 移植到 nano-vllm 技术调研报告

## 摘要

Qwen3.5-2B（内部模型名 Qwen3-Next）是一个混合架构语言模型，将线性注意力（GatedDeltaNet）与全注意力（Full Attention）交替组合，比例约为 3:1（18 层线性 + 6 层全注意力）。vllm 在 v0.21.0（2025 年 9 月）中正式支持该模型，核心实现位于 `qwen3_next.py` 和专用 GDN 内核层。

nano-vllm 仓库已有设计草稿（`docs/model_adaptation/qwen35_2b_adaptation.md`），整体思路与 vllm 上游对齐，但尚无代码实现。本报告在调研 vllm 真实实现后，对草稿进行交叉验证并指出需要补充/修订之处。

---

## 参照实现对比

### 1. vllm v0.21.0（上游参照）

#### 架构与文件布局

| 文件 | 职责 |
|------|------|
| `vllm/model_executor/models/qwen3_next.py` | 混合模型骨干：`Qwen3NextAttention`（全注意力 + 输出门）、`Qwen3NextDecoderLayer`（分层类型分发） |
| `vllm/model_executor/models/qwen3_5.py` | `Qwen3_5ForCausalLMBase` 继承自 `qwen3_next`，处理权重映射 |
| `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py` | `QwenGatedDeltaNetAttention`：线性注意力完整实现 |
| `vllm/model_executor/layers/mamba/gdn/fla_gated_delta_rule.py` | Triton/FLA 内核（chunk 和 recurrent 两路） |

#### 关键设计决策

**层类型分发（Qwen3NextDecoderLayer）**

```python
if layer_type == "linear_attention":
    self.mixer = QwenGatedDeltaNetAttention(config, ...)
elif layer_type == "full_attention":
    self.mixer = Qwen3NextAttention(config, ...)
```

`layer_types` 来自 `config.layer_types`，长度等于 `num_hidden_layers`，值为 `"linear_attention"` 或 `"full_attention"`。

**全注意力（Qwen3NextAttention）**

- `q_proj` 输出 `2 × num_heads × head_dim`：前半为 query，后半为 output gate
- `q_norm` / `k_norm` 使用 **Qwen35RMSNorm**（`(1 + w) * norm(x)`，零初始化权重）
- Partial RoPE：`partial_rotary_factor=0.25`，仅旋转每 head 前 64 维
- 输出门：`out = attn_out * sigmoid(gate)`

**线性注意力（QwenGatedDeltaNetAttention）**

- 投影：`in_proj_qkvz`（packed q+k+v+z）和 `in_proj_ba`（b+a，用于 beta 和 g）
- 卷积：`conv1d`，depthwise，kernel_size=4，conv_dim = 2×k_dim + v_dim = 6144
- 状态：
  - `conv_state`: `[max_seqs, conv_dim, kernel_size]`（dtype = model dtype）
  - `recurrent_state`: `[max_seqs, num_v_heads, k_dim, v_dim]`（dtype = float32）
- 执行路径：prefill → `ChunkGatedDeltaRule`（Triton/FLA）；decode → recurrent

**状态管理（vllm 方式）**

vllm 通过 `compilation_config.static_forward_context[prefix] = self` 将 GDN 实例注册到编译框架，由 MambaStateShapeCalculator 统一管理状态形状，支持 CUDA graph。

**hybrid KV cache manager**

vllm 对两类层分别管理：
- 全注意力层：标准 PagedAttention KV cache
- 线性注意力层：固定大小状态张量（与序列长度无关）

#### packed_modules_mapping（Qwen3.5 文本模型）

```python
packed_modules_mapping = {
    "qkv_proj":    ["q_proj", "k_proj", "v_proj"],
    "gate_up_proj": ["gate_proj", "up_proj"],
    # GDN 特有 packed 投影：
    "in_proj_qkvz": ["in_proj_qkvz"],  # 已经是 packed，直接加载
    "in_proj_ba":   ["in_proj_ba"],
}
```

注：Qwen3.5 的 `q_proj` 无需合并（已内置 gate），`in_proj_qkvz` 是整体 packed 权重。

#### 权重前缀（VLM 文本分支）

HF 权重格式为 VLM：`model.language_model.*`。加载时需：
1. 跳过 `model.visual.*`、`mtp.*`
2. 剥离前缀 `model.language_model.`

#### 已知问题（来自 Issues）

| Issue | 描述 | 影响 |
|-------|------|------|
| #39231 | `Qwen3_5ForCausalLM` 文本模型配置类型冲突（`Qwen3_5Config` vs `Qwen3_5TextConfig`） | 需在 Config 初始化中特判 |
| #35924 | GDN `in_proj_ba` Marlin kernel 在 TP≥2 时崩溃（已修复） | nano-vllm 单 GPU，不受影响 |
| #36236 | transformers 5.x 将 `Qwen3_5MoeConfig` 改名为 `Qwen3_5MoeTextConfig` | 关注 HF transformers 版本依赖 |

---

### 2. flash-linear-attention（FLA，算子参照）

FLA 提供 `chunk_gated_delta_rule`（prefill 并行）和 `recurrent_gated_delta_rule`（decode 递推）的 Triton CUDA kernel。

接口特征：

```python
# Prefill（分块并行）
out, last_state = chunk_gated_delta_rule(
    q,      # [B, T, nk, dk]
    k,      # [B, T, nk, dk]
    v,      # [B, T, nv, dv]
    g,      # [B, T, nk]     衰减门控
    beta,   # [B, T, nk]     delta 更新率
    initial_state=None,     # [B, nv, dk, dv] 或 None
    output_final_state=True,
)

# Decode（单步递推）
out, new_state = recurrent_gated_delta_rule(
    q, k, v, g, beta,
    initial_state,          # [B, nv, dk, dv]
    output_final_state=True,
)
```

nano-vllm 可选安装 FLA（`pip install flash-linear-attention`）；未安装时提供纯 PyTorch fallback。

---

### 3. 当前 nano-vllm 代码库分析

#### 现有能力

| 模块 | 现状 | Qwen3.5 需要的变化 |
|------|------|-------------------|
| `models/qwen3.py` | Qwen3 Dense，完整 | 无改动，新增 qwen35.py |
| `config.py` | `AutoConfig.from_pretrained()`，无特殊处理 | 需识别 `qwen3_5`/`qwen3_5_text`，提取 text_config，归一化 dtype 字符串 |
| `utils/loader.py` | 支持 packed_modules_mapping，直接匹配权重名 | 需增加前缀剥离和视觉权重跳过逻辑 |
| `utils/context.py` | 含 is_prefill, cu_seqlens, slot_mapping 等 | 需增加 `lin_attn_seq_slots: list[int] \| None` |
| `engine/sequence.py` | 完整序列状态机 | 需增加 `lin_attn_slot: int = -1` |
| `engine/scheduler.py` | FCFS，KV block 管理 | 需增加线性注意力 slot 池（大小=max_num_seqs） |
| `engine/model_runner.py` | 单模型 Qwen3，kv_cache 含所有层 | 需：(1) 按 layer_types 只给 full_attention 层分配 KV；(2) 分配 GDN 状态张量；(3) 传递 lin_attn_seq_slots |
| `engine/llm_engine.py` | 直接构造 Scheduler | 需向 ModelRunner/Scheduler 传递 num_lin_attn_slots |
| `layers/rotary_embedding.py` | 有 `assert rotary_dim == head_size` | 需移除 assert，支持 partial rotary；`get_rope` lru_cache maxsize 改为 16 |

#### 关键差距

1. **GatedDeltaNet 无 PyTorch 实现**：需要从头实现 chunk_gated_delta_rule 和 recurrent_gated_delta_rule 的纯 PyTorch 版本作为 fallback。
2. **Qwen35RMSNorm 需新增**：`(1 + w) * norm(x)`，零初始化，与现有 `RMSNorm` 不同。
3. **KV cache 分层分配**：现有 model_runner 为所有 `num_hidden_layers` 层分配 KV cache；Qwen3.5 只有 6 层需要，需按 layer_types 过滤。
4. **状态生命周期管理**：线性注意力状态不参与 block_manager 的 prefill-cache 机制，需独立 slot 池。

---

## 结论与选型建议

### 整体策略

**Fork Qwen3 架构，新建 `qwen35.py`**，而非修改 Qwen3。现有草稿方案（`docs/model_adaptation/qwen35_2b_adaptation.md`）经验证与 vllm 上游实现高度一致，可直接作为实现蓝本，补充如下细节：

### 关键选型决策

| 决策点 | 建议选择 | 理由 |
|--------|---------|------|
| GDN 算子 | 纯 PyTorch fallback 为主，FLA 可选 | 降低依赖，FLA 安装后自动加速 |
| 状态管理 | 显式 slot 池（草稿方案） | vllm 的 static_forward_context 依赖编译框架，nano-vllm 无此基础设施 |
| 权重加载前缀 | loader.py 增加 `prefix_to_strip` 参数 | 通用化，不只针对 Qwen3.5 |
| Config 解析 | 在 `Config.__post_init__` 中特判 model_type | 最小改动，不引入新的 Config 子类 |
| Partial RoPE | 修改 rotary_embedding.py，移除 assert | 复用现有 RotaryEmbedding，只扩展 partial 支持 |
| dtype 归一化 | 在 Config 中将 `"bfloat16"` 字符串转为 `torch.bfloat16` | qwen3_5 的 hf_config.dtype 为字符串，不同于 Qwen3 |

### 实现优先级

1. **P0（核心路径）**：qwen35.py 模型 + rotary partial + config 解析 + loader 前缀剥离 + kv cache 分层
2. **P1（状态管理）**：sequence slot + scheduler slot 池 + context lin_slots + GDN 状态分配
3. **P2（算子）**：GatedDeltaNet 纯 PyTorch 实现（chunk + recurrent）
4. **P3（验证）**：greedy decoding 对齐测试

### 与草稿设计的主要差异

现有草稿 `docs/model_adaptation/qwen35_2b_adaptation.md` 整体准确，以下几处需在 design 阶段确认：

1. **GQA head 数量**：草稿写 `linear_num_v_heads=16`，而 gist 分析写 `v_heads=32`（note：recurrent_state 形状可能是 `[seqs, nv, dk, dv]` 中 nv 值需从实际 config 取）；
2. **conv_dim 计算**：`q_dim + k_dim + v_dim = 16×128 + 16×128 + 16×128 = 6144`（草稿值正确）；
3. **recurrent_state dtype**：草稿写 float32，vllm 也用 float32，正确；
4. **loader.py 的 prefix_to_strip**：草稿中写死为 `"model.language_model."`，建议作为类属性 `weight_prefix_to_strip`，更规范。
