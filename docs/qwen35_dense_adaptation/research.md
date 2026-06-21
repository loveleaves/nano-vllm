# Qwen3.5 dense 适配 · 调研

## 目标
在 phase7 工程中适配 **Qwen3.5 dense**（混合线性注意力）文本模型，对齐 vLLM 0.15.1
对 Qwen3-Next 系列的实现。dense = MLP 为标准 SwiGLU（无 MoE）。

## 模型架构（取自实际权重 `Qwen3.5-2B/config.json` + safetensors）

| 项 | 值 |
|---|---|
| 顶层架构 | `Qwen3_5ForConditionalGeneration`（VLM 包装，含 vision/MTP；nano 仅取文本主干） |
| 文本主干 | `model_type=qwen3_5_text`，超参在 `config.text_config` 下 |
| 层数 / 模式 | 24 层；`full_attention_interval=4` → `layer_types` 每 4 层一个 `full_attention`，其余 `linear_attention` |
| hidden / inter | 2048 / 6144 |
| 全注意力 | heads=8, kv_heads=2, head_dim=256；`attn_output_gate=true`；`partial_rotary_factor=0.25`（rotary_dim=64）；rope_theta=1e7 |
| 线性注意力 | GatedDeltaNet：nk=nv=16, dk=dv=128, conv_kernel=4 |
| 归一化 | layernorm/qk_norm/final = `(1+w)·rmsnorm`（GemmaRMSNorm，zeros-init）；GDN 内部 norm = 标准 `w·rmsnorm`（RMSNormGated，ones-init） |
| 词表 / tie | vocab=248320；`tie_word_embeddings=true` |

## 关键发现

1. **投影布局与 vLLM main 不同**。vLLM main 的 `qwen3_next.py`（面向 Qwen3-Next-80B）用
   **合并** `in_proj_qkvz` / `in_proj_ba`；而实际 Qwen3.5-2B 权重是**分离** 的
   `in_proj_qkv` / `in_proj_z` / `in_proj_b` / `in_proj_a`。适配须按实际权重命名（分离）。
   全注意力的 q/k/v 亦为分离的 `q_proj`/`k_proj`/`v_proj`（`q_proj` 输出 2×heads，后半为门）。

2. **conv1d 为深度因果卷积**，权重形状 `[conv_dim=6144, 1, kernel=4]`，可直接映射到
   `nn.Conv1d(conv_dim, conv_dim, kernel, groups=conv_dim, bias=False)`。

3. **线性注意力不进 KV cache**。它维护**每序列递归状态**：
   - `conv_state`：因果卷积的左侧上下文（最近 kernel-1 个原始 qkv 列）；
   - `recurrent_state`：门控 Δ-rule 线性注意力记忆 `[nv, dk, dv]`（float32）。
   这与全注意力的分页 KV cache 是两套并行的状态机制。

4. **transformers 4.57.6 不认识 `model_type=qwen3_5`**，`AutoConfig.from_pretrained` 抛错；
   须回退为直接解析 `config.json`。

5. 权重含 nano 未实现的子结构（**MTP 头** `mtp_num_hidden_layers=1`、**视觉塔** `model.visual.*`）；
   加载须容忍「目标参数不存在 → 跳过」，包括带 `gate_proj`/`up_proj` 的 packed 分支。

## 数学（门控 Δ-rule，逐 token 递推）
对每个 value head（k 已 L2 归一化，q 归一化后乘 1/√dk）：
```
state  ← state · exp(g)                       # 衰减，g = -exp(A_log)·softplus(a+dt_bias) < 0
kv_mem ← Σ_k state[k] · key[k]                 # 检索
state  ← state + key ⊗ (value - kv_mem)·β     # 差分更新，β = sigmoid(b)
out    ← Σ_k query[k] · state[k]              # 查询
```
conv（silu 激活的因果深度卷积）在投影后、递推前作用于 q/k/v。

## 参考
- vLLM `vllm/model_executor/models/qwen3_next.py`（架构与数学；注意投影布局差异）
- vLLM `Qwen3NextRMSNorm = GemmaRMSNorm`、`RMSNormGated`（归一化语义）
- 工程内既有 [[project_qwen35_status]]（phase3 验证过的同模型实现，本次移植其数学到 phase7）
