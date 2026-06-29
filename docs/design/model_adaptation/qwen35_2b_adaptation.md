# Qwen3.5-2B 模型适配设计文档

## 1. 模型架构概述

Qwen3.5-2B（内部架构名 `Qwen3Next`）是一个**混合架构**视觉语言模型（VLM），其语言骨干由以下两类层交替组成：

| 参数 | 值 |
|------|----|
| 总层数 | 24 |
| 全注意力层（full_attention） | 6 |
| 线性注意力层（linear_attention） | 18 |
| Hidden size | 2048 |
| Attention heads | 32 |
| KV heads | 8 |
| Head dim | 256 |
| Partial RoPE factor | 0.25（仅旋转前 64 维） |
| 词表大小 | 151,936 |

### 线性注意力层参数

| 参数 | 值 |
|------|----|
| linear_num_key_heads | 16 |
| linear_num_value_heads | 16 |
| linear_key_head_dim | 128 |
| linear_value_head_dim | 128 |
| linear_conv_kernel_dim | 4 |
| conv_dim (=key_dim×2+value_dim) | 6144 |

### VLM 结构

顶层配置 `model_type=qwen3_5`，语言骨干嵌套在 `text_config` 下（`model_type=qwen3_5_text`）。  
权重文件中所有语言模型权重带有前缀 `model.language_model.`；视觉编码器权重（`model.visual.*`）和 MTP 头权重（`mtp.*`）需跳过。

---

## 2. 关键技术差异

### 2.1 Qwen35RMSNorm（零初始化乘性权重）

标准 RMSNorm：`w * norm(x)`（权重初始化为 ones）  
Qwen3Next 风格：`(1 + w) * norm(x)`（权重初始化为 **zeros**，等效于恒等变换）

```python
class Qwen35RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        self.weight = nn.Parameter(torch.zeros(dim))  # zeros init!

    def forward(self, x):
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return ((1.0 + self.weight.float()) * xf).to(x.dtype)
```

### 2.2 GatedDeltaNet（线性注意力）

#### 计算流程

```
hidden → in_proj_qkvz → [q_raw, k_raw, v_raw, z]   (z 不经过 conv)
hidden → in_proj_ba   → [b, a]

[q_raw, k_raw, v_raw] → causal conv1d + SiLU → [q_c, k_c, v_c]

beta = sigmoid(b)
g = -A_log.exp() * softplus(a + dt_bias)   # 每 head 的门控衰减

q_c, k_c → L2 normalize (per head)
if num_v_heads > num_k_heads: repeat_interleave expand

output, state = gated_delta_rule(q_c, k_c, v_c, g, beta, initial_state)

output = RMSNormGated(output, z)   # norm(output) * silu(z)
return out_proj(output)
```

#### 门控 Delta 规则（recurrent 形式）

```
for each token t:
    state = state * g_t.exp()                   # 状态衰减
    kv_mem = (state * k_t).sum(dim=key_dim)     # 检索
    delta = (v_t - kv_mem) * beta_t             # 差分更新
    state = state + k_t ⊗ delta                 # 外积写入
    out_t = (state * q_t).sum(dim=key_dim)      # 查询
```

#### 两种执行路径

| 场景 | 方法 | 说明 |
|------|------|------|
| Prefill（T > 1） | `chunk_gated_delta_rule` | 分块并行，无 initial_state；结束后保存最终 state |
| Decode（T = 1） | `recurrent_gated_delta_rule` | 单步递推，读取 initial_state；结束后更新 state |

### 2.3 Qwen35Attention（全注意力 + 输出门）

- `q_proj` 输出 `2 × num_heads × head_dim`：前半为 query，后半为 output gate
- `q_norm`、`k_norm` 使用 `Qwen35RMSNorm`（zeros-init）
- Partial RoPE：仅旋转每个 head 的前 64 维（`partial_rotary_factor=0.25`），剩余 192 维 pass-through
- 输出门：`attn_out = attn_out * sigmoid(gate)`

```python
qg = self.q_proj(hidden).view(T, num_heads * 2, head_dim)
q, gate = qg[:, :num_heads, :], qg[:, num_heads:, :]
gate = gate.reshape(T, num_heads * head_dim)
# ...
o = attn(q, k, v)
o = o.flatten(1) * sigmoid(gate)
```

---

## 3. nano-vllm 适配方案

### 3.1 文件改动总览

| 文件 | 改动类型 | 说明 |
|------|----------|------|
| `nanovllm/models/qwen35.py` | **新增** | 完整模型实现 |
| `nanovllm/config.py` | 修改 | 提取 VLM 的 text_config，处理 dtype 字符串 |
| `nanovllm/utils/loader.py` | 修改 | 跳过视觉权重，剥离 LM 前缀 |
| `nanovllm/utils/context.py` | 修改 | 新增 `lin_attn_seq_slots` 字段 |
| `nanovllm/engine/sequence.py` | 修改 | 新增 `lin_attn_slot` 字段 |
| `nanovllm/engine/scheduler.py` | 修改 | 线性注意力 slot 池管理 |
| `nanovllm/engine/model_runner.py` | 修改 | 检测模型类型；分层 KV cache；线性状态分配 |
| `nanovllm/engine/llm_engine.py` | 修改 | 混合模型传入 `num_lin_attn_slots` |
| `nanovllm/layers/rotary_embedding.py` | 修改 | 支持 Partial RoPE；cache 容量扩大 |

### 3.2 权重加载

**前缀剥离**：`Qwen35ForCausalLM.weight_prefix_to_strip = "model.language_model."`

`loader.py` 在加载每个 tensor 前：
1. 跳过 `model.visual.*` 和 `mtp.*`
2. 剥离前缀后得到参数路径（如 `model.layers.0.linear_attn.in_proj_qkvz.weight`）
3. 查找模型参数并调用 `weight_loader`

**Packed 映射**（MLP 权重合并）：
```python
packed_modules_mapping = {
    "gate_proj": ("gate_up_proj", 0),
    "up_proj":   ("gate_up_proj", 1),
}
```
注：Qwen3.5 的 q_proj 已内置 gate（输出 2×），无需 q/k/v 合并映射。

### 3.3 KV Cache 分配

```
num_kv_layers = len([t for t in layer_types if t == "full_attention"])  # = 6
kv_cache shape: [2, 6, num_blocks, block_size, num_kv_heads, head_dim]
```

Warmup 后精确估算空闲显存，只为 6 层全注意力层分配 KV 块。

### 3.4 线性注意力状态管理

#### Slot 池（Scheduler 层）

```python
# Scheduler.__init__
self.free_lin_attn_slots = set(range(max_num_seqs))  # 大小 = max_num_seqs

# 每次 prefill 调度前分配 slot
seq.lin_attn_slot = self.free_lin_attn_slots.pop()

# 序列完成后归还 slot
self.free_lin_attn_slots.add(seq.lin_attn_slot)
seq.lin_attn_slot = -1
```

#### 状态张量（ModelRunner 层）

每个 `GatedDeltaNet` 实例持有：
```python
conv_state:      [max_seqs, conv_dim=6144, kernel=4]   dtype=model_dtype
recurrent_state: [max_seqs, nv=16, dk=128, dv=128]     dtype=float32
```

`allocate_lin_attn_states()` 在 `allocate_kv_cache()` 后调用，绑定到每个 GatedDeltaNet 实例。

#### 上下文传递

```python
# context.lin_attn_seq_slots: list[int] | None
# prepare_prefill / prepare_decode 中：
lin_slots = [seq.lin_attn_slot for seq in seqs]
set_context(..., lin_attn_seq_slots=lin_slots)
```

GatedDeltaNet.forward 读取 context 得到本批次各序列的 slot 索引。

### 3.5 GatedDeltaNet 执行路径（详细）

#### Prefill（is_prefill=True，对每个序列循环）

```python
for i, slot in enumerate(slots):
    h = hidden[cu_q[i]:cu_q[i+1]]     # [T, hidden]
    # 1. 投影
    q, k, v, z, beta, g = split_projections(h)
    # 2. Causal conv1d（整段）
    qkv = cat(q, k, v).T.unsqueeze(0)           # [1, conv_dim, T]
    conv_out = silu(conv1d(qkv)[:, :, :T])       # [1, conv_dim, T]
    conv_state[slot] = qkv[0, :, -kernel:]       # 保存尾部 kernel 帧
    # 3. 分割 q_c, k_c, v_c from conv_out
    # 4. GQA expand
    # 5. chunk_gated_delta_rule → core_out [1, T, nv, dv], last_state [1, nv, dk, dv]
    recurrent_state[slot] = last_state[0]
    # 6. gated norm + out_proj → [T, hidden]
```

#### Decode（is_prefill=False，对每个序列循环）

```python
for i, slot in enumerate(slots):
    h = hidden[i:i+1]                 # [1, hidden]
    # 1. 投影（单 token）
    q, k, v, z, beta, g = split_projections(h)
    # 2. Conv 单步更新
    token = cat(q, k, v).T.unsqueeze(0)         # [1, conv_dim, 1]
    combined = cat(conv_state[slot], token, dim=-1)  # [1, conv_dim, kernel+1]
    conv_state[slot] = combined[0, :, -kernel:]
    conv_out = silu(F.conv1d(combined, weight, padding=0, groups=conv_dim)[:, :, -1:])
    # 3. recurrent_gated_delta_rule with initial_state=recurrent_state[slot]
    # 4. 更新 recurrent_state[slot]
    # 5. gated norm + out_proj → [1, hidden]
```

---

## 4. Partial RoPE 支持

`rotary_embedding.py` 修改：

1. 移除 `assert rotary_dim == head_size`
2. 新增 `self.rotary_dim` 存储实际旋转维度
3. Forward 中按 `rotary_dim` 切分：

```python
if self.rotary_dim < self.head_size:
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q = cat([apply_rotary_emb(q_rot, cos, sin), q_pass], dim=-1)
    k = cat([apply_rotary_emb(k_rot, cos, sin), k_pass], dim=-1)
```

4. `@lru_cache(maxsize=16)` 支持多种 rotary 配置同时共存

---

## 5. 模型注册与调用链

```
LLMEngine.__init__
  → Config.__post_init__            # 检测 qwen3_5，提取 text_config
  → ModelRunner.__init__
      → _build_model(hf_config)    # model_type==qwen3_5_text → Qwen35ForCausalLM
      → load_model(model, path)    # 剥离前缀，跳过视觉权重
      → warmup_model()             # 峰值显存测量（lin_attn_slots=None）
      → allocate_kv_cache()        # 6 层 full attn KV cache
          → allocate_lin_attn_states()  # 18 层 GatedDeltaNet 状态
  → Scheduler(..., num_lin_attn_slots=max_num_seqs)

LLMEngine.step
  → Scheduler.schedule()           # 分配 seq.lin_attn_slot
  → ModelRunner.run(seqs, is_prefill)
      → prepare_prefill/decode()   # set_context(..., lin_attn_seq_slots=slots)
      → model.forward()            # GatedDeltaNet 读 context.lin_attn_seq_slots
  → Scheduler.postprocess()        # 序列完成时归还 slot
```

---

## 6. 权重名称对照

以 layer 0（linear_attention 层）为例：

| HF 权重名 | 剥离前缀后 | 参数路径 |
|-----------|-----------|---------|
| `model.language_model.model.layers.0.linear_attn.in_proj_qkvz.weight` | `model.layers.0.linear_attn.in_proj_qkvz.weight` | `model.layers[0].linear_attn.in_proj_qkvz.weight` |
| `model.language_model.model.layers.0.linear_attn.conv1d.weight` | `model.layers.0.linear_attn.conv1d.weight` | `model.layers[0].linear_attn.conv1d.weight` |
| `model.language_model.model.layers.0.linear_attn.A_log` | `model.layers.0.linear_attn.A_log` | `model.layers[0].linear_attn.A_log` |
| `model.language_model.model.layers.0.input_layernorm.weight` | `model.layers.0.input_layernorm.weight` | `model.layers[0].input_layernorm.weight` |
| `model.language_model.model.layers.2.self_attn.q_proj.weight` | `model.layers.2.self_attn.q_proj.weight` | `model.layers[2].self_attn.q_proj.weight` |
| `model.language_model.model.layers.2.mlp.gate_proj.weight` | `model.layers.2.mlp.gate_proj.weight` | → packed → `model.layers[2].mlp.gate_up_proj` shard 0 |
| `model.language_model.lm_head.weight` | `lm_head.weight` | `lm_head.weight` |

---

## 7. 显存估算

以 Qwen3.5-2B 为例（bf16，max_num_seqs=32，max_model_len=4096）：

| 组件 | 显存 |
|------|------|
| 模型权重（~2B 参数） | ~4 GB |
| KV Cache（6 层，按剩余显存） | 动态 |
| Conv 状态（18 层 × 32 seq × 6144 × 4 × bf16） | ~27 MB |
| Recurrent 状态（18 层 × 32 seq × 16 × 128 × 128 × fp32） | ~72 MB |

线性注意力状态相比 KV cache 极小，对总显存影响可忽略。

---

## 8. 已知限制

1. **GatedDeltaNet 性能**：当前使用纯 PyTorch 实现，prefill 对长序列效率较低。可安装 `flash-linear-attention`（fla）库启用 CUDA kernel 加速（HF 自动检测）。
2. **Chunked Prefill 不支持**：调度器目前不支持对同一序列的 prefill 分多步执行（chunked prefill），线性注意力层在跨 chunk 时无法正确传递 recurrent state。
3. **前缀缓存**：KV cache 前缀缓存（PagedAttention block reuse）仅对 full_attention 层有效；线性注意力层状态不参与前缀复用。
4. **TP 支持**：当前所有层均为单 GPU 版本（tp_size=1）。
