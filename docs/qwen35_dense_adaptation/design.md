# Qwen3.5 dense 适配 · 设计

## 总览
新增模型 `nanovllm/models/qwen35.py`，并在引擎层增加**线性注意力递归状态池**这一与分页
KV cache 并行的状态机制。其余复用 phase7 既有执行器（CPU/GPU、采样、连续批、权重加载）。

## 模型层（`models/qwen35.py`）

```
Qwen35ForCausalLM                      # 注册名 Qwen3_5ForConditionalGeneration
└─ model (_LanguageModelShell)         # 命名外壳，对齐权重键 model.language_model.*
   └─ language_model (Qwen35Model)
      ├─ embed_tokens (VocabParallelEmbedding)
      ├─ layers[i] (Qwen35DecoderLayer)，按 config.layer_types 分发：
      │    linear_attention → linear_attn (GatedDeltaNet)
      │    full_attention   → self_attn   (Qwen35Attention)
      │    + mlp (Qwen35MLP, SwiGLU dense) + 两个 Qwen35RMSNorm
      └─ norm (Qwen35RMSNorm)
└─ lm_head (ParallelLMHead, 与 embed_tokens tied)
```

要点：
- **Qwen35RMSNorm**：`(1+w)·rmsnorm`，zeros-init（layernorm/qk_norm/final）；GDN 内部 norm
  复用 `layers.layernorm.RMSNorm`（ones-init 标准）。
- **Qwen35Attention**：`q_proj` 输出 2×heads（按 head 交错 chunk 出 query/gate）；部分 RoPE；
  `attn(q,k,v) · sigmoid(gate)` 后过 `o_proj`。
- **GatedDeltaNet**：分离投影 + 深度因果 conv + 逐 token 门控 Δ-rule 递推；输出门 `silu(z)`。
  类属性 `needs_state_pool=True` 供引擎鸭子类型探测（引擎不硬依赖具体模型类）。
- VLM config：`__init__` 取 `config.text_config`；权重的 `model.visual.*` / MTP 在加载时自动跳过。

### 统一的 prefill / chunked-prefill / decode 线性注意力
关键简化：以 `conv_state` 为左 context 的**分段卷积** + 递推，对任意 query 长度 q≥1 统一处理
（q==1 即 decode 退化）。
- conv：`inp = [conv_state(kernel-1 列) ‖ 本段 qkv]` → 无 padding 卷积得 q 个输出 →
  `conv_state ← inp 末 kernel-1 列`。
- recurrent：从 `recurrent_state[slot]` 出发逐 token 递推，写回末态。

这天然支持 chunked prefill（跨 step 续算）与连续批（每序列独立 slot）。

## 引擎层

### 1. 线性注意力递归状态池（`engine/model_runner.py`）
- **状态归属**：状态张量是模型参数邻接物，由 ModelRunner 分配并绑定到各 GatedDeltaNet
  （类比 KV cache 切片绑定到各 Attention），每模块 `allocate_state(num_slots)`：
  - `conv_state[num_slots, conv_dim, kernel-1]`（运行 dtype）
  - `recurrent_state[num_slots, nv, dk, dv]`（float32）
- **槽位分配器**：`seq_id → slot` 映射 + 空闲槽栈，`num_slots = max_num_seqs`。
  每步 `_assign_state_slots(ordered, finished)`：回收已结束/被抢占序列的槽 → 为本批各序列
  取/分配槽（新序列状态清零，因槽可复用），返回按批行序对齐的 `list[int]`。
- **下发**：写入 `AttentionMetadata.state_slots`（新增字段）；GatedDeltaNet 据 `query_start_loc`
  分段、据 `state_slots` 索引状态。纯全注意力模型该字段恒为 None，零开销。
- **强制 eager**：含线性注意力 → `enforce_eager=True`（递归状态的数据依赖索引 + Python 扫描
  无法稳定进 CUDA graph）。
- **生命周期**：warmup 前 `_init_linear_attn_state` 分配；warmup 后 `_reset_linear_attn_state`
  清空槽与状态。

### 2. text_config 抽象（`engine/model_runner.py` + `config.py`）
VLM 包装把解码器超参放在 `text_config` 下。`self.text_config = getattr(hf_config,
"text_config", hf_config)`，引擎层（dtype / KV cache / 采样 vocab / graph）一律读它；
普通文本模型退回 hf_config 自身，零影响。

### 3. KV cache 仅按全注意力层数分配（`allocate_kv_cache`）
`num_layers = 全模型 Attention 模块计数`（混合模型只有 full_attention 层持有 KV；纯 Qwen3
等于 num_hidden_layers）。避免按 24 层超额分配/超额均摊内存。

### 4. 部分 RoPE（`layers/rotary_embedding.py`）
放宽 `rotary_dim == head_size` 断言为 `rotary_dim ≤ head_size`：仅前 rotary_dim 维做 RoPE，
尾部直通。对 Qwen3（rotary_dim==head_dim）行为不变。

### 5. config.json 回退加载（`config.py`）
`AutoConfig` 失败时解析 `config.json` → SimpleNamespace（保留顶层 `architectures` 与嵌套
`text_config`，`dtype` 字符串转 `torch.dtype`，内层 `rope_parameters`/`layer_types` 原样保留）。

### 6. 权重加载容错（`utils/loader.py`）
packed 分支（gate/up 合并）也容忍「目标参数不存在 → 跳过」，以略过 MTP 头 / 视觉塔权重。

## 取舍与限制
- **线性注意力仅 TP=1**（GatedDeltaNet 断言）：递归状态/卷积分组的 TP 切分未实现；Qwen3.5-2B
  规模下 TP=1 足够。
- **状态池内存 ∝ max_num_seqs**：`recurrent_state` 每槽每线性层 ~1MB（2B 配置）。示例用小
  `max_num_seqs`。
- **不支持 CUDA graph / 投机 / async / swap 抢占叠加线性注意力**：这些路径不下发 state_slots，
  与递归状态语义冲突；标准同步 run() 路径（含 chunked prefill、连续批）完整支持。
- **仅文本主干**：视觉塔 / MTP 头不实现（权重自动跳过）。

## 注册
`models/registry.py`：`Qwen3_5ForConditionalGeneration → nanovllm.models.qwen35:Qwen35ForCausalLM`。
