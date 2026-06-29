# Qwen3.5-35B-A3B MoE 适配 详细设计文档

## Motivation

在 8GB 显存下运行 Qwen3.5-35B-A3B：
1. 修复 GDN 的 nk/nv 维度 Bug（35B 暴露，2B 因 nk=nv 未暴露）
2. 实现 MoE FFN（256 专家 top-8 + 1 共享专家）
3. 通过减层 config.json（3 层 GDN）使模型权重约 6.74GB 以适配 8GB

---

## Architecture

```
用户层
  Config(model="/path/to/Qwen3.5-35B-A3B-3L")
    ├── config.json → model_type="qwen3_5_moe"
    │     └─ __post_init__ 识别 → 提取 text_config (model_type="qwen3_5_moe_text")
    └── hf_config.model_type = "qwen3_5_moe_text"

ModelRunner._build_model(hf_config)
  └── model_type == "qwen3_5_moe_text"
        → Qwen35MoEForCausalLM(hf_config)

Qwen35MoEForCausalLM
  ├── language_model: Qwen35MoEModel
  │     ├── embed_tokens: VocabEmbedding
  │     ├── layers: ModuleList[3 × Qwen35MoEDecoderLayer(type="linear_attention")]
  │     └── norm: Qwen35RMSNorm
  └── lm_head: LMHead

Qwen35MoEDecoderLayer (linear_attention)
  ├── linear_attn: GatedDeltaNet  ← 从 qwen35.py 复用（修正后）
  ├── mlp: Qwen35MoEFFN           ← 新增
  ├── input_layernorm: Qwen35RMSNorm
  └── post_attention_layernorm: Qwen35RMSNorm

Qwen35MoEFFN
  ├── gate: nn.Linear(hidden, num_experts)       # 路由器
  ├── experts: ExpertWeights                      # packed 参数存储
  │     ├── gate_up_proj: Parameter[256,1024,2048]
  │     └── down_proj:    Parameter[256,2048,512]
  ├── shared_expert: SharedExpertMLP             # 始终激活
  │     ├── gate_up_proj: MergedColumnParallelLinear
  │     └── down_proj:    RowParallelLinear
  └── shared_expert_gate: nn.Linear(hidden, 1)   # 标量门
```

### 数据流（per step）

```
hidden [T, 2048]
  │
  ├─→ gate → softmax → topk(8) → top_indices[T,8], top_scores[T,8]
  │                                      │
  │            [for each of 256 experts] │
  │            gather tokens by expert   │
  │            einsum expert matmul      │
  │            scatter_add weighted out  │
  │                                      ↓
  │                              routed_out [T, 2048]
  │
  ├─→ shared_expert_gate → sigmoid → gate_val [T, 1]
  │   shared_expert(hidden)          → shared_out [T, 2048]
  │   gate_val × shared_out          → gated_shared [T, 2048]
  │
  └─→ routed_out + gated_shared = final_out [T, 2048]
```

---

## Interfaces

### 1. `nanovllm/models/qwen35.py` 修改

**GatedDeltaNet.__init__** 维度修正：
```python
# 旧（nk）
self.in_proj_b = ColumnParallelLinear(hidden, nk, bias=False)
self.in_proj_a = ColumnParallelLinear(hidden, nk, bias=False)
self.A_log     = nn.Parameter(torch.empty(nk))
self.dt_bias   = nn.Parameter(torch.empty(nk))

# 新（nv）
self.in_proj_b = ColumnParallelLinear(hidden, nv, bias=False)
self.in_proj_a = ColumnParallelLinear(hidden, nv, bias=False)
self.A_log     = nn.Parameter(torch.empty(nv))
self.dt_bias   = nn.Parameter(torch.empty(nv))
```

**`_recurrent_step` 修正**（移除 g/beta 扩展，仅扩展 k 和 q）：
```python
def _recurrent_step(state, q, k, v, g, beta):
    nv, dk, dv = state.shape
    nk = k.shape[0]
    # g: [nv], beta: [nv] — 已经是 nv-sized，不再扩展
    if nv > nk:
        ratio = nv // nk
        k   = k.repeat_interleave(ratio, dim=0)    # [nk→nv, dk]
        q_f = q.float().repeat_interleave(ratio, dim=0)  # [nk→nv, dk]
    else:
        q_f = q.float()
    # 后续逻辑不变（使用 nv-sized g, beta）
```

**`_split_projections` 更新**（返回 nv-sized g/beta，不变接口，参数含义更清晰）：
```python
beta = torch.sigmoid(b)                              # [T, nv]
g    = -self.A_log.exp() * F.softplus(a + self.dt_bias)  # [T, nv]
```

---

### 2. `nanovllm/models/qwen35_moe.py`（新建）

```python
class ExpertWeights(nn.Module):
    """打包存放所有专家权重，参数路径匹配 safetensors key。"""
    gate_up_proj: nn.Parameter  # [E, 2*moe_inter, hidden]
    down_proj:    nn.Parameter  # [E, hidden, moe_inter]

class SharedExpertMLP(nn.Module):
    """始终激活的共享专家，SwiGLU FFN。"""
    gate_up_proj: MergedColumnParallelLinear  # [2*moe_inter, hidden]
    down_proj:    RowParallelLinear           # [hidden, moe_inter]

class Qwen35MoEFFN(nn.Module):
    def forward(self, x: Tensor) -> Tensor: ...
    # 路由 → top-8 dispatch → scatter_add → + gated shared

class Qwen35MoEDecoderLayer(nn.Module):
    # 与 Qwen35DecoderLayer 相同结构，mlp 字段替换为 Qwen35MoEFFN
    linear_attn:              GatedDeltaNet    # linear_attention 层
    self_attn:                Qwen35Attention  # full_attention 层
    mlp:                      Qwen35MoEFFN
    input_layernorm:          Qwen35RMSNorm
    post_attention_layernorm: Qwen35RMSNorm

class Qwen35MoEModel(nn.Module):
    embed_tokens: VocabEmbedding
    layers:       ModuleList[Qwen35MoEDecoderLayer]
    norm:         Qwen35RMSNorm

class Qwen35MoEForCausalLM(nn.Module):
    weight_prefix_to_strip = "model."
    weight_skip_prefixes   = ("model.visual.", "mtp.")
    packed_modules_mapping = {
        "gate_proj": ("gate_up_proj", 0),
        "up_proj":   ("gate_up_proj", 1),
    }
    language_model: Qwen35MoEModel
    lm_head:        LMHead
```

---

### 3. `nanovllm/config.py` 修改

在 `__post_init__` 的 JSON 路径中：
```python
# 旧
if cfg.get('model_type') == 'qwen3_5' and 'text_config' in cfg:
    cfg = cfg['text_config']

# 新
if cfg.get('model_type') in ('qwen3_5', 'qwen3_5_moe') and 'text_config' in cfg:
    cfg = cfg['text_config']
```

在 `AutoConfig` 路径中：
```python
# 旧
if getattr(hf, 'model_type', '') == 'qwen3_5':
    hf = hf.text_config

# 新
if getattr(hf, 'model_type', '') in ('qwen3_5', 'qwen3_5_moe'):
    hf = hf.text_config
```

---

### 4. `nanovllm/engine/model_runner.py` 修改

**`_build_model`** 新增分支：
```python
if model_type == 'qwen3_5_text':
    from nanovllm.models.qwen35 import Qwen35ForCausalLM
    return Qwen35ForCausalLM(hf_config)
if model_type == 'qwen3_5_moe_text':
    from nanovllm.models.qwen35_moe import Qwen35MoEForCausalLM
    return Qwen35MoEForCausalLM(hf_config)
```

**`allocate_kv_cache`** 修复除零：
```python
if num_kv_layers == 0:
    config.num_kvcache_blocks = 0
    self.kv_cache = torch.empty(0)
    return
```

---

### 5. 减层 config.json（3L）

创建 `/home/cb/model/Qwen3.5-35B-A3B-3L/`：
- `config.json`：text_config 中修改 `num_hidden_layers: 3`，`layer_types: [lin, lin, lin]`
- 软链接到原始 14 个 safetensors 文件

---

## State Machine（KV cache 与线性注意力状态共存）

3 层全为 GDN（无 full_attention）时：
- `num_kv_layers = 0` → 跳过 KV cache 分配
- `num_lin_attn_slots = max_num_seqs` → 每个序列持有 conv_state + recurrent_state
- 调度器照常管理 `lin_attn_slot`（free slots pool）
- decode 路径不进入任何 `AttentionWithKVCache` 分支

---

## Risks

| 风险 | 缓解 |
|------|------|
| nk/nv 修正破坏 2B 推理 | 先跑 `tests/test_qwen35.py` 确认 2B 推理不退化 |
| MoE 循环 dispatch 在 prefill 长序列时慢 | 可接受，验证正确性为主，不优化性能 |
| 显存实际超出估算 | 降低 `gpu_memory_utilization=0.85` 或 max_num_seqs |
| 3 层输出乱码 | 预期，架构验证目标不是质量 |

---

## Test Plan

1. **Unit**：`tests/test_qwen35_moe.py` — 测试 `Qwen35MoEFFN.forward` 的输出形状和 top-k 路由正确性
2. **Integration**：加载 3L 模型，运行单 token prefill + decode 不报错
3. **E2E**：`example.py` 以 3L 模型生成 20 tokens，不 OOM，输出非全空白
