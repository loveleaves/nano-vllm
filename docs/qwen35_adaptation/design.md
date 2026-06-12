# Qwen3.5-2B 适配 nano-vllm 详细设计文档

## Motivation

Qwen3.5-2B（Qwen3-Next）是混合架构模型：24 层中 18 层为 GatedDeltaNet 线性注意力，6 层为带输出门的全注意力（Full Attention）。  
nano-vllm 当前只支持纯 Dense 架构（Qwen3），适配该模型需要：新增混合模型实现、扩展 KV cache 管理（仅对全注意力层分配 KV cache）、引入线性注意力状态（conv_state + recurrent_state）的生命周期管理。

---

## Architecture（模块图 / 数据流）

```
┌─────────────────────────────────────────────────────────────┐
│ LLMEngine                                                   │
│  ├── Config.__post_init__                                   │
│  │     检测 qwen3_5 → 提取 text_config，归一化 dtype        │
│  ├── ModelRunner                                            │
│  │     ├── _build_model()    # model_type → Qwen35ForCausalLM │
│  │     ├── load_model()      # 剥离前缀，跳过 visual/mtp 权重 │
│  │     ├── warmup_model()    # 峰值显存测量                  │
│  │     ├── allocate_kv_cache()  # 仅 6 层全注意力 KV         │
│  │     └── allocate_lin_attn_states() # 18×GDN 状态张量     │
│  └── Scheduler(num_lin_attn_slots=max_num_seqs)             │
└─────────────────────────────────────────────────────────────┘

step() 数据流：
  scheduler.schedule()
    └─ 新 prefill seq → seq.lin_attn_slot = free_slots.pop()

  model_runner.run(seqs, is_prefill)
    ├─ prepare_prefill/decode()
    │    └─ set_context(..., lin_attn_seq_slots=[seq.lin_attn_slot for seq in seqs])
    └─ model.forward(input_ids, positions)
         └─ Qwen35Model.layers[i]
              ├─ 全注意力层 → Qwen35Attention  → AttentionWithKVCache
              └─ 线性注意力层 → GatedDeltaNet  → 读 context.lin_attn_seq_slots
                                                  访问 self.conv_state[slot]
                                                  访问 self.recurrent_state[slot]

  scheduler.postprocess()
    └─ finished seq → free_slots.add(seq.lin_attn_slot); slot = -1
```

### 显存布局（Qwen3.5-2B，bf16，max_num_seqs=32）

| 组件 | 形状 | 约占用 |
|------|------|--------|
| 模型权重 | ~2B 参数 | ~4 GB |
| KV cache（6 层全注意力） | [2, 6, N_blocks, 256, 8, 256] | 动态（按剩余显存） |
| conv_state（18 层 GDN） | 18 × [32, 6144, 4] × bf16 | ~27 MB |
| recurrent_state（18 层 GDN） | 18 × [32, 16, 128, 128] × f32 | ~72 MB |

---

## Interfaces（关键接口定义）

### 1. `nanovllm/layers/rotary_embedding.py`（修改）

移除 `assert rotary_dim == head_size`，支持 Partial RoPE；`get_rope` maxsize 改为 16。

```python
class RotaryEmbedding(nn.Module):
    def __init__(self, head_size: int, rotary_dim: int,
                 max_position_embeddings: int, base: float):
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        # 不再 assert rotary_dim == head_size
        # cos_sin_cache: [max_position, 1, rotary_dim]（只存旋转维度的 cos/sin）

    def forward(self, positions: torch.Tensor,
                query: torch.Tensor, key: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        cos_sin = self.cos_sin_cache[positions]          # [N, 1, rotary_dim]
        cos, sin = cos_sin.chunk(2, dim=-1)              # 各 [N, 1, rotary_dim/2]
        if self.rotary_dim < self.head_size:
            # Partial RoPE：只旋转前 rotary_dim 维，后续 pass-through
            q_rot, q_pass = query[..., :self.rotary_dim], query[..., self.rotary_dim:]
            k_rot, k_pass = key[..., :self.rotary_dim], key[..., self.rotary_dim:]
            query = torch.cat([apply_rotary_emb(q_rot, cos, sin), q_pass], dim=-1)
            key   = torch.cat([apply_rotary_emb(k_rot, cos, sin), k_pass], dim=-1)
        else:
            query = apply_rotary_emb(query, cos, sin)
            key   = apply_rotary_emb(key, cos, sin)
        return query, key

@lru_cache(maxsize=16)   # 从 1 改为 16，支持多种旋转配置并存
def get_rope(head_size: int, rotary_dim: int, max_position: int, base: float) -> RotaryEmbedding:
    ...
```

---

### 2. `nanovllm/models/qwen35.py`（新增）

#### 2a. Qwen35RMSNorm

```python
class Qwen35RMSNorm(nn.Module):
    """零初始化乘性权重：(1 + w) * rms_norm(x)"""
    def __init__(self, dim: int, eps: float = 1e-6):
        self.weight = nn.Parameter(torch.zeros(dim))   # zeros!
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return ((1.0 + self.weight.float()) * xf).to(x.dtype)
```

#### 2b. RMSNormGated（GDN 输出归一化）

```python
class RMSNormGated(nn.Module):
    """norm(x) * silu(z)，无可学习权重"""
    def __init__(self, dim: int, eps: float = 1e-6): ...

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # x: [T, nv*dv], z: [T, hidden]
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (xf.to(x.dtype) * torch.nn.functional.silu(z))
```

#### 2c. GatedDeltaNet（线性注意力）

```python
class GatedDeltaNet(nn.Module):
    """
    GatedDeltaNet 线性注意力层。
    状态张量由 ModelRunner.allocate_lin_attn_states() 在 warmup 后注入。
    """
    def __init__(self, config):
        hidden  = config.hidden_size
        nk = config.linear_num_key_heads      # 16
        nv = config.linear_num_value_heads    # 16
        dk = config.linear_key_head_dim       # 128
        dv = config.linear_value_head_dim     # 128
        kernel = config.linear_conv_kernel_dim # 4
        conv_dim = (nk * dk) * 2 + (nv * dv)  # 6144: q+k+v

        self.nk, self.nv, self.dk, self.dv = nk, nv, dk, dv
        self.kernel = kernel
        self.conv_dim = conv_dim

        # 投影层
        self.in_proj_qkvz = ColumnParallelLinear(hidden, nk*dk + nk*dk + nv*dv + hidden, bias=False)
        self.in_proj_ba   = ColumnParallelLinear(hidden, nk + nk, bias=True)
        self.conv1d       = nn.Conv1d(conv_dim, conv_dim, kernel, groups=conv_dim, bias=True)
        self.out_norm     = RMSNormGated(nv * dv, eps=config.rms_norm_eps)
        self.out_proj     = RowParallelLinear(nv * dv, hidden, bias=False)

        # 可学习参数
        self.A_log   = nn.Parameter(torch.empty(nk))
        self.dt_bias = nn.Parameter(torch.empty(nk))

        # 状态张量（由 ModelRunner 注入，初始为空）
        self.conv_state: torch.Tensor = torch.empty(0)       # [max_seqs, conv_dim, kernel]
        self.recurrent_state: torch.Tensor = torch.empty(0)  # [max_seqs, nv, dk, dv]

    def allocate_states(self, max_seqs: int):
        """由 ModelRunner 调用，分配并持久化状态张量（在 CUDA 上）。"""
        device = self.A_log.device
        dtype  = self.A_log.dtype
        self.conv_state = torch.zeros(
            max_seqs, self.conv_dim, self.kernel, device=device, dtype=dtype
        )
        self.recurrent_state = torch.zeros(
            max_seqs, self.nv, self.dk, self.dv, device=device, dtype=torch.float32
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        context = get_context()
        slots = context.lin_attn_seq_slots
        if context.is_prefill:
            return self._prefill(hidden, slots)
        else:
            return self._decode(hidden, slots)

    def _prefill(self, hidden: torch.Tensor, slots: list[int]) -> torch.Tensor: ...
    def _decode(self, hidden: torch.Tensor, slots: list[int]) -> torch.Tensor: ...
```

**GatedDeltaNet 状态更新（recurrent 核心，PyTorch fallback）**

```python
def _recurrent_step(state, q, k, v, g, beta):
    """
    单步递推（decode）。
    state:  [nv, dk, dv] float32
    q,k:    [nk, dk]  L2-normalized
    v:      [nv, dv]
    g:      [nk]      gate（负值，exp(g) < 1）
    beta:   [nk]      update rate（已 sigmoid）
    返回: out [nv, dv], new_state [nv, dk, dv]
    """
    # GQA expand：若 nv > nk，则 k/g/beta 按比例扩展
    if v.shape[0] > k.shape[0]:
        ratio = v.shape[0] // k.shape[0]
        k = k.repeat_interleave(ratio, dim=0)
        g = g.repeat_interleave(ratio)
        beta = beta.repeat_interleave(ratio)

    state = state * g.exp()[:, None, None]             # 衰减
    kv_mem = torch.einsum('vk,vkd->vd', k.float(), state)   # 检索 [nv, dv]
    delta  = (v.float() - kv_mem) * beta[:, None]       # 差分 [nv, dv]
    state  = state + torch.einsum('vk,vd->vkd', k.float() * beta[:, None], delta)  # 写入
    out    = torch.einsum('vk,vkd->vd', q.float(), state)   # 输出 [nv, dv]
    return out, state
```

**GatedDeltaNet prefill（逐 seq 循环 + 逐 token 递推）**

```python
def _prefill(self, hidden, slots):
    context = get_context()
    cu_q = context.cu_seqlens_q       # [num_seqs+1]
    out_parts = []
    for i, slot in enumerate(slots):
        h = hidden[cu_q[i]:cu_q[i+1]]   # [T, hidden]
        T = h.shape[0]
        # 投影
        qkvz_raw = self.in_proj_qkvz(h)  # [T, nk*dk + nk*dk + nv*dv + hidden]
        q_raw, k_raw, v_raw, z = qkvz_raw.split([...], dim=-1)
        ba = self.in_proj_ba(h)          # [T, nk+nk]
        b_raw, a_raw = ba.split([self.nk, self.nk], dim=-1)

        # causal conv1d（prefill 整段）
        qkv_cat = torch.cat([q_raw, k_raw, v_raw], dim=-1).T.unsqueeze(0)  # [1, conv_dim, T]
        padded  = F.pad(qkv_cat, (self.kernel - 1, 0))
        conv_out = F.silu(self.conv1d(padded)[:, :, :T])     # [1, conv_dim, T]
        self.conv_state[slot] = qkv_cat[0, :, -self.kernel:] # 保存尾部

        q_c, k_c, v_c = conv_out[0].T.split([...], dim=-1)   # 各 [T, n*d]
        q_c = q_c.view(T, self.nk, self.dk)
        k_c = k_c.view(T, self.nk, self.dk)
        v_c = v_c.view(T, self.nv, self.dv)

        # L2 normalize
        q_c = F.normalize(q_c, dim=-1)
        k_c = F.normalize(k_c, dim=-1)

        # gate 计算
        g    = -self.A_log.exp() * F.softplus(a_raw + self.dt_bias)  # [T, nk]
        beta = torch.sigmoid(b_raw)                                    # [T, nk]

        # 逐 token 递推（保证正确性；可选升级为 chunk 并行）
        state = self.recurrent_state[slot].clone()
        seq_out = []
        for t in range(T):
            o_t, state = _recurrent_step(
                state, q_c[t], k_c[t], v_c[t], g[t], beta[t]
            )
            seq_out.append(o_t)
        self.recurrent_state[slot] = state
        core_out = torch.stack(seq_out, dim=0).view(T, -1).to(h.dtype)  # [T, nv*dv]

        # 输出归一化 + 投影
        normed = self.out_norm(core_out, z)
        out_parts.append(self.out_proj(normed))

    return torch.cat(out_parts, dim=0)
```

#### 2d. Qwen35Attention（全注意力 + 输出门）

```python
class Qwen35Attention(nn.Module):
    def __init__(self, config):
        hidden   = config.hidden_size
        num_h    = config.num_attention_heads     # 32
        num_kv_h = config.num_key_value_heads     # 8
        head_dim = getattr(config, 'head_dim',
                           hidden // num_h)        # 256
        partial_rotary_factor = getattr(config, 'partial_rotary_factor', 1.0)  # 0.25
        rotary_dim = int(head_dim * partial_rotary_factor)                       # 64

        # q_proj: 输出 2×num_h×head_dim（前半 query，后半 gate）
        self.q_proj  = ColumnParallelLinear(hidden, 2 * num_h * head_dim, bias=False)
        self.kv_proj = QKVParallelLinear(hidden, head_dim, 0, num_kv_h, bias=False)
        # 注：kv_proj 只包含 k 和 v，packed 为 kv
        self.o_proj  = RowParallelLinear(num_h * head_dim, hidden, bias=False)

        self.q_norm = Qwen35RMSNorm(head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen35RMSNorm(head_dim, eps=config.rms_norm_eps)

        self.rotary_emb = get_rope(
            head_dim, rotary_dim=rotary_dim,
            max_position=config.max_position_embeddings,
            base=getattr(config, 'rope_theta', 1000000),
        )
        self.attn = Attention(num_h, head_dim, head_dim ** -0.5, num_kv_h)

        self.num_heads   = num_h
        self.num_kv_heads = num_kv_h
        self.head_dim    = head_dim

    def forward(self, positions: torch.Tensor,
                hidden_states: torch.Tensor) -> torch.Tensor:
        T = hidden_states.shape[0]
        # q_proj → [T, 2*num_h, head_dim]，后半为 gate
        qg = self.q_proj(hidden_states).view(T, self.num_heads * 2, self.head_dim)
        q, gate = qg[:, :self.num_heads], qg[:, self.num_heads:]
        gate = gate.reshape(T, self.num_heads * self.head_dim)

        kv = self.kv_proj(hidden_states)  # [T, 2*num_kv_h*head_dim]
        k, v = kv.split([self.num_kv_heads * self.head_dim] * 2, dim=-1)
        k = k.view(T, self.num_kv_heads, self.head_dim)
        v = v.view(T, self.num_kv_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = self.rotary_emb(positions, q, k)

        o = self.attn(q, k, v)           # [T, num_h, head_dim]
        o = o.flatten(1) * torch.sigmoid(gate)
        return self.o_proj(o)
```

#### 2e. 层结构与模型

```python
class Qwen35DecoderLayer(nn.Module):
    def __init__(self, config, layer_type: str):
        if layer_type == "linear_attention":
            self.mixer = GatedDeltaNet(config)
        else:  # "full_attention"
            self.mixer = Qwen35Attention(config)
        self.mlp = Qwen3MLP(...)           # 复用 Qwen3MLP
        self.input_layernorm        = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layer_type = layer_type

    def forward(self, positions, hidden_states, residual):
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        if self.layer_type == "linear_attention":
            hidden_states = self.mixer(hidden_states)
        else:
            hidden_states = self.mixer(positions, hidden_states)

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen35ForCausalLM(nn.Module):
    weight_prefix_to_strip = "model.language_model."
    weight_skip_prefixes   = ("model.visual.", "mtp.")

    packed_modules_mapping = {
        "gate_proj": ("gate_up_proj", 0),
        "up_proj":   ("gate_up_proj", 1),
        # 注意：q_proj 已内置 gate，无需 q/k/v 合并；
        # kv_proj 对应 HF 的 k_proj + v_proj（packed 为 kv）
        "k_proj": ("kv_proj", "k"),
        "v_proj": ("kv_proj", "v"),
    }
```

---

### 3. `nanovllm/config.py`（修改）

```python
def __post_init__(self):
    # ... 现有校验 ...
    try:
        from transformers import AutoConfig
        hf = AutoConfig.from_pretrained(self.model)

        # Qwen3.5 VLM 包装：提取 text_config
        if getattr(hf, 'model_type', '') == 'qwen3_5':
            hf = hf.text_config

        # dtype 字符串归一化（Qwen3.5 hf_config.torch_dtype 可能是字符串）
        raw_dtype = getattr(hf, 'torch_dtype', None)
        if isinstance(raw_dtype, str):
            hf.torch_dtype = getattr(torch, raw_dtype, torch.bfloat16)

        # 为后续代码兼容，把 hf.dtype 统一挂成 torch.dtype
        if not hasattr(hf, 'dtype'):
            hf.dtype = getattr(hf, 'torch_dtype', torch.bfloat16)

        self.hf_config = hf
        self.max_model_len = min(self.max_model_len, hf.max_position_embeddings)
    except Exception:
        pass
```

---

### 4. `nanovllm/utils/loader.py`（修改）

```python
def load_model(model: nn.Module, path: str):
    prefix_to_strip = getattr(model, 'weight_prefix_to_strip', '')
    skip_prefixes   = getattr(model, 'weight_skip_prefixes', ())
    packed = getattr(model, 'packed_modules_mapping', {})

    for file in sorted(glob(os.path.join(path, "*.safetensors"))):
        with safe_open(file, framework="pt", device="cpu") as f:
            for weight_name in f.keys():
                # 跳过视觉/MTP 权重
                if any(weight_name.startswith(p) for p in skip_prefixes):
                    continue
                # 剥离前缀
                param_name = weight_name
                if prefix_to_strip and weight_name.startswith(prefix_to_strip):
                    param_name = weight_name[len(prefix_to_strip):]

                # packed 映射（逻辑不变，操作 param_name 而非 weight_name）
                for k, (v, shard_id) in packed.items():
                    if k in param_name:
                        mapped = param_name.replace(k, v)
                        param = model.get_parameter(mapped)
                        loader = getattr(param, 'weight_loader', default_weight_loader)
                        loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    try:
                        param = model.get_parameter(param_name)
                    except AttributeError:
                        continue
                    loader = getattr(param, 'weight_loader', default_weight_loader)
                    loader(param, f.get_tensor(weight_name))
```

---

### 5. `nanovllm/utils/context.py`（修改）

```python
@dataclass
class Context:
    # ... 现有字段不变 ...
    lin_attn_seq_slots: list[int] | None = None   # 新增

def set_context(..., lin_attn_seq_slots=None):
    global _CONTEXT
    _CONTEXT = Context(..., lin_attn_seq_slots=lin_attn_seq_slots)
```

---

### 6. `nanovllm/engine/sequence.py`（修改）

```python
class Sequence:
    def __init__(self, token_ids, sampling_params=None):
        # ... 现有字段不变 ...
        self.lin_attn_slot: int = -1   # 新增；-1 表示未分配
```

---

### 7. `nanovllm/engine/scheduler.py`（修改）

```python
class Scheduler:
    def __init__(self, ..., num_lin_attn_slots: int = 0):
        # ... 现有代码 ...
        self.free_lin_attn_slots: set[int] = set(range(num_lin_attn_slots))

    def schedule(self):
        # Prefill 调度：为每个新 seq 分配 slot
        while self.waiting ...:
            seq = self.waiting[0]
            # 若模型需要线性注意力 slot 且槽位不足，停止调度
            if self.free_lin_attn_slots is not None and \
               len(self.free_lin_attn_slots) == 0:
                break
            # ... 现有分配逻辑 ...
            if self.free_lin_attn_slots:
                seq.lin_attn_slot = self.free_lin_attn_slots.pop()

    def postprocess(self, seqs, token_ids, is_prefill):
        for seq, token_id in zip(seqs, token_ids):
            # ... 现有逻辑 ...
            if seq.is_finished and seq.lin_attn_slot >= 0:
                self.free_lin_attn_slots.add(seq.lin_attn_slot)
                seq.lin_attn_slot = -1
```

---

### 8. `nanovllm/engine/model_runner.py`（修改）

```python
class ModelRunner:
    def _build_model(self, hf_config):
        model_type = getattr(hf_config, 'model_type', '')
        if model_type in ('qwen3_5_text',):
            from nanovllm.models.qwen35 import Qwen35ForCausalLM
            return Qwen35ForCausalLM(hf_config)
        else:
            from nanovllm.models.qwen3 import Qwen3ForCausalLM
            return Qwen3ForCausalLM(hf_config)

    def allocate_kv_cache(self):
        hf = self.config.hf_config
        # 仅为 full_attention 层分配 KV cache
        layer_types = getattr(hf, 'layer_types', None)
        num_kv_layers = (
            sum(1 for t in layer_types if t == 'full_attention')
            if layer_types else hf.num_hidden_layers
        )
        # ... 其余计算逻辑同现有代码，将 hf.num_hidden_layers 替换为 num_kv_layers ...

        self.kv_cache = torch.empty(
            2, num_kv_layers, config.num_kvcache_blocks,
            self.block_size, num_kv_heads, head_dim,
        )
        # 仅绑定 AttentionWithKVCache（跳过 GatedDeltaNet）
        layer_id = 0
        for module in self.model.modules():
            if isinstance(module, AttentionWithKVCache):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

        # 分配线性注意力状态
        self._allocate_lin_attn_states()

    def _allocate_lin_attn_states(self):
        from nanovllm.models.qwen35 import GatedDeltaNet
        for module in self.model.modules():
            if isinstance(module, GatedDeltaNet):
                module.allocate_states(self.config.max_num_seqs)

    def prepare_prefill(self, seqs):
        # ... 现有代码 ...
        lin_slots = [seq.lin_attn_slot for seq in seqs]
        set_context(True, cu_q, cu_k, ..., lin_attn_seq_slots=lin_slots)

    def prepare_decode(self, seqs):
        # ... 现有代码 ...
        lin_slots = [seq.lin_attn_slot for seq in seqs]
        set_context(False, ..., lin_attn_seq_slots=lin_slots)
```

---

### 9. `nanovllm/engine/llm_engine.py`（修改）

```python
def __init__(self, model, **kwargs):
    # ... 现有代码 ...
    hf = config.hf_config
    layer_types = getattr(hf, 'layer_types', None)
    is_hybrid = layer_types is not None and 'linear_attention' in layer_types
    num_lin_slots = config.max_num_seqs if is_hybrid else 0

    self.scheduler = Scheduler(
        ...,
        num_lin_attn_slots=num_lin_slots,
    )
```

---

## State Machine（lin_attn_slot 生命周期）

```
                   ┌─────────────────────────────┐
                   │    free_lin_attn_slots        │
                   │  {0, 1, 2, ..., max_seqs-1}  │
                   └──────────────┬───────────────┘
                                  │ pop()
                    ┌─────────────▼──────────────┐
                    │  Sequence.lin_attn_slot = i │
                    │  状态：GDN state[i] 可读写   │
                    │  (WAITING → RUNNING)        │
                    └─────────────┬───────────────┘
                                  │ 序列完成（EOS/max_tokens）
                    ┌─────────────▼──────────────┐
                    │  free_slots.add(i)          │
                    │  seq.lin_attn_slot = -1     │
                    │  状态：GDN state[i] 被下次使用│
                    └────────────────────────────┘
```

**不变式**：
- prefill 调度前：`free_lin_attn_slots` 非空（否则停止 prefill）
- GDN.forward 中：`slot >= 0`，否则为 BUG
- decode 调度：所有 running seq 已有 slot，无需重新分配
- slot 归还：仅在 `postprocess` 中 `seq.is_finished` 时执行

---

## Risks（风险与缓解措施）

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| HF config 字段名与实际模型不符（如 `linear_num_key_heads` vs `gdn_num_key_heads`） | 中 | 构建失败 | `__init__` 中 `getattr(config, field, default)` 防御，启动时打印 config 字段 |
| recurrent_state 在 prefill 后未正确初始化为 0 | 低 | 跨 seq 状态污染 | `allocate_states` 用 `torch.zeros`；slot 回收时不清零（下次 prefill 整体覆盖） |
| Partial RoPE cos_sin_cache 尺寸错误 | 中 | 推理结果偏差 | 单元测试：对比 full RoPE 的前 rotary_dim 维是否一致 |
| Qwen3ForCausalLM 被 kv_cache 分配逻辑破坏 | 低 | 原有 Qwen3 回归 | layer_types 为 None 时走旧路径，保持向后兼容 |
| PyTorch recurrent fallback 数值精度 | 中 | 输出与 HF 偏差较大 | float32 state，f32 计算，对比 HF greedy token 序列 |

---

## Test Plan

### Unit Tests

| 测试点 | 文件 | 验证方法 |
|--------|------|---------|
| Qwen35RMSNorm 零初始化 | `tests/test_qwen35.py` | weight 全零 → 输出 == norm(x) |
| Partial RoPE（rotary_dim < head_size） | `tests/test_rotary.py` | 前 64 维旋转，后 192 维不变 |
| GDN recurrent 状态隔离 | `tests/test_qwen35.py` | 两个 slot 互不影响 |
| loader 前缀剥离 + 跳过 | `tests/test_loader.py` | virtual safetensors，验证参数名映射 |
| Config qwen3_5 → text_config | `tests/test_config.py` | mock AutoConfig，验证 dtype 归一化 |

### Integration Tests

| 测试点 | 验证方法 |
|--------|---------|
| Qwen35ForCausalLM 可构建 + 权重全部加载 | 无 unexpected keys / missing keys |
| KV cache 仅分配 6 层 | `model_runner.kv_cache.shape[1] == 6` |
| GDN state 形状正确 | `gdn.conv_state.shape == (max_seqs, 6144, 4)` |

### E2E Tests

| 测试点 | 验证方法 |
|--------|---------|
| Greedy 输出前 20 token 与 HF 一致 | `torch.equal(nanovllm_tokens, hf_tokens)` |
| 并发 2 路请求状态不串 | 两次单独推理 == 并发推理结果 |
| example.py 跑通 | 无 OOM/CUDA error，输出合理文本 |
