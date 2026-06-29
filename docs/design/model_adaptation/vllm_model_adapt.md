# 大模型适配推理引擎实战指南（以 vLLM 为参照）

> **目标读者**：需要把一个 HuggingFace（HF）格式的新大模型接入高性能推理引擎（vLLM / nano-vllm 一类）的工程师。
> **本文定位**：一份**通用的大模型适配流程**——从"拿到权重"到"提交 PR"的完整工程方法论，覆盖架构分析、代码实现、权重映射、并行、数值对齐与验证。
> **运行示例**：流程中部分环节以 **Qwen3.5-35B-A3B** 适配为例。该模型在 vLLM 中实际是 `Qwen3NextForCausalLM`（混合架构：GatedDeltaNet 线性注意力 + 全注意力 + MoE），既能演示"相似架构复用"的常规路径，也能演示"含新算子"的困难路径。**但所有方法论本身与具体模型无关。**
> 与本仓库（nano-vllm）的 Qwen3.5 适配实践相互印证，见 [qwen35_moe_adaptation/](qwen35_moe_adaptation/) 与 [qwen35_adaptation/](qwen35_adaptation/)。

---

## 0. 适配复杂度分级与三条集成路径

**先评估复杂度，再决定投入。** 集成难度几乎完全由"模型架构与引擎已有实现的差异"决定：

| 难度 | 特征 | 工作量 | 示例 |
|------|------|--------|------|
| ★ 容易 | 与已有模型仅超参不同（层数/头数/专家数） | 改 config，复用现有类 | Qwen3-4B → Qwen3-32B |
| ★★ 中等 | 结构相似但有局部改造（新增 MoE、共享专家、partial RoPE、QK-Norm） | Fork 一个相近模型 + 调整权重映射 | Qwen3 → Qwen3-MoE |
| ★★★ 困难 | 引入**新算子**（新注意力机制 / 线性注意力 / Mamba / MLA） | 需新增注意力 backend、状态缓存与自定义 kernel | Qwen3 → Qwen3.5（GDN 线性注意力） |

**三条集成路径**，按优先级：

1. **先直接试 `vllm serve <model>`。** 现代 vLLM 内置 **Transformers 建模后端**，很多 decoder-only 语言模型无需任何适配代码即可自动加载。**能跑通就不要写代码。**
2. **Out-of-tree 插件（不改引擎源码）。** 通过插件在外部注册模型，适合私有模型或快速迭代：

   ```python
   # 插件入口；用字符串路径做惰性导入，避免 fork 子进程时 "Cannot re-initialize CUDA"
   def register():
       from vllm import ModelRegistry
       ModelRegistry.register_model(
           "Qwen3_5MoeForCausalLM",
           "your_pkg.qwen3_5:Qwen3_5ForCausalLM",
       )
   ```
3. **Built-in（改引擎源码 / 提 PR）。** 把模型文件放进 `vllm/model_executor/models/`，在 `registry.py` 注册，最终回馈上游。本文主要讲这条路径，因为它涵盖了全部环节。

> **原则贯穿全文**：能复用就不要重写，能继承就不要复制。每多写一行模型代码，就多一处与上游漂移、需要长期维护的风险。

---

## 通用适配流程总览

```
0. 评估复杂度 / 试 Transformers 后端
        │
1. 解析模型结构（config.json + 权重清单）──► 得到"架构指纹"
        │
2. 定位 HF 参照实现，与最相近的已支持模型做 diff
        │
3. 建立"复用 / 改造 / 新增"差异矩阵，选定基线模型
        │
4. 实现模型代码（满足引擎兼容约定：prefix / 扁平 forward / 继承复用）
        │
5. 替换并行层（TP）与量化接口
        │
6. 适配 MoE（FusedMoE）│ 7. 适配非标准算子 / 混合架构（如需）
        │
8. 实现权重加载 load_weights（stacked + expert 映射）
        │
9. 注册 Architecture 与 Config
        │
10. 分级验证（加载 → logits 对齐 → 采样一致 → 并行 → 长上下文 → 推理特性 → benchmark）
        │
11. 补单元测试，按 Checklist 提 PR
```

---

## 1. 第一步：解析模型结构

适配的第一动作不是写代码，而是**读懂 `config.json` 和权重清单**，提炼一份"架构指纹"。

### 1.1 config.json 重点字段

| 字段 | 含义 | 适配影响 |
|------|------|----------|
| `architectures` | 架构类名（注册键） | 决定 registry 注册名 |
| `model_type` | 配置类型 | 决定 config 解析路径 |
| `hidden_size` / `num_hidden_layers` | 隐藏维 / 层数 | 基本规模 |
| `num_attention_heads` / `num_key_value_heads` | Q 头 / KV 头 | 相等→MHA，KV 少→**GQA**，KV=1→MQA |
| `head_dim` | 单头维度 | 不一定等于 hidden/heads（Qwen3.5 为 256） |
| `intermediate_size` / `moe_intermediate_size` | FFN / 专家 FFN 宽度 | dense vs MoE |
| `num_experts` / `num_experts_per_tok` | 专家数 / top-k | MoE 路由 |
| `shared_expert_intermediate_size` | 共享专家宽度 | 是否有共享专家 |
| `rope_theta` / `rope_scaling` / `partial_rotary_factor` | RoPE 配置 | 长上下文与 partial RoPE |
| `tie_word_embeddings` | 词嵌入与 LM head 共享 | 权重加载分支 |
| `layer_types` / `layers_block_type` | 每层类型 | **混合/滑窗架构必读** |

### 1.2 权重清单也要看

```bash
huggingface-cli download Qwen/Qwen3.5-35B-A3B --local-dir Qwen3.5-35B-A3B
# 列出权重名（无需加载即可用 safetensors header 解析）
python -c "from safetensors import safe_open; \
import glob; f=sorted(glob.glob('Qwen3.5-35B-A3B/*.safetensors'))[0]; \
[print(k) for k in safe_open(f,'pt').keys()][:50]"
```

权重命名直接决定后续 `load_weights` 的映射规则——`q_proj/k_proj/v_proj` 是否要融合成 `qkv_proj`、`experts.N.*` 如何映射到 `FusedMoE`、是否存在 `visual.*` / `mtp.*` 等需跳过的旁支。

### 1.3 示例：Qwen3.5-35B-A3B 的架构指纹

读完 config 与权重后应能写出这样一张卡片（具体数值以实际 config.json 为准）：

```text
Decoder-only · RMSNorm · SwiGLU · GQA · 旋转位置编码（partial RoPE）
混合层结构：GatedDeltaNet 线性注意力 : 全注意力 ≈ 3 : 1（layer_types 标注）
FFN：MoE — 256 专家，top-8 激活 + 1 共享专家
词嵌入：tie_word_embeddings
旁支权重：visual.*（VLM 视觉塔）、mtp.*（多 token 预测）→ 纯文本推理需跳过
```

**关键判断**：它**不是** Qwen3 的纯 MoE 变体——线性注意力层是引擎里没有的**新算子**，落入 ★★★ 困难档。这一步若误判为"Fork Qwen3 即可"，会在权重加载阶段才发现 GDN 层无处安放。**架构分析的价值正在于此。**

---

## 2. 第二步：定位 HF 参照实现并做 diff

确定 HF 侧建模文件，逐类对比最相近的已支持模型：

```text
transformers/models/qwen3_next/modeling_qwen3_next.py   ← 目标
transformers/models/qwen3_moe/modeling_qwen3_moe.py     ← 最相近的基线
```

对 `*Attention` / `*MLP` / `*SparseMoeBlock` / `*DecoderLayer` / `*Model` / `*ForCausalLM` 逐一 diff，回答三个问题：

1. **哪些类逐行相同？** → 直接继承复用。
2. **哪些类只是超参/小改？** → Fork 后微调。
3. **哪些类是全新机制？** → 需要在引擎侧新增算子/层。

---

## 3. 第三步：建立差异矩阵，选定基线模型

把 diff 结论落成一张"复用 / 改造 / 新增"矩阵，作为实现计划：

| 模块 | 基线(Qwen3-MoE) | 目标(Qwen3.5) | 策略 |
|------|----------------|---------------|------|
| RMSNorm | √ | √ | **复用** |
| RoPE | √ | √(partial 0.25) | 复用，传 `partial_rotary_factor` |
| 全注意力 + QK-Norm | √ | √ | **复用/微改** |
| MoE Router / FusedMoE | √ | √(256/top-8+shared) | 复用，调超参 |
| 线性注意力(GatedDeltaNet) | × | √ | **新增算子 + 状态缓存** |
| MTP（多 token 预测） | × | √ | 可选，纯推理可跳过 |

**选基线模型的标准**：选"差异最小、且已支持你需要的特性（MoE/GQA/PP）"的那个，而不是名字最像的那个。Qwen3.5 选 `qwen3_moe` 作 MoE/注意力骨架的基线，再叠加 `mamba`/线性注意力的状态管理范式。

---

## 4. 第四步：实现模型代码（引擎兼容约定）

把 HF 模型代码移植成引擎风格，需满足几条硬约定（以 vLLM 为例）：

### 4.1 每个子模块都接受 `prefix`

`prefix` 是模块在 state_dict 中的全名，用于：① 给每个 Attention 算子唯一名字（运行时按名注册，避免冲突）；② 支持**非均匀量化**（按 prefix 匹配量化配置，决定该层是否量化）。

```python
class Qwen3_5DecoderLayer(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str):
        super().__init__()
        # 子模块 prefix 逐级拼接
        self.self_attn = Qwen3_5Attention(prefix=f"{prefix}.self_attn")
        self.mlp = Qwen3_5SparseMoeBlock(vllm_config, prefix=f"{prefix}.mlp")
```

### 4.2 计算接口：`embed_input_ids` + 扁平化 `forward`

```python
class Qwen3_5Model(nn.Module):
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        # 统一的文本嵌入接口（便于被组合进多模态模型）
        return self.embed_tokens(input_ids)

class Qwen3_5ForCausalLM(nn.Module, SupportsPP, SupportsLoRA, MixtureOfExperts):
    def forward(
        self,
        input_ids: torch.Tensor,         # 扁平张量：[total_tokens]，无 [batch, seq] 维度
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,  # PP 用
        inputs_embeds: torch.Tensor | None = None,                # 多模态用
    ) -> torch.Tensor:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states):
        return self.logits_processor(self.lm_head, hidden_states)
```

要点：删去训练相关代码；`input_ids`/`positions` 视为**单维扁平张量**（连续批处理把所有序列拼平），没有 max-seq-len 维度。

### 4.3 用继承吃掉"相同"的部分

注意力若与基线完全一致，一行继承即可，绝不复制：

```python
class Qwen3_5Attention(Qwen3MoeAttention):
    pass
```

### 4.4 声明能力接口（Protocol）

引擎用一组 `Protocol` 判断模型支持哪些特性，按需声明：

| 接口 | 含义 |
|------|------|
| `SupportsPP` | 流水线并行 |
| `SupportsLoRA` | LoRA |
| `MixtureOfExperts` | MoE（暴露专家元数据、支持 EPLB） |
| `SupportsEagle3` | 投机解码 |
| `IsAttentionFree` | 纯 Mamba（无注意力，无 KV cache） |
| `IsHybrid` | **混合架构（注意力 + Mamba/线性注意力）** ← Qwen3.5 走这条 |

---

## 5. 第五步：张量并行（TP）与量化层替换

把普通 `nn.Linear` / `nn.Embedding` 替换为引擎的并行版本，并行与权重切分逻辑由这些层内部处理：

| 层 | 切分方式 | 典型用途 |
|----|----------|----------|
| `ReplicatedLinear` | 不切分（复制） | MoE gate、shared_expert_gate |
| `ColumnParallelLinear` | 按输出维(列)切 | 注意力 QKV、FFN 第一层 |
| `RowParallelLinear` | 按输入维(行)切 + all-reduce | 注意力 o_proj、FFN 第二层 |
| `MergedColumnParallelLinear` | 合并多个列并行(如 gate+up) | SwiGLU 的 gate_up_proj |
| `QKVParallelLinear` | QKV 合并列并行，KV 头不足时自动复制 | 注意力 qkv_proj |
| `VocabParallelEmbedding` / `ParallelLMHead` | 词表维切分 | 输入嵌入 / 输出头 |

约束：`hidden_size % tp_size == 0`、`num_kv_heads` 与 `tp_size` 的关系决定 KV 头是否需要复制（GQA 在 `tp_size > num_kv_heads` 时由 `QKVParallelLinear` 自动复制 KV）。所有并行 Linear 接受 `quant_config`，量化由引擎按 `prefix` 注入。

---

## 6. 第六步：MoE 适配（FusedMoE）

MoE 是高难点，核心是**绝不要用 `nn.ModuleList` 逐专家循环**——会让吞吐崩塌。必须用引擎的融合算子 `FusedMoE` / `SharedFusedMoE`（后者把共享专家与路由专家一起融合）。

```python
class Qwen3_5SparseMoeBlock(nn.Module):
    def __init__(self, vllm_config, prefix=""):
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        # 路由器：复制，不切分
        self.gate = ReplicatedLinear(config.hidden_size, config.num_experts,
                                     bias=False, prefix=f"{prefix}.gate")
        # 共享专家（若 shared_expert_intermediate_size > 0）
        self.shared_expert = Qwen3_5MLP(..., reduce_results=False) \
            if getattr(config, "shared_expert_intermediate_size", 0) > 0 else None
        # 融合专家：一次 kernel 完成 top-k 路由 + 专家计算
        self.experts = SharedFusedMoE(
            shared_experts=self.shared_expert,
            gate=self.gate,
            num_experts=config.num_experts,          # 如 256
            top_k=config.num_experts_per_tok,        # 如 8
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.norm_topk_prob,
            reduce_results=False,
        )

    def forward(self, hidden_states):
        router_logits, _ = self.gate(hidden_states)
        shared_out, fused_out = self.experts(hidden_states=hidden_states,
                                             router_logits=router_logits)
        out = shared_out + fused_out if shared_out is not None else fused_out
        # TP>1 时由 experts 统一做 all-reduce
        return self.experts.maybe_all_reduce_tensor_model_parallel(out)
```

MoE 的并行有专门维度——**专家并行(EP)**：`ep_size` 决定每个 rank 持有的物理专家数（`n_physical_experts // ep_size`），并支持 EPLB（专家负载均衡，含冗余专家）。这部分与权重加载的 `expert_params_mapping` 紧密耦合，见第 8 步。

---

## 7. 第七步：非标准算子 / 混合架构（困难路径）

这是 Qwen3.5 真正的难点，也是"通用流程"必须覆盖的分支：**当模型含引擎没有的算子时**，光 Fork 模型文件不够，要动引擎底层。

vLLM 把"非标准注意力/状态类"层分三类处理：

1. **纯 Mamba（无注意力）**：模型继承 `IsAttentionFree`，用 `MambaMixer` / `MambaMixer2`，实现 `get_mamba_state_shape_from_config` / `get_mamba_state_dtype_from_config`。
2. **Mamba + 注意力混合**（如 Jamba/Bamba）：继承 `IsHybrid`，KV cache 与 Mamba 状态缓存共存。
3. **类 Mamba 机制 + 注意力**（线性注意力 / ShortConv）：参考 `MiniMaxText01LinearAttention`、`Lfm2`，自定义"mamba-like"层。**Qwen3.5 的 GatedDeltaNet 属于此类。**

对第 3 类，落地清单：

- 自定义层继承 `MambaBase`，实现 `get_state_dtype` / `get_state_shape` / `mamba_type` / `get_attn_backend`；
- 实现该机制的 **attention metadata** 类（参考 `LinearAttentionMetadata`），统一管理跨层元数据；
- 在 attention backend 注册表（`MAMBA_TYPE_TO_BACKEND_MAP` / `MambaAttentionBackendEnum`）登记新 backend；
- **状态语义不同于 KV cache**：注意力的 KV 是"追加"，线性注意力/Mamba 的状态是"原地更新"（conv_state + recurrent_state），其生命周期、显存分配、prefix caching 都要单独处理；
- 若要支持 `torch.compile` + CUDA Graph，把该层调用包进 `direct_register_custom_op` 自定义算子，并把算子名加入 `_attention_ops`，否则 piecewise CUDA graph 会出错。

> **混合架构的 KV cache 要点**：只有"全注意力层"才分配 KV cache，线性注意力层分配的是固定大小的状态张量。这一点在本仓库 nano-vllm 的实现里同样成立，详见 [qwen35_adaptation/design.md](qwen35_adaptation/design.md)。

---

## 8. 第八步：权重加载 `load_weights`（最关键环节）

HF 权重名 ≠ 引擎参数名，`load_weights` 负责桥接。两条主线：**stacked 映射**（融合 qkv / gate_up）与 **expert 映射**（融合专家）。

### 8.1 顶层用 `AutoWeightsLoader`

```python
def load_weights(self, weights):
    loader = AutoWeightsLoader(self)   # 自动按子模块递归分发
    return loader.load_weights(weights)
```

### 8.2 stacked_params_mapping：融合 QKV 与 gate_up

```python
stacked_params_mapping = [
    # (引擎参数名, checkpoint 名, shard_id)
    ("qkv_proj", "q_proj", "q"),
    ("qkv_proj", "k_proj", "k"),
    ("qkv_proj", "v_proj", "v"),
    ("gate_up_proj", "gate_proj", 0),
    ("gate_up_proj", "up_proj", 1),
]
```

同时声明 `packed_modules_mapping`（供量化/LoRA 识别融合层）：

```python
class Qwen3_5MoeForCausalLM(...):
    packed_modules_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}
```

### 8.3 expert_params_mapping：融合专家

```python
def get_expert_mapping(self):
    return SharedFusedMoE.make_expert_params_mapping(
        self,
        ckpt_gate_proj_name="gate_proj",
        ckpt_down_proj_name="down_proj",
        ckpt_up_proj_name="up_proj",
        num_experts=self.config.num_experts,
        num_redundant_experts=self.num_redundant_experts,
    )
```

加载主循环的关键控制流（精简自 `qwen3_moe.py`）：

```python
for name, loaded_weight in weights:
    # 1) 先尝试 stacked（qkv / gate_up）；专家权重在此跳过，留给下面处理
    for param_name, weight_name, shard_id in stacked_params_mapping:
        if weight_name not in name or "mlp.experts" in name:
            continue
        name = name.replace(weight_name, param_name)
        if is_pp_missing_parameter(name, self):   # PP：本 rank 无此层 → 跳过
            continue
        param = params_dict[name]
        param.weight_loader(param, loaded_weight, shard_id)
        break
    else:
        # 2) 专家权重：带 expert_id/shard_id，weight_loader 返回是否命中本 rank
        for p_name, w_name, expert_id, shard_id in expert_params_mapping:
            if w_name not in name:
                continue
            mapped = name.replace(w_name, p_name)
            success = params_dict[mapped].weight_loader(
                param, loaded_weight, mapped,
                shard_id=shard_id, expert_id=expert_id, return_success=True)
            if success:
                break
        # 3) 其余常规权重直接 default_weight_loader
```

### 8.4 加载阶段还要处理的细节

- **前缀剥离 / 旁支跳过**：跳过 `visual.*` / `mtp.*` 等纯推理用不到的权重（Qwen3.5 必做）。
- **`tie_word_embeddings`**：`self.lm_head.weight = self.model.embed_tokens.weight`，且不要重复加载 lm_head。
- **量化标量**：`.weight_scale` / `.input_scale` / kv-scale 等后缀按 `ignore_suffixes` 与 `maybe_remap_kv_scale_name` 处理。
- **PP 缺层**：`is_pp_missing_parameter` 过滤不属于本 rank 的层。
- **EP 未命中**：专家不在本 rank 时 `return_success=False`，跳过而非报错。

---

## 9. 第九步：注册 Architecture 与 Config

### 9.1 注册模型架构

把模型类加入 `vllm/model_executor/models/registry.py` 的 `_VLLM_MODELS`（**按字母序**）：

```python
"Qwen3MoeForCausalLM": ("qwen3_moe", "Qwen3MoeForCausalLM"),
"Qwen3NextForCausalLM": ("qwen3_next", "Qwen3NextForCausalLM"),  # Qwen3.5 实际入口
# 键 = config.json 的 architectures；值 = (模块名, 类名)
```

未注册时启动报 `Architecture not supported`。

### 9.2 注册/解析 Config

vLLM 通过 `config.json` 的 `architectures` 解析模型；解析失败的常见原因与对策：

- `architectures` 缺失或被非官方仓库改名 → 用 `hf_overrides` 强制指定：

  ```python
  LLM(model=..., hf_overrides={"architectures": ["Qwen3NextForCausalLM"]})
  ```
- `model_type` 是自定义类型 → 在 `_CONFIG_REGISTRY` 注册自定义 `PretrainedConfig`。
- **多模态/复合模型**：语言骨干配置嵌在 `text_config`（或 `llm_config`）里，用 `config.get_text_config()` 取出。Qwen3.5（VLM）的纯文本推理正是从顶层 config 提取 `text_config`，本仓库 nano-vllm 在 `Config.__post_init__` 里做了同样的事。

---

## 10. 第十步：分级验证（决定成败）

适配的正确性必须**分级递进**验证，每级通过再进下一级，定位问题成本最低：

| 级别 | 验证内容 | 方法 / 判据 |
|------|----------|-------------|
| **L0 加载** | 模型能用 dummy 权重初始化 | `tests/models/registry.py` 加载测试不报错 |
| **L1 数值对齐** | 单步 logits 与 HF 一致 | `max(abs(hf_logits - vllm_logits)) < 1e-3`（BF16 可放宽到 ~1e-2） |
| **L2 采样一致** | 解码结果一致 | greedy 文本逐 token 一致；采样 logprobs 落在 HF top-k 内 |
| **L3 并行** | TP/PP/EP 正确 | `tp=2/8` 输出与 `tp=1` 一致；检查 QKV/Row/Merged 切分与 `expert_id` 映射 |
| **L4 长上下文** | RoPE scaling 正确 | 32K/64K/128K 不崩、不乱码，验证 `rope_scaling` |
| **L5 推理特性** | 引擎特性正常 | `--enable-prefix-caching`（命中率合理）、chunked prefill、千级并发连续批处理 |
| **L6 性能** | 吞吐/显存达标 | `vllm bench throughput`：tokens/s、GPU util、KV cache 占用，对比 HF 应吞吐升、显存降 |

> **L1 是分水岭**：logits 对齐失败几乎都来自权重映射错误（qkv 顺序、gate/up 反了、专家 shard_id 错、partial RoPE 维度、QK-Norm 漏接、缩放因子）。先用**单层**对齐缩小范围，再扩到整模型。

**资源受限时的验证技巧**（本仓库实践）：用**减层 config** 跑通正确性——把 `num_hidden_layers` 改成 3，软链接原始权重，即可在 8GB 显存上验证 35B MoE 的结构正确性，详见 [qwen35_moe_adaptation/testing.md](qwen35_moe_adaptation/testing.md)。

---

## 11. 第十一步：单元测试与提交 PR

### 11.1 必需 / 可选测试（vLLM CI 口径）

- **必需**：在 `tests/models/registry.py` 登记一个 HF 仓库示例，CI 会用 dummy 权重测加载。需要 HF 开发版时设 `min_transformers_version` 跳过。
- **可选但强烈建议**——正确性对比测试（`tests/models/utils.py`）：
  - `check_outputs_equal`：vLLM 文本与 HF **逐字一致**；
  - `check_logprobs_close`：vLLM logprobs 落在 HF top-k 内（反之亦然）。

### 11.2 提交前 Checklist

```text
功能      [ ] Architecture 注册  [ ] Config 解析  [ ] Attention  [ ] MoE  [ ] load_weights
正确性    [ ] logits 对齐(<阈值)  [ ] greedy 一致  [ ] sampling logprobs 一致
并行      [ ] TP   [ ] PP   [ ] EP（专家分片正确）
推理特性  [ ] Prefix Cache  [ ] Chunked Prefill  [ ] Continuous Batching
混合/算子 [ ] 状态缓存形状/dtype  [ ] 自定义算子注册(CUDA Graph 兼容)
性能      [ ] 吞吐无明显下降  [ ] 显存无明显增加
工程      [ ] 跳过 visual/mtp 等旁支权重  [ ] tie_word_embeddings 处理  [ ] 更新 supported_models 文档
```

---

## 12. 常见坑（FAQ）

| 症状 | 根因 | 对策 |
|------|------|------|
| `Architecture not supported` | 未注册 / `architectures` 不匹配 | registry 注册或 `hf_overrides` |
| 加载报 shape 不匹配 | qkv/gate_up 融合顺序错、专家 shard_id 错 | 核对 `stacked_params_mapping` / `expert_params_mapping` |
| 第二条 prompt 起输出乱码 | 跨序列注意力污染（连续批处理边界） | 见 [bug-prefill-cross-seq-attention.md](bug-prefill-cross-seq-attention.md) |
| logits 偏差大但不崩 | partial RoPE 维度、QK-Norm 漏接、scaling 错 | 单层对齐逐步排查 |
| 吞吐暴跌 | MoE 用了 `nn.ModuleList` 而非 `FusedMoE` | 改用融合算子 |
| 长上下文崩溃 | `rope_scaling` 未生效 | 校验 RoPE 配置与 `max_model_len` |
| 滑窗模型结果错 | 未按层解析滑窗 | config 加 `layer_types`，逐层传 `per_layer_sliding_window` |
| 混合架构 CUDA Graph 报错 | 自定义算子未注册进 `_attention_ops` | `direct_register_custom_op` + 登记算子名 |
| 专家维度在大模型才暴露的 Bug | `nv != nk`（小模型 nk=nv 掩盖） | 见 [qwen35_moe_adaptation/design.md](qwen35_moe_adaptation/design.md) 的 GDN nv/nk 修复 |

---

## 附录：与 nano-vllm 适配实践的对应

本指南以 vLLM 为参照梳理通用流程；本仓库 nano-vllm 在更小的代码量上走通了同一条路，可作为"最小可读实现"对照阅读：

| 本指南环节 | nano-vllm 对应文档 |
|------------|--------------------|
| 架构分析 / 复用策略 | [qwen35_adaptation/research.md](qwen35_adaptation/research.md)、[qwen35_2b_adaptation.md](qwen35_2b_adaptation.md) |
| 混合架构设计（GDN + 全注意力 + KV cache 分配） | [qwen35_adaptation/design.md](qwen35_adaptation/design.md) |
| MoE + GDN nv/nk 修复 + 减层适配 | [qwen35_moe_adaptation/design.md](qwen35_moe_adaptation/design.md)、[qwen35_moe_adaptation/research.md](qwen35_moe_adaptation/research.md) |
| 分级验证 / 资源受限测试 | [qwen35_adaptation/testing.md](qwen35_adaptation/testing.md)、[qwen35_moe_adaptation/testing.md](qwen35_moe_adaptation/testing.md) |
| 权重加载踩坑 | [bug-prefill-cross-seq-attention.md](bug-prefill-cross-seq-attention.md) |

> **参考来源**：vLLM 官方贡献指南 `docs/contributing/model/`（basic / registration / tests）、`docs/configuration/model_resolution.md`，及 `vllm/model_executor/models/qwen3_moe.py`、`qwen3_next.py`、`interfaces.py`、`registry.py`（vLLM main 分支）。流程与 API 名称以引擎实际版本为准。
