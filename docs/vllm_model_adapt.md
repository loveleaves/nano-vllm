# Qwen3.5-35B-A3B 适配 vLLM 实战指南

huggingface-cli download     Qwen/Qwen3.5-2B     --local-dir Qwen3.5-2B

## 1. 目标

假设当前 vLLM 已支持：

* Qwen3-4B
* Qwen3-8B
* Qwen3-32B
* Qwen3-30B-A3B

但尚未支持：

* Qwen3.5-35B-A3B

实现目标：

```bash
vllm serve /data/Qwen3.5-35B-A3B
```

能够正常完成：

* 模型加载
* PagedAttention
* KV Cache
* Tensor Parallel
* Prefix Cache
* Continuous Batching

---

# 2. 第一步：分析 HF 模型结构

首先查看：

```text
config.json
```

重点关注：

```json
{
  "architectures": [
    "Qwen3_5MoeForCausalLM"
  ],

  "model_type": "qwen3_5",

  "hidden_size": 4096,

  "num_hidden_layers": 48,

  "num_attention_heads": 32,

  "num_key_value_heads": 8,

  "num_experts": 128,

  "num_experts_per_tok": 8
}
```

核心结论：

Qwen3.5-35B-A3B：

```text
Decoder Only
MoE
GQA
RoPE
RMSNorm
SwiGLU
```

与 Qwen3-A3B 非常接近。

因此：

优先采用：

```text
Fork Qwen3
而不是重写
```

---

# 3. 第二步：确认 HF Reference

查看：

```text
transformers/models/qwen3_5/
```

重点文件：

```text
modeling_qwen3_5.py
```

分析：

```python
Qwen3_5Attention

Qwen3_5MLP

Qwen3_5SparseMoeBlock

Qwen3_5DecoderLayer

Qwen3_5Model

Qwen3_5ForCausalLM
```

然后与：

```python
transformers/models/qwen3/
```

做 diff。

---

# 4. 第三步：确定复用策略

建立差异表：

| 模块          | Qwen3 | Qwen3.5 |
| ----------- | ----- | ------- |
| RMSNorm     | √     | √       |
| RoPE        | √     | √       |
| GQA         | √     | √       |
| SwiGLU      | √     | √       |
| MoE Router  | √     | √       |
| TopK Expert | Top4  | Top8    |
| MLA         | ×     | ×       |
| MTP         | ×     | √       |

若发现：

```text
Attention完全一致
```

则：

```python
直接复用Qwen3Attention
```

---

# 5. 新建模型目录

复制：

```text
vllm/model_executor/models/qwen3.py
```

生成：

```text
vllm/model_executor/models/qwen3_5.py
```

结构：

```python
Qwen3_5Attention

Qwen3_5MLP

Qwen3_5MoE

Qwen3_5DecoderLayer

Qwen3_5Model

Qwen3_5ForCausalLM
```

---

# 6. 注册 Architecture

修改：

```python
vllm/model_executor/models/registry.py
```

新增：

```python
_MODEL_REGISTRY.update(
{
    "Qwen3_5MoeForCausalLM":
    (
        "qwen3_5",
        "Qwen3_5ForCausalLM"
    )
})
```

否则启动直接报：

```text
Architecture not supported
```

---

# 7. 注册 Config

修改：

```python
vllm/transformers_utils/config.py
```

新增：

```python
if config.model_type == "qwen3_5":
    ...
```

或者：

```python
AutoConfig.register(
    "qwen3_5",
    Qwen3_5Config
)
```

否则：

```python
AutoConfig.from_pretrained()
```

失败。

---

# 8. Attention适配

Qwen3 已实现：

```python
class Qwen3Attention
```

内部：

```python
self.qkv_proj

self.o_proj

self.attn = Attention(...)
```

vLLM 文档中的 Qwen3 模型实现也是基于统一 Attention Backend 构建。

若 Qwen3.5 Attention 无变化：

```python
class Qwen3_5Attention(Qwen3Attention):
    pass
```

即可。

---

# 9. MoE适配

重点：

```python
Qwen3_5SparseMoeBlock
```

映射：

```python
FusedMoE
```

不能直接：

```python
nn.ModuleList
```

否则吞吐量暴跌。

参考：

```python
Qwen3MoE
MixtralMoE
DeepSeekMoE
```

实现：

```python
self.experts = FusedMoE(...)
```

关键参数：

```python
num_experts

top_k

hidden_size

intermediate_size
```

---

# 10. 权重映射

最关键环节。

HF：

```python
q_proj.weight

k_proj.weight

v_proj.weight
```

vLLM：

```python
qkv_proj.weight
```

建立映射：

```python
stacked_params_mapping = [

("qkv_proj","q_proj","q"),

("qkv_proj","k_proj","k"),

("qkv_proj","v_proj","v"),

]
```

MoE：

HF：

```python
experts.0.w1
experts.0.w2
...
```

映射：

```python
fused_experts.weight
```

---

# 11. LoadWeights实现

新增：

```python
def load_weights(...)
```

主要完成：

## Attention融合

```python
q
k
v

→

qkv
```

## Gate融合

```python
gate_proj
up_proj

→

gate_up_proj
```

## MoE融合

```python
expert_weights

→

fused_experts
```

---

# 12. KV Cache验证

检查：

```python
num_attention_heads

num_key_value_heads
```

例如：

```python
32 heads

8 kv heads
```

说明：

```text
GQA
```

验证：

```python
Attention(
    num_heads=32,
    num_kv_heads=8
)
```

是否正确。

---

# 13. Tensor Parallel验证

验证：

```bash
tensor_parallel_size=2
```

重点检查：

```python
QKVParallelLinear

MergedColumnParallelLinear

RowParallelLinear
```

确保：

```python
hidden_size % tp_size == 0
```

---

# 14. MoE Parallel验证

验证：

```bash
tensor_parallel_size=8
```

检查：

```python
expert_id
```

映射是否正确。

避免：

```text
expert shard错误
```

---

# 15. Logits对齐测试

最重要测试。

HF：

```python
hf_logits
```

vLLM：

```python
vllm_logits
```

对比：

```python
torch.max(
 abs(
   hf_logits-vllm_logits
 )
)
```

要求：

```text
< 1e-3
```

---

# 16. 长上下文测试

测试：

```bash
32K
64K
128K
```

验证：

```python
rope_scaling
```

是否正确。

---

# 17. Prefix Cache测试

验证：

```bash
--enable-prefix-caching
```

观察：

```text
cache hit rate
```

是否正常。

---

# 18. Continuous Batching测试

构造：

```python
1000 requests
```

验证：

```text
prefill

decode

scheduler
```

正常运行。

---

# 19. Benchmark

测试：

```bash
vllm bench throughput
```

关注：

```text
tokens/s

GPU util

KV Cache占用
```

与 HF 对比：

```text
吞吐提升
显存下降
```

符合预期。

---

# 20. 提交PR前Checklist

## 功能

* [ ] Architecture注册
* [ ] Config注册
* [ ] Attention支持
* [ ] MoE支持
* [ ] Weight Loader支持

## 正确性

* [ ] logits一致
* [ ] greedy一致
* [ ] sampling一致

## 并行

* [ ] TP
* [ ] PP
* [ ] EP

## 推理

* [ ] Prefix Cache
* [ ] Chunked Prefill
* [ ] Continuous Batch

## 性能

* [ ] 吞吐无明显下降
* [ ] 显存无明显增加
