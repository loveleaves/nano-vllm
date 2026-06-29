# Bug 定位与解决：Prefill 跨序列注意力污染

## 现象

运行 `example.py`，第一条 prompt 输出正常，第二条 prompt 输出完全乱码，表现为某个 token 无限重复（如 `"iónióniónión..."`）。

```
Prompt: '...introduce yourself...'
Completion: "<think>Okay, the user wants me to introduce myself..."  ← 正常

Prompt: '...list all prime numbers within 100...'
Completion: '<think>aciónaciónacióniónión...'  ← 乱码
```

## 根本原因

**文件**：`nanovllm/engine/model_runner.py`，`AttentionWithKVCache.forward`，prefill 分支（修复前约第 73-83 行）

```python
# 修复前的错误代码
if context.is_prefill:
    if self.num_kv_groups > 1:
        k = k.repeat_interleave(self.num_kv_groups, dim=1)
        v = v.repeat_interleave(self.num_kv_groups, dim=1)
    q_t = q.transpose(0, 1).unsqueeze(0)   # [1, num_heads, ALL_TOKENS, head_dim]
    k_t = k.transpose(0, 1).unsqueeze(0)
    v_t = v.transpose(0, 1).unsqueeze(0)
    o = F.scaled_dot_product_attention(q_t, k_t, v_t, scale=self.scale, is_causal=True)
    return o.squeeze(0).transpose(0, 1)
```

### 问题机制

Scheduler 在一次 prefill step 中会把多个请求的 token 拼接成一个 batch。假设：

- Seq 0（"introduce yourself" 对话模板）= 20 tokens，位置 0..19
- Seq 1（"list all prime numbers..." 对话模板）= 23 tokens，位置 20..42

拼接后共 43 个 token，全部传入 `F.scaled_dot_product_attention(..., is_causal=True)`。

`is_causal=True` 只生成**全局**下三角掩码：

```
位置   0  1 ... 19 | 20 21 ... 42
  0  [ ✓  ×  ×  × |  ×  ×  ×  × ]
  1  [ ✓  ✓  ×  × |  ×  ×  ×  × ]
 ...
 19  [ ✓  ✓  ✓  ✓ |  ×  ×  ×  × ]
 ─────────────── seq 边界 ────────────
 20  [ ✓  ✓  ✓  ✓ |  ✓  ×  ×  × ]  ← Seq 1 的第 0 个 token 能 attend Seq 0！
 21  [ ✓  ✓  ✓  ✓ |  ✓  ✓  ×  × ]
 ...
```

**后序序列的每个 token 都能 attend 到前序序列的全部 token**，注意力输出被污染。污染后的 hidden states 写入 KV cache，后续 decode 步读取这些错误的 KV 值，产生乱码。

Seq 0 不受影响（所有前序 token 都属于自己），所以第一条 prompt 输出正常。

相同的 bug 也存在于 `nanovllm/layers/attention.py`（Phase 2 的 `Attention` 类，Phase 3 中已被 `AttentionWithKVCache` 替换，但代码逻辑一致，同步修复）。

## 定位步骤

1. 运行 `example.py`，观察第二条输出乱码
2. 排查 decode 阶段：`AttentionWithKVCache.forward` decode 路径对每条序列独立循环，逻辑正确
3. 排查 prefill 阶段：prefill 路径将所有序列 token 拼在一起做单次 SDPA，`is_causal=True` 的 mask 不感知序列边界
4. 验证：将两条 prompt 改为单独提交（一次 generate 一条），输出均正常 → 确认是多序列 batch prefill 的跨序列 attend 问题

## 修复方案

在 prefill 路径中，用 `context.cu_seqlens_q` 取出每条序列的边界，对每条序列**独立**计算 causal attention，再 cat 拼回。

### `nanovllm/engine/model_runner.py` — `AttentionWithKVCache.forward`

```python
# 修复后
if context.is_prefill:
    cu_q = context.cu_seqlens_q  # [num_seqs+1]
    out_parts = []
    for s in range(cu_q.shape[0] - 1):
        s0, s1 = cu_q[s].item(), cu_q[s + 1].item()
        q_s, k_s, v_s = q[s0:s1], k[s0:s1], v[s0:s1]
        if self.num_kv_groups > 1:
            k_s = k_s.repeat_interleave(self.num_kv_groups, dim=1)
            v_s = v_s.repeat_interleave(self.num_kv_groups, dim=1)
        q_t = q_s.transpose(0, 1).unsqueeze(0)
        k_t = k_s.transpose(0, 1).unsqueeze(0)
        v_t = v_s.transpose(0, 1).unsqueeze(0)
        o_s = F.scaled_dot_product_attention(q_t, k_t, v_t, scale=self.scale, is_causal=True)
        out_parts.append(o_s.squeeze(0).transpose(0, 1))
    return torch.cat(out_parts, dim=0)
```

### `nanovllm/layers/attention.py` — `Attention.forward`（同步修复）

```python
if context.is_prefill:
    cu_q = context.cu_seqlens_q
    if cu_q is None:  # 单序列 fallback
        ...  # 原有逻辑
    out_parts = []
    for s in range(cu_q.shape[0] - 1):
        s0, s1 = cu_q[s].item(), cu_q[s + 1].item()
        # 与上方相同的逐序列计算逻辑
        ...
    return torch.cat(out_parts, dim=0)
```

## 修复后的正确输出

```
Prompt: '...introduce yourself...'
Completion: "<think>Okay, the user wants me to introduce myself..."

Prompt: '...list all prime numbers within 100...'
Completion: "<think>Okay, so I need to list all the prime numbers between 1 and 100..."
```

## 受影响范围

| 场景 | 是否受影响 |
|------|-----------|
| 单条 prompt 推理 | 否 |
| 多条 prompt 同时推理（batch prefill） | 是（除第一条外均受污染） |
| decode 阶段 | 否（已对每条序列独立循环） |

## 深层原因与规范做法

vLLM 等生产推理框架在 prefill 阶段使用 `flash_attn_varlen_func`，该接口接受 `cu_seqlens` 参数，内核层面保证序列间隔离。Phase 3 当前使用朴素的 PyTorch SDPA，需要手动按序列切分以达到相同的隔离效果。Phase 4 引入 FlashAttention 后此问题将从框架层面彻底解决。
