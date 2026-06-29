# Qwen3.5-2B 适配测试文档

## 测试环境

| 项目 | 版本 / 配置 |
|------|------------|
| Python | 3.12.13 |
| PyTorch | 2.x（CPU，单元/集成测试）|
| pytest | 9.0.3 |
| 平台 | Linux (WSL2 x86_64) |
| GPU 测试 | 需设置 `NANO_VLLM_QWEN35_MODEL` 环境变量（当前跳过）|

运行全套 CPU 测试：
```bash
source /path/to/.venv/bin/activate
pytest tests/ -q
```

运行 GPU 集成测试（需真实 Qwen3.5-2B 权重）：
```bash
NANO_VLLM_QWEN35_MODEL=/path/to/Qwen3.5-2B pytest tests/test_qwen35.py -m gpu -v
```

---

## 测试用例清单

### tests/test_qwen35.py（本次新增，39 个用例）

| 测试类 | 测试方法 | 类型 | 测试点 | 结果 |
|--------|---------|------|--------|------|
| `TestQwen35RMSNorm` | `test_weight_initialized_to_zeros` | Unit | weight 初始化为零向量 | ✅ |
| | `test_zero_weight_equals_plain_rmsnorm` | Unit | weight=0 时退化为标准 RMSNorm | ✅ |
| | `test_nonzero_weight_scales_output` | Unit | weight=1 时输出为 2×rms_norm(x) | ✅ |
| `TestRMSNormGated` | `test_output_shape_preserved` | Unit | 输出 shape 与 x 相同 | ✅ |
| | `test_gate_scales_output` | Unit | silu(z)≈1 时输出≈rms_norm(x) | ✅ |
| | `test_zero_gate_produces_zero` | Unit | silu(z)→0 时输出趋近零 | ✅ |
| `TestRecurrentStep` | `test_output_shape` | Unit | out/new_state shape 正确 | ✅ |
| | `test_decay_with_zero_beta` | Unit | beta=0 时只衰减无写入 | ✅ |
| | `test_zero_state_with_unit_input` | Unit | 从零态单步后 state 值验证 | ✅ |
| | `test_gqa_expand` | Unit | nv>nk 时 k/g/beta 自动扩展 | ✅ |
| `TestGatedDeltaNet` | `test_allocate_states_shapes` | Unit | conv_state/recurrent_state shape 正确 | ✅ |
| | `test_allocate_states_dtype` | Unit | recurrent_state 强制 float32 | ✅ |
| | `test_slot_isolation` | Unit | 更新 slot 0 不污染 slot 1 | ✅ |
| | `test_prefill_output_shape` | Integration | prefill 输出 [T, hidden] | ✅ |
| | `test_decode_output_shape` | Integration | decode 输出 [BS, hidden] | ✅ |
| | `test_prefill_updates_recurrent_state` | Integration | prefill 后 recurrent_state 非零 | ✅ |
| | `test_decode_updates_conv_state` | Integration | decode 后 conv_state 已滚动 | ✅ |
| | `test_reset_to_zero_clears_state` | Unit | zero_() 清零模拟 warmup 后重置 | ✅ |
| `TestQwen35Attention` | `test_q_proj_output_width` | Unit | q_proj 输出 2×num_h×head_dim | ✅ |
| | `test_q_k_norm_zeros_init` | Unit | q_norm/k_norm weight 零初始化 | ✅ |
| | `test_partial_rope_rotary_dim` | Unit | rotary_dim = head_dim × factor | ✅ |
| | `test_forward_output_shape` | Integration | 全注意力层输出 [T, hidden] | ✅ |
| `TestQwen35Structure` | `test_model_has_correct_layer_count` | Unit | 模型层数与 layer_types 一致 | ✅ |
| | `test_layer_type_assignment` | Unit | linear/full 层类型正确分配 | ✅ |
| | `test_allocate_lin_attn_states_called` | Unit | conv/recurrent_state 正确分配 | ✅ |
| | `test_forward_output_shape` | Integration | 模型前向输出 [T, hidden] | ✅ |
| | `test_compute_logits_shape` | Integration | logits 输出 [T, vocab_size] | ✅ |
| `TestWeightPrefixStrip` | `test_class_attrs_defined` | Unit | 模型类属性存在 | ✅ |
| | `test_skip_prefixes_are_visual_and_mtp` | Unit | skip 前缀含 visual/mtp | ✅ |
| | `test_prefix_to_strip_is_language_model` | Unit | strip 前缀为 model.language_model. | ✅ |
| `TestSchedulerLinAttnSlots` | `test_slot_assigned_on_prefill` | Unit | prefill 分配 lin_attn_slot | ✅ |
| | `test_slot_freed_on_finish` | Unit | 序列完成后 slot 归还 | ✅ |
| | `test_no_slot_assigned_for_dense_model` | Unit | dense 模型 slot 始终为 -1 | ✅ |
| | `test_slot_pool_exhaustion_blocks_prefill` | Unit | slot 耗尽时停止 prefill | ✅ |
| GPU（跳过） | `TestGPUQwen35` × 6 | GPU | 真实权重端到端推理 | ⏭ skip |

### tests/test_rotary_embedding.py（本次追加，5 个用例）

| 测试类 | 测试方法 | 类型 | 测试点 | 结果 |
|--------|---------|------|--------|------|
| `TestPartialRoPE` | `test_rotary_dims_change_passthrough_unchanged` | Unit | 旋转段变化，pass-through 不变 | ✅ |
| | `test_partial_rope_output_shape_unchanged` | Unit | shape 与输入相同 | ✅ |
| | `test_full_rope_equals_original_behavior` | Unit | rotary_dim=head_size 等价旧行为 | ✅ |
| | `test_lru_cache_supports_multiple_rotary_configs` | Unit | maxsize=16 支持多套 RoPE 配置 | ✅ |
| | `test_cos_sin_cache_shape_partial` | Unit | cache shape 为 [max_pos,1,rotary_dim] | ✅ |

### tests/test_model_loader.py（本次追加，4 个用例）

| 测试类 | 测试方法 | 类型 | 测试点 | 结果 |
|--------|---------|------|--------|------|
| `TestLoaderPrefixStrip` | `test_prefix_stripped_on_load` | Unit | 前缀剥离后参数名正确匹配 | ✅ |
| | `test_skip_prefixes_not_loaded` | Unit | skip 前缀权重被忽略 | ✅ |
| | `test_no_prefix_strip_unchanged` | Unit | 无前缀配置时行为不变 | ✅ |
| | `test_prefix_strip_with_packed_mapping` | Unit | 前缀剥离后仍触发 packed_mapping | ✅ |

---

## 整体测试结果汇总

```
tests/ — 177 collected
  167 passed
   10 skipped (GPU tests, require NANO_VLLM_QWEN35_MODEL)
    0 failed

覆盖文件：
  nanovllm/models/qwen35.py
  nanovllm/layers/rotary_embedding.py  (partial RoPE)
  nanovllm/utils/loader.py            (prefix strip / skip)
  nanovllm/engine/scheduler.py        (lin_attn slot pool)
  nanovllm/utils/context.py           (lin_attn_seq_slots 字段)
  nanovllm/engine/sequence.py         (lin_attn_slot 字段)
  nanovllm/config.py                  (qwen3_5 model_type 处理)
```

---

## 验收标准对照

| 验收标准（来自 PRD）| 测试方法 | 实测值 | 是否达标 |
|-------------------|---------|--------|---------|
| Qwen3.5-2B hybrid 架构（18 GDN + 6 full_attention）正确分层 | `test_layer_type_assignment` | 4 层小模型按 layer_types 正确分类 | ✅ |
| GatedDeltaNet 前向 prefill/decode 输出 shape 正确 | `test_prefill_output_shape`, `test_decode_output_shape` | [T, H] / [BS, H] ✓ | ✅ |
| GDN slot 隔离：并发序列状态互不污染 | `test_slot_isolation` | slot 1 全零，slot 0 非零 | ✅ |
| lin_attn slot 池：prefill 分配、序列完成后归还 | `test_slot_assigned_on_prefill`, `test_slot_freed_on_finish` | slot 正确分配和归还 | ✅ |
| slot 耗尽时停止调度新 prefill | `test_slot_pool_exhaustion_blocks_prefill` | 第 3 个 seq 未被调度 | ✅ |
| dense 模型（Qwen3）不受 slot 机制影响 | `test_no_slot_assigned_for_dense_model`, 全套 Qwen3 测试 | 120 个 Qwen3 测试全绿 | ✅ |
| Partial RoPE（rotary_dim < head_size）：旋转段变化，pass-through 不变 | `test_rotary_dims_change_passthrough_unchanged` | 通过 | ✅ |
| 多 RoPE 配置并存（lru_cache maxsize=16）| `test_lru_cache_supports_multiple_rotary_configs` | 三种配置各自独立缓存 | ✅ |
| VLM 权重前缀剥离（model.language_model.）| `test_prefix_stripped_on_load` | 参数正确匹配 | ✅ |
| visual/mtp 权重跳过 | `test_skip_prefixes_not_loaded` | 仅 lm 参数被加载 | ✅ |
| Qwen35RMSNorm 零初始化，公式 (1+w)×norm(x) | `test_weight_initialized_to_zeros`, `test_zero_weight_equals_plain_rmsnorm` | 与手算结果一致 | ✅ |
| Qwen35Attention 输出门：q_proj 输出 2× 宽 | `test_q_proj_output_width` | weight.shape[0] = 2×num_h×head_dim | ✅ |

---

## 关键问题修复记录

### 1. Scheduler 回归（Task 5）
- **现象**：`num_lin_attn_slots=0`（dense 模型）时 prefill 全被阻塞
- **根因**：条件写为 `len(free_slots) == 0`，空 set 也满足
- **修复**：改为 `num_lin_attn_slots > 0 and len(free_slots) == 0`

### 2. 测试断言错误（Task 9）
- **现象**：`test_zero_state_with_unit_input` expected state 值写反
- **根因**：einsum `'vk,vd->vkd'` 索引顺序推算失误
- **修复**：修正期望值并补充推算注释

### 3. FakeQwen35Config 维度不一致（Task 9）
- **现象**：`RMSNormGated.forward` 报 shape mismatch (16 vs 64)
- **根因**：`nv*dv = 2*8 = 16 ≠ hidden_size = 64`；真实模型保证 `nv*dv == hidden`
- **修复**：将 `linear_value_head_dim` 从 8 改为 32，使 `nv*dv = 64 = hidden_size`

### 4. `_apply_conv_decode` token shape 错误（Task 9）
- **现象**：decode 路径 `cat(...).T.unsqueeze(0)` 在 1D 输入下产生 `[1, conv_dim]` 而非 `[1, conv_dim, 1]`
- **根因**：1D tensor 的 `.T` 是 no-op，缺少最后一维
- **修复**：改为 `.unsqueeze(0).unsqueeze(-1)`

---

## 已知局限

1. **GPU 测试未执行**：所有 `@pytest.mark.gpu` 测试（6 个）需要真实 Qwen3.5-2B 权重目录，当前环境未配置，已 skip。端到端推理正确性尚未通过 GPU 验证。

2. **prefill conv1d 因果性**：当前使用 `nn.Conv1d(padding=kernel-1)` 并截取前 T 帧，实现语义正确（已分析验证），但与 vllm 参考实现采用显式左侧 padding 的写法不同，存在微小数值差异风险。

3. **多进程 / TP 未测试**：nano-vllm 为单进程架构，`ColumnParallelLinear` / `RowParallelLinear` 在 TP=1 下等价于普通线性层，多 GPU Tensor Parallel 场景未覆盖。

4. **长序列性能**：prefill 采用逐 token 循环的顺序递推，时间复杂度 O(T)；对比 vllm 使用 chunk scan（带并行化），长序列时吞吐差距显著。此为 Phase 8 性能测试的主要关注点。
