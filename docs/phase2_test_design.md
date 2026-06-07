# Phase 2 测试设计文档

## 1. 测试范围

Phase 2 实现神经网络层，测试分两类：
- **CPU 单元测试**（CI 可运行）：验证张量形状、数学正确性，无 GPU 要求
- **GPU 集成测试**（标记为 gpu，CI 跳过）：完整 forward pass 验证

涵盖模块：
- `nanovllm/utils/context.py`
- `nanovllm/layers/layernorm.py`
- `nanovllm/layers/activation.py`
- `nanovllm/layers/rotary_embedding.py`
- `nanovllm/layers/linear.py`
- `nanovllm/layers/embed_head.py`
- `nanovllm/layers/sampler.py`
- `nanovllm/layers/attention.py`
- `nanovllm/models/qwen3.py`

## 2. 测试策略

### CPU 可运行测试（@pytest.mark.unit）

| 测试目标 | 验证方式 |
|----------|----------|
| Context 设置/获取/重置 | 字段值对比 |
| RMSNorm 归一化正确性 | 手算参考值 |
| Fused Add-RMSNorm | 等价于 x+residual 后 rms_norm |
| SiluAndMul 形状与数值 | 已知输入验证输出 |
| RoPE 旋转等价性 | 正交旋转保长度 |
| Linear weight_loader | 各 shard_id 写入正确位置 |
| Sampler 形状 | 输出范围在 [0, vocab_size) |
| Attention 输出形状 | prefill 输出维度 |
| Qwen3ForCausalLM forward 形状 | 随机权重 + 随机输入 |

### GPU 测试（@pytest.mark.gpu）

| 测试目标 | 验证方式 |
|----------|----------|
| CUDA 设备上的完整 forward | logits 形状 [batch, vocab_size] |
| RoPE CUDA 精度 | CPU/CUDA 结果差异 < 1e-5 |

## 3. 关键测试用例

### 3.1 RMSNorm
- 白噪声输入归一化后方差约为 1（weight=1 时）
- add_rms_forward 与 rms_forward(x + residual) 数值等价
- 返回 dtype 与输入 dtype 一致

### 3.2 SiluAndMul
- 输出维度是输入的一半（chunk 操作）
- SiLU(0) * 0 = 0，SiLU 正数单调性验证

### 3.3 RotaryEmbedding
- 旋转前后向量 L2 范数不变（旋转是正交变换）
- 不同位置的旋转矩阵不同（位置区分性）
- lru_cache：相同参数返回同一实例

### 3.4 Linear weight_loader
- QKVParallelLinear: q/k/v shard 写入位置不重叠
- MergedColumnParallelLinear: shard 0/1 拼接后等于完整权重

### 3.5 Attention（prefill）
- 输出形状 [N, num_heads, head_dim]
- GQA 扩展：num_kv_heads < num_heads 时正确处理

### 3.6 Qwen3ForCausalLM
- forward 输出形状 [N, hidden_size]
- compute_logits 在 prefill 上下文中只输出最后一个 token 的 logits

## 4. 测试通过标准

- 所有 `@pytest.mark.unit` 用例在 CPU 环境通过
- 可在无 GPU 的 CI 中运行（torch CPU 安装）
- GPU 测试标记跳过但在有 GPU 环境手动验证
