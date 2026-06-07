# Phase 1 测试设计文档

## 1. 测试范围

Phase 1 实现纯 Python 数据结构，无 GPU 依赖，全部用例均可在 CI 环境运行。

涵盖模块：
- `nanovllm/config.py`
- `nanovllm/sampling_params.py`
- `nanovllm/engine/sequence.py`
- `nanovllm/engine/block_manager.py`
- `nanovllm/engine/scheduler.py`

## 2. 测试策略

| 测试类型 | 策略 | 标记 |
|----------|------|------|
| 单元测试 | 每个类的核心方法独立验证 | `@pytest.mark.unit` |
| 集成测试 | 模拟完整调度流程（prefill + decode + finish） | `@pytest.mark.unit` |
| 边界条件 | 空队列、内存不足、单 token seq、多 seq 并发 | `@pytest.mark.unit` |

## 3. 测试用例说明

### 3.1 Config 测试
- 正常初始化（不加载 hf_config，仅验证字段约束）
- 非法路径应抛出 AssertionError
- kvcache_block_size 非 256 倍数应抛出 AssertionError

### 3.2 SamplingParams 测试
- temperature=0 应抛出 AssertionError
- 默认参数构造正常

### 3.3 Sequence 测试
- block(i) 返回正确的 token 子序列
- num_blocks 向上取整正确
- last_block_num_tokens 在各边界位置正确
- append_token 更新 num_tokens / last_token / token_ids
- __getstate__ / __setstate__ pickle 往返一致

### 3.4 BlockManager 测试
- 初始状态：free_block_ids 包含所有 block_id，used_block_ids 为空
- allocate：seq.block_table 长度等于 num_blocks，空闲块数减少
- deallocate：block_table 清空，空闲块数恢复
- can_allocate：空闲块不足时返回 -1
- can_append：在 block_size 整数倍位置处需要新块
- may_append：追加后 block_table 长度增加

### 3.5 Scheduler 测试
- 单 seq prefill：调度后 seq 进入 running，is_prefill=True
- 多 seq prefill：batch 内 token 数不超过 max_num_batched_tokens
- prefill → decode 切换：waiting 空后调度 decode
- decode 终止：token == eos 后 seq 进入 FINISHED，block 释放
- 内存不足：can_allocate 返回 -1 时停止 prefill 调度
- 完整流程：从 add 到 is_finished() == True

## 4. 测试通过标准

- 所有 `@pytest.mark.unit` 用例通过
- 无 GPU 依赖（纯 Python + 标准库）
- CI 环境（pytest --timeout=30）全部通过
