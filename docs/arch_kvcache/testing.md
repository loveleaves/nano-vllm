# KV Cache 管理对齐 — 测试设计

运行：`pytest tests/test_block_pool.py tests/test_kv_cache_spec.py tests/test_block_manager.py`

## 测试矩阵

| 文件 | 覆盖点 | 依赖 |
|---|---|---|
| `test_block_pool.py` | 初始状态；get_new_block + deref；ref>1 时 deref 不释放；register_hash/cached_block_id 往返；reuse_cached_block（ref++ / 从空闲取出）；重分配淘汰旧 hash；FIFO 复用顺序 | CPU |
| `test_kv_cache_spec.py` | page_size_bytes；kv_cache_shape；num_blocks_for_memory 往返 + 向下取整；dtype 影响块大小 | CPU（仅用 torch.dtype.itemsize） |
| `test_block_manager.py`（保留） | 原 BlockManager 全部用例（基础分配 + 前缀缓存哈希），验证兼容别名/委派属性行为不变 | CPU |

## 回归

- 全量套件：**202 passed, 4 skipped**（skip 为需权重的 loader 用例），较对齐前 +11 用例。
- `test_scheduler`（用 `block_manager`）与 GPU `test_qwen3`（走 `ModelRunner.allocate_kv_cache`，
  实际分配 KV cache 张量）均绿，证明池/管理器拆分与 Spec 重构端到端等价。
- `example.py` / `bench.py` 无需改动。
