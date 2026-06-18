# nano-vllm KV Cache 管理对齐 V1 — 详细设计文档

> 基于 `research.md`。目标：把单类 `engine/block_manager.py::BlockManager` 拆为 V1 风格的
> `engine/kv_cache/` 子包——**BlockPool（块级原语 + 前缀缓存）/ KVCacheManager（每请求编排）/
> KVCacheSpec（块字节·块数规格）**，并让 ModelRunner 用 Spec 计算 KV cache 张量。

## Motivation

对齐前 `BlockManager` 一个类里揉了三件事：物理块池与空闲队列、前缀缓存哈希表与引用计数、
每请求的分配/释放/追加编排；显存→块数的计算又散落在 `ModelRunner.allocate_kv_cache`。
与 V1 的差距在"形"：缺池/管理器分层、缺 KVCacheSpec。本轮按 V1 归位，前缀缓存算法与
分页策略本身不变。

## 范围决策（与 V1 的取舍）

| V1 特性 | 是否对齐 | 说明 |
|---|---|---|
| `BlockPool`（块原语 + 前缀缓存哈希表） | ✅ | 从 BlockManager 拆出 |
| `KVCacheManager`（每请求编排，建于 BlockPool） | ✅ | 即原 BlockManager 的序列级逻辑 |
| `KVCacheSpec` / `FullAttentionSpec` | ✅ | 封装 page_size_bytes / 显存→块数 / 形状 |
| ModelRunner 用 Spec 算块数与张量 | ✅ | 行为完全等价于旧内联计算 |
| `KVCacheCoordinator`（多组/异构） | ❌ | nano 单组同构 full-attention，单 BlockPool 足够 |
| SlidingWindow / Mamba / MLA / Encoder Spec、KV offload、KV connector | ❌ | 无对应模型层/功能 |

## Architecture

### 包结构

```
nanovllm/engine/kv_cache/
├── __init__.py            # 导出 BlockPool/KVCacheBlock/Block/KVCacheManager/BlockManager/KVCacheSpec/FullAttentionSpec
├── interface.py           # KVCacheSpec(ABC) + FullAttentionSpec（page_size_bytes / kv_cache_shape / num_blocks_for_memory）
├── block_pool.py          # KVCacheBlock(=旧 Block) + BlockPool（块级原语 + 哈希表 + 引用计数）
└── kv_cache_manager.py    # KVCacheManager（每请求编排）; BlockManager = KVCacheManager 别名
nanovllm/engine/block_manager.py   # 向后兼容垫片：re-export BlockManager/KVCacheManager/BlockPool/Block
```

### 职责切分

```
BlockPool（不感知 Sequence）
  blocks / free_block_ids / used_block_ids / hash_to_block_id
  compute_hash(static) · get_num_free_blocks · cached_block_id · is_used
  get_new_block · reuse_cached_block · deref_block · register_hash

KVCacheManager（每请求，持有 1 个 BlockPool）
  can_allocate(seq)->int   # 前缀命中探测 + 空闲块校验
  allocate(seq, num_cached) # 复用命中块 + 新分配剩余块
  deallocate(seq)          # 逐块 deref（逆序）
  can_append/may_append(seq)  # decode 追加块
  hash_blocks(seq)         # 注册本步新满块哈希
  # 兼容：blocks/free_block_ids/used_block_ids/hash_to_block_id 属性委派 BlockPool；compute_hash 类方法
```

### 数据流

```
ModelRunner.allocate_kv_cache:
  spec = FullAttentionSpec(block_size, num_kv_heads, head_dim, dtype)
  num_blocks = spec.num_blocks_for_memory(available_bytes, num_layers)   # 替换旧内联 block_bytes 计算
  kv_cache = torch.empty(2, num_layers, *spec.kv_cache_shape(num_blocks)[1:])

Scheduler（不变）→ block_manager.can_allocate/allocate/deallocate/can_append/may_append/hash_blocks
                  （BlockManager 现为 KVCacheManager，方法签名一致）
```

### 关键设计点

- **行为等价**：`FullAttentionSpec.page_size_bytes = 2*block_size*num_kv_heads*head_dim*itemsize`，
  `num_blocks = available // (page_size_bytes * num_layers)`，与旧 `block_bytes` 公式逐位一致；
  全局张量形状 `(2, num_layers, num_blocks, block_size, num_kv_heads, head_dim)` 不变。
- **向后兼容**：`BlockManager` 为 `KVCacheManager` 别名；旧属性（blocks/free_block_ids/
  used_block_ids/hash_to_block_id）与 `compute_hash` 经委派保留，Scheduler 与既有
  `test_block_manager` 零改动通过。
- **延迟淘汰**：释放块时 hash 暂留（`deref_block` 仅归还空闲队列），下次 `get_new_block`
  取到该块才删旧 hash，保证前缀缓存最大化命中——逻辑从旧 BlockManager 原样迁入。
