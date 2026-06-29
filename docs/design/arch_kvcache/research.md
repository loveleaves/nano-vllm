# KV Cache 管理对齐 — V1 现状调研

> 对照基准：本机 `/home/cb/work/vllm/vllm` @ tag `v0.15.1`（V1 架构）。

## 背景：什么是分页 KV cache（PagedAttention）

**问题**：自回归生成时，每个序列已生成 token 的 Key/Value 要缓存复用（KV cache），否则每步都要
重算全部历史注意力。但序列长度事先未知且差异巨大，若按"最大长度"为每序列预留**连续**显存，会
浪费大量显存（内部碎片），并发数被严重压低。

**核心思想（借鉴操作系统虚拟内存分页）**：把 KV cache 切成固定大小的**物理块**（block，如
256 token/块），序列的逻辑 token 位置经一张**块表（block table）**映射到任意物理块——逻辑连续、
物理离散。按块分配、无需连续大块；显存碎片几乎消除，并发数大幅提升。

**作用 / 收益**：①显存利用率高（碎片仅末块）；②**前缀缓存**——相同前缀的不同请求共享物理块
（块内容哈希 + 引用计数），跳过重复 prefill；③**抢占**——显存紧张时换出/重算某些序列的块。

**三层结构（本轮对齐 V1）**：
```
BlockPool        物理块池 + 空闲队列 + 块哈希表 + 引用计数（不感知 Sequence）
KVCacheManager   每请求 can_allocate/allocate/append/前缀命中（建于 BlockPool 之上）
KVCacheSpec      "单块字节数"与"显存→块数"换算（FullAttentionSpec）
```

## V1 KV cache 组件

| 文件 | 职责 | nano 对应 |
|---|---|---|
| `v1/core/block_pool.py` | `BlockPool`：物理块池、空闲队列、前缀缓存哈希表、引用计数、缓存块淘汰；`KVCacheBlock` | `kv_cache/block_pool.py`（同名同构子集） |
| `v1/core/kv_cache_manager.py` | `KVCacheManager`：每请求块编排（get_computed_blocks / allocate_slots / free / cache_blocks），对外是调度器使用的入口 | `kv_cache/kv_cache_manager.py`（方法名 nano 化：can_allocate/allocate/...） |
| `v1/core/single_type_kv_cache_manager.py` | 单一类型层的管理器（full / sliding / mamba 各一种） | 合并进 KVCacheManager（nano 只有 full） |
| `v1/core/kv_cache_coordinator.py` | 协调多组管理器（Unitary 单组 / Hybrid 异构多组） | ❌ 不引入（nano 单组同构） |
| `v1/kv_cache_interface.py` | `KVCacheSpec` 及子类 FullAttention / SlidingWindow / Mamba / MLA / Encoder...；`page_size_bytes` / `max_memory_usage_bytes` | `kv_cache/interface.py`（仅 KVCacheSpec + FullAttentionSpec） |
| `v1/core/kv_cache_utils.py` | 块哈希、spec 推导、内存规划工具 | 内联（compute_hash 在 BlockPool） |

## 关键观察

1. **池 / 管理器分层**：V1 把"物理块原语 + 前缀缓存哈希表"（BlockPool）与"每请求块编排"
   （KVCacheManager）分开。nano 原 `BlockManager` 把两者揉在一个类里——这是本轮主要拆分点。
2. **KVCacheSpec 收敛内存计算**：每层 KV cache 的块字节数 / 显存→块数 由 Spec 描述，
   而非散落在 ModelRunner。nano 原 `allocate_kv_cache` 内联算 `block_bytes` 与 `num_blocks`。
3. **异构是 coordinator 的存在理由**：BlockPool/KVCacheManager 本身不关心层类型；只有当一个
   模型混用 full + sliding window + Mamba 等不同 KV 布局时，才需要 coordinator 编排多组。
   nano 全是 full-attention 同构层，单组即可，故不引入 coordinator / 多 Spec。
4. **前缀缓存机制**：链式块哈希（含前块 hash）+ hash→block 表 + 释放保留 hash（延迟淘汰），
   nano 已具备，本轮仅把它归位到 BlockPool。
