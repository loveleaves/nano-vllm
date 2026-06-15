# Prefix Caching 设计与实现对比调研报告：nano-vllm vs 开源 vLLM

**调研日期**：2026-06-14
**调研主体**：当前工程 nano-vllm（phase4 分支）与开源 vLLM（v1 引擎）的 Prefix Caching 设计与实现
**调研意图**（技术向）：拆解当前工程的前缀缓存实现，与生产级 vLLM v1 的设计逐层对比，识别简化取舍、正确性边界与可演进方向，服务于工程优化决策。
**代码基准**：
- nano-vllm：`nanovllm/engine/{block_manager,scheduler,sequence,model_runner}.py`、`nanovllm/layers/attention.py`（commit `f8d495d`）
- vLLM：`vllm/v1/core/{block_pool,kv_cache_utils,kv_cache_manager,single_type_kv_cache_manager}.py`（本地 checkout `/home/cb/work/vllm/vllm`）

---

## 一、执行摘要

1. **核心机制同源**：两者都采用「分页 KV cache + 块级链式哈希（chained hash）+ 哈希表查找 + 引用计数共享」这一套范式。nano-vllm 的 `BlockManager` 实质是 vLLM `BlockPool` 的最小可用子集，连「链入父块哈希以区分相同 token 不同前缀」的关键设计都一致。

2. **最大的实现差异在淘汰策略**：nano-vllm 用 `deque` + 「释放时追加到队尾、分配时从队头取」近似 FIFO/LRU，淘汰发生在 `_allocate_block` 取到一个仍带 hash 的块时**惰性删除**；vLLM 用**自实现的 O(1) 双向链表** `FreeKVCacheBlockQueue`，支持从队列中间 O(1) 摘除（`touch` 命中复用），并通过「释放时逆序入队」精确实现 LRU + 同请求内长块优先淘汰。

3. **粒度与对齐不同**：nano-vllm 只在「整块满」时缓存，命中也只能按 `block_size` 对齐（默认 **256**，且强制为 256 的倍数）；vLLM 的 `block_size` 默认 16，并通过 `max_cache_hit_length = num_tokens - 1` 精心处理「全命中时必须重算最后一个 token 以产出 logits」的边界。nano-vllm 的 `can_allocate` 同样隐含地跳过最后一块（`range(num_blocks - 1)`），方向正确但粒度粗。

4. **正确性已落地但功能裁剪**：nano-vllm 前缀命中后，prefill 阶段通过 `cu_seqlens_k > cu_seqlens_q` 触发 `flash_attn_varlen_func(block_table=...)` 从 KV cache 读历史前缀，逻辑闭环、与 chunked prefill 协同。但它**不支持** vLLM 的诸多生产特性：多模态/LoRA/cache_salt 的 extra keys、sliding window/Mamba 混合 KV group、P/D 分离的外部缓存、KV cache events、prefix cache 指标统计、`reset_prefix_cache`。

5. **哈希函数取舍**：nano-vllm 固定用 `xxhash.xxh64`（64-bit 整数，极快但有碰撞风险，故额外存 `token_ids` 做完整性校验）；vLLM 默认 `sha256_cbor`/可选 `xxhash_cbor`，哈希入 key 的是 `(parent_hash, token_tuple, extra_keys)` 三元组，并用 `NONE_HASH`（进程随机种子）防跨进程哈希注入攻击。

**核心结论**：nano-vllm 的前缀缓存是一份**教学级、正确闭环的精简实现**，抓住了 vLLM 的全部核心 idea（链式哈希 + 引用计数 + 惰性淘汰），代价是放弃了 O(1) 中间摘除、细粒度块、多场景 extra keys 和安全哈希。对单机单模型推理足够，但在高并发淘汰频繁、小块复用、多租户安全等场景下与 vLLM 有量级差距。

---

## 二、背景与概述

### 2.1 什么是 Prefix Caching

Prefix Caching（前缀缓存）解决的是：多个请求共享相同前缀（system prompt、few-shot 示例、多轮对话历史）时，避免重复计算这些前缀的 KV cache。其前提是分页 KV cache（PagedAttention）——KV 被切成定长物理块，块可被多个序列以引用计数方式共享。

核心三要素：
1. **块级哈希**：给每个「满块」算一个能唯一标识「从序列开头到本块」内容的哈希。
2. **链式哈希（chained / prefix hash）**：块哈希必须**链入前一块的哈希**，否则同样 token 内容在不同前缀下会误命中（例如块 B 接在前缀 A 后 vs 接在前缀 C 后，KV 完全不同）。
3. **哈希表 + 引用计数**：`hash → block` 查找表实现命中；命中块 `ref_count++` 实现共享，归零才可回收。

### 2.2 两个工程的定位

| 维度 | nano-vllm | vLLM v1 |
|---|---|---|
| 定位 | 教学/最小实现 | 生产级推理引擎 |
| 前缀缓存代码量 | 单文件 ~184 行 `block_manager.py` | 跨 4+ 文件、数千行（block_pool/utils/manager/coordinator） |
| 是否可关闭 | **不可**（始终启用） | `enable_prefix_caching` 开关 |
| 默认 block_size | 256（强制 256 倍数） | 16 |

---

## 三、nano-vllm 的设计与实现

### 3.1 数据结构（`block_manager.py:8-62`）

```
Block:
  block_id   物理块编号（= blocks 下标）
  ref_count  引用计数，0 可回收
  hash       链式 xxhash，-1 表示未哈希
  token_ids  该块 token（用于碰撞校验）

BlockManager:
  blocks            list[Block]，下标即 block_id
  hash_to_block_id  dict[int, int]，前缀缓存核心查找表
  free_block_ids    deque[int]，FIFO 空闲队列（左取右还）
  used_block_ids    set[int]，在用块
```

### 3.2 链式哈希（`block_manager.py:64-75`）

```python
@classmethod
def compute_hash(cls, token_ids, prefix=-1):
    h = xxhash.xxh64()
    if prefix != -1:
        h.update(prefix.to_bytes(8, "little"))   # 链入父块哈希
    h.update(np.array(token_ids, dtype=np.int64).tobytes())
    return h.intdigest()
```

- 用 `prefix`（上一块 hash）参与计算，实现链式。
- 返回 64-bit 整数。**因 64-bit 有碰撞概率，命中时额外比对 `block.token_ids == token_ids`**（`can_allocate:114`）防脏命中。

### 3.3 命中探测 `can_allocate`（`block_manager.py:94-121`）

调度器在 prefill 前调用，**一次遍历同时完成两件事**：判断内存是否够 + 探测命中块数。

```python
for i in range(seq.num_blocks - 1):          # 跳过最后一块（可能不满）
    token_ids = seq.block(i)
    h = self.compute_hash(token_ids, h)
    block_id = self.hash_to_block_id.get(h, -1)
    if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
        break                                # 链断了，后续不可能命中
    num_cached_blocks += 1
    if block_id in self.used_block_ids:
        num_new_blocks -= 1                  # 命中且在用→共享，无需新分配
```

返回 `-1`（内存不足）或 `num_cached_blocks`。注意：命中块若在 `free`（未被任何 seq 引用）也算命中，但 `num_new_blocks` 不减——因为它仍占一个空闲名额（后面 `allocate` 会从 free 里捞回来）。

### 3.4 正式分配 `allocate`（`block_manager.py:123-146`）

```python
for i in range(num_cached_blocks):
    block_id = self.hash_to_block_id[h]
    if block_id in used_block_ids:           # 已在用 → 仅 ref_count++
        block.ref_count += 1
    else:                                     # 在 free 队列 → 捞出复用
        block.ref_count = 1
        self.free_block_ids.remove(block_id)  # ⚠️ deque.remove 是 O(n)
        self.used_block_ids.add(block_id)
    seq.block_table.append(block_id)
for _ in range(num_cached_blocks, seq.num_blocks):
    seq.block_table.append(self._allocate_block())   # 其余新分配
seq.num_cached_tokens = num_cached_blocks * block_size   # 跳过这些 token 的 prefill
```

### 3.5 惰性淘汰（`block_manager.py:77-92`）

- **释放**：`deallocate` 逆序遍历 block_table，`ref_count--`，归零者 `_deallocate_block` 把 block_id **append 到 free 队尾**（hash 保留，块仍可被后续前缀命中）。
- **淘汰**：`_allocate_block` 从队头 `popleft` 取块，若取到的块 `hash != -1` 且仍在哈希表中，**此刻才 `del hash_to_block_id[hash]`**（惰性淘汰），再 `reset`。

这套「释放保留 hash + 分配时才删」让块在被真正复用前一直可命中，是个巧妙的低成本 LRU 近似。

### 3.6 哈希注册 `hash_blocks`（`block_manager.py:167-183`）

每步推理后 `scheduler.postprocess` 调用，对「本步新填满的块」算哈希并注册：

```python
start = num_cached_tokens // block_size
end   = (num_cached_tokens + num_scheduled_tokens) // block_size
h = blocks[block_table[start-1]].hash if start > 0 else -1   # 接上父块哈希
for i in range(start, end):
    h = compute_hash(seq.block(i), h)
    block.update(h, token_ids)
    hash_to_block_id[h] = block.block_id
```

与 chunked prefill 天然协同——只对本 chunk 跨过的满块注册。

### 3.7 正确性闭环：attention 读历史前缀（`attention.py:134-150`、`model_runner.py:202-203`）

命中后 prefill 阶段，`prepare_prefill` 检测 `cu_seqlens_k[-1] > cu_seqlens_q[-1]`（K 比 Q 长 = 有历史前缀未在本步重算），就传入 `block_tables`；attention forward 据此从 `k_cache/v_cache` 用 `flash_attn_varlen_func(block_table=...)` 读历史 KV，causal 注意力自然覆盖「已缓存前缀 + 本步新 token」。逻辑闭环。

---

## 四、vLLM v1 的设计与实现

### 4.1 分层架构

vLLM 把职责拆得很细：

```
KVCacheManager        对外接口：get_computed_blocks / allocate_slots / free / cache_blocks
  └ KVCacheCoordinator   协调多个 KV cache group（full attn / sliding window / mamba…）
      └ SingleTypeKVCacheManager   每类 attention 一个：find_longest_cache_hit 各异
          └ BlockPool               物理块池：分配/释放/缓存/淘汰
              ├ FreeKVCacheBlockQueue   O(1) 双向链表空闲队列（LRU）
              └ BlockHashToBlockMap     hash → block(s) 查找表
Request.block_hashes  请求自带的链式块哈希列表（创建/append 时增量计算）
```

### 4.2 数据结构差异

**`KVCacheBlock`**（`kv_cache_utils.py:117`）多了 `prev_free_block / next_free_block`（侵入式双向链表指针）、`is_null`（占位空块）。

**`BlockHashToBlockMap`**（`block_pool.py:34`）：`hash → 单个 block` 或 `hash → {block_id: block}` 的联合类型——**允许同一 hash 对应多个物理块**（不去重，保证 block table append-only），用 union 类型而非总用 dict 来降低 GC 开销。nano-vllm 是严格 `hash → 单 block_id`，后写覆盖先写。

**`FreeKVCacheBlockQueue`**（`kv_cache_utils.py:165`）：核心差异点。自实现双向链表 + fake head/tail 哨兵，支持：
- `popleft` O(1) 取 LRU；
- **`remove(block)` O(1) 从中间摘除**——这是 nano-vllm 用 `deque.remove`（O(n)）做不到的，命中复用一个 free 块时必须用到。

### 4.3 链式哈希（`kv_cache_utils.py:563-590`）

```python
def hash_block_tokens(hash_function, parent_block_hash, curr_block_token_ids, extra_keys=None):
    if not parent_block_hash:
        parent_block_hash = NONE_HASH          # 进程随机种子，防哈希注入
    return BlockHash(hash_function(
        (parent_block_hash, tuple(curr_block_token_ids), extra_keys)))
```

四点比 nano-vllm 强：
1. **`extra_keys`**：把 MM hash / LoRA id / cache_salt 拼进 key（`generate_block_hash_extra_keys`），实现多模态、多适配器、租户隔离的正确缓存。
2. **`NONE_HASH`**：用 `os.urandom` 或带 `PYTHONHASHSEED` 的种子做根哈希，**防止恶意构造前缀注入污染缓存**。
3. **可插拔哈希**：`sha256_cbor`（默认，抗碰撞）/ `xxhash_cbor`（快）。
4. **哈希在 Request 侧增量计算**（`get_request_block_hasher`），prefill/decode append token 时只算新满块，结果存 `request.block_hashes`，BlockPool 直接复用，避免重复哈希。

### 4.4 命中查找 `find_longest_cache_hit`（`single_type_kv_cache_manager.py:523-569`）

```python
max_num_blocks = max_length // block_size
for block_hash in itertools.islice(block_hashes, max_num_blocks):
    if cached_block := block_pool.get_cached_block(block_hash, kv_cache_group_ids):
        ...append...
    else:
        break                       # 链断即停（与 nano-vllm 同思路）
# 还要处理 eagle drop、alignment 对齐回退
```

关键边界（`kv_cache_manager.py:215-221`）：`max_cache_hit_length = request.num_tokens - 1`——**即使全部 token 都命中，也要保留最后一个 token 重算以产出 logits**。nano-vllm 用「`range(num_blocks - 1)` 跳过最后一块」达成相同效果，但因 block 粒度粗（256），最坏会多重算近 256 个 token；vLLM block_size=16 则浪费小得多。

各 attention 类型有不同 `find_longest_cache_hit`：FullAttention 线性扫；SlidingWindow 只需窗口内连续块命中；ChunkedLocal/Mamba 各有对齐逻辑。nano-vllm 只有 full attention 一种。

### 4.5 淘汰与 LRU（`block_pool.py:333-441`）

- **分配** `get_new_blocks`：`popleft_n` 取队头，对每块 `_maybe_evict_cached_block`（若带 hash 则从查找表删除并 `reset_hash`）——与 nano-vllm 惰性淘汰**思路一致**。
- **命中复用** `touch`：命中的块若 `ref_cnt==0`（在 free 队列）则 `free_block_queue.remove(block)` O(1) 摘除再 `ref_cnt++`。
- **释放** `free_blocks`：`ref_cnt--`，归零的块按**调用方给定的逆序**入队（`append_n`），实现「同请求内尾块（hash 链更长）先被淘汰」的精细 LRU。nano-vllm 也逆序释放（`deallocate` 用 `reversed`），同思路但队列是 deque。

### 4.6 生产特性（nano-vllm 全部缺失）

- **KV cache events**（`BlockStored/BlockRemoved/AllBlocksCleared`）：用于 P/D 分离、前缀缓存可观测。
- **prefix cache 指标**（`prefix_cache_stats.record`，命中率统计）。
- **`reset_prefix_cache`**：RLHF 权重更新后失效缓存。
- **外部缓存**（connector / `num_external_computed_tokens`）：P/D 分离从远端拉 KV。
- **多 KV cache group 协调**：混合 full + sliding window + mamba。
- **`null_block`**：sliding window / mamba 的占位块。

---

## 五、横向对比与关键判断

### 5.1 核心机制对照表

| 维度 | nano-vllm | vLLM v1 | 判断 |
|---|---|---|---|
| 范式 | 分页+链式哈希+引用计数 | 同 | ✅ 同源 |
| 链式哈希 | xxh64(prefix‖tokens) | hash(parent, tokens, extra_keys) | vLLM 更全 |
| 哈希函数 | xxh64（64-bit） | sha256_cbor/xxhash_cbor | vLLM 抗碰撞+防注入 |
| 碰撞处理 | 存 token_ids 比对 | sha256 概率可忽略；同存 token | 思路一致 |
| 查找表 | hash→单 block_id | hash→block 或 {id:block} | vLLM 不去重、append-only |
| 空闲队列 | `deque`（O(n) 中间删） | 自实现双向链表（O(1) 中间删） | **vLLM 显著优** |
| 淘汰 | 惰性（分配时删 hash） | 惰性（`_maybe_evict`）+ 精细 LRU | 思路一致，vLLM 更精细 |
| 命中复用 free 块 | `deque.remove` O(n) | `touch`+`remove` O(1) | **vLLM 显著优** |
| block_size | 256（强制倍数） | 16 默认 | vLLM 细粒度、命中率高 |
| 全命中边界 | 跳过最后一块 | `num_tokens-1` | 同思路，vLLM 浪费更小 |
| extra keys | ❌ | MM/LoRA/salt | vLLM 独有 |
| 多 attention 类型 | 仅 full | full/SWA/chunked/mamba | vLLM 独有 |
| P/D 外部缓存 | ❌ | connector | vLLM 独有 |
| 可观测/指标/reset | ❌ | events+stats+reset | vLLM 独有 |
| 正确性闭环 | ✅ varlen+block_table | ✅ | 都正确 |

### 5.2 关键判断

**判断一：nano-vllm 抓住了 prefix caching 的全部「第一性原理」。** 链式哈希防误命中、引用计数共享、惰性淘汰保留缓存——这三个最容易做错的点都做对了，且与 chunked prefill / 抢占协同（preempt 时保留 hash，复算大概率再命中，`scheduler.py:119-127`）。作为教学实现是高质量的。

**判断二：最痛的性能短板是空闲队列的数据结构。** nano-vllm 在 `allocate`（`block_manager.py:141`）命中 free 块时用 `deque.remove(block_id)`，这是 **O(n)** 操作（n = 空闲块数，可达数万）。高命中率 + 大 KV pool 场景下，每次复用都线性扫描空闲队列，是真实的吞吐瓶颈。vLLM 为此专门自实现 O(1) 双向链表，注释明确点出「Python deque 无法 O(1) 中间删除」。**这是最值得移植的优化。**

**判断三：block_size=256 是命中率与碎片的双输选择。** 前缀必须 256 对齐才能命中——两个请求前缀差 1 个 token，或前缀长度不是 256 整数倍，命中块数大幅下降；且全命中时最坏重算 255 个 token。vLLM 默认 16 命中粒度细 16 倍。nano-vllm 强制 256 倍数（`config.py:37`）应是为了 CUDA kernel/对齐简化，但前缀缓存命中率为此付出大代价。

**判断四：安全与多场景是「能力」而非「优化」差距。** 缺 extra_keys 意味着**多模态、LoRA、多租户场景下前缀缓存会出错或必须关闭**（不同图片/适配器的相同 token 前缀会误命中）；缺 NONE_HASH 随机种子意味着理论上可被构造前缀注入。单模型纯文本场景无碍，但这是功能边界而非性能调优。

---

## 六、改进建议（按性价比排序）

1. **【高性价比】替换空闲队列为 O(1) 双向链表**。直接移植 vLLM `FreeKVCacheBlockQueue` 思路，消除 `allocate` 中 `deque.remove` 的 O(n)。改动局部，收益直接，对高命中率工作负载吞吐提升明显。

2. **【高性价比】下调 block_size 默认值并解除 256 倍数强制**。若 kernel 不强依赖 256 对齐，降到 16/32 可大幅提升命中率、降低尾块浪费。需先确认 `store_kvcache` / flash-attn 对块大小的约束。

3. **【中】增加 prefix cache 命中率指标**。`can_allocate` 已算出 `num_cached_blocks`，顺手累加 hit/total，暴露命中率便于调参与验证，成本极低。

4. **【中】若要支持多模态/LoRA**，必须引入 extra_keys 进哈希，否则前缀缓存会产生错误结果——这是正确性前提，不是可选项。

5. **【低/按需】哈希安全**：若面向多租户/不可信输入，把根哈希换成进程随机种子（NONE_HASH 思路），或换 sha256。单机可信场景可不动。

6. **【可选】查找表支持 hash→多 block**。当前后写覆盖先写，极端并发下可能让先写者的缓存提前不可命中；vLLM 的 union 类型解决此问题，但实现复杂度上升，单流场景非必需。

---

## 七、调研局限性与待补充方向

- **未做性能实测**：O(n) vs O(1) 队列、block_size 256 vs 16 的命中率/吞吐差距均为代码层面推断，未在 nano-vllm 上跑 benchmark 量化。建议构造「长公共前缀 + 高并发」负载实测 `deque.remove` 占比与命中率。
- **vLLM 侧只读了 full attention 主路径**，sliding window / mamba / P-D connector 的 `find_longest_cache_hit` 细节、`allocate_slots` 的三阶段分配只做了概览，未逐行核对。
- **未核对 nano-vllm 在 TP（张量并行）下前缀缓存的正确性**——多 worker 各自持有 BlockManager 还是共享，block_table 同步逻辑未在本次调研覆盖，建议后续专项确认。
- **CUDA graph 与前缀缓存的交互**未深入：prefill 命中后 `cu_seqlens_k != cu_seqlens_q` 走 eager 路径，是否影响 decode graph 复用未验证。

---

## 参考来源

- nano-vllm 源码：`nanovllm/engine/block_manager.py`、`scheduler.py`、`sequence.py`、`model_runner.py`、`nanovllm/layers/attention.py`、`config.py`（commit `f8d495d`，phase4 分支）
- vLLM v1 源码（本地 checkout `/home/cb/work/vllm/vllm`）：`vllm/v1/core/block_pool.py`、`kv_cache_utils.py`、`kv_cache_manager.py`、`single_type_kv_cache_manager.py`
- 相关设计文档：`docs/detailed_design.md`、`docs/chunked_prefill-调研报告-20260614.md`
