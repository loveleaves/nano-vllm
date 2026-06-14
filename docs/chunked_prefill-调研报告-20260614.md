# Chunked Prefill 设计与实现对比调研报告：nano-vllm vs vLLM

**调研日期**：2026-06-14
**调研意图**：技术向 —— 剖析 nano-vllm（当前工程，phase4）与开源 vLLM（V1，0.1.dev3 本地检出）的 Chunked Prefill 设计与实现，定位架构差异、设计取舍，为后续优化/学习提供依据
**对比基准**：
- nano-vllm：`nanovllm/engine/scheduler.py`、`model_runner.py`、`layers/attention.py`（commit f8d495d）
- vLLM V1：`/home/cb/work/vllm/test/vllm/vllm/v1/core/scheduler.py`（637 行单文件版，早于 `sched/` 拆分）

---

## 一、执行摘要

1. **核心架构分歧**：nano-vllm 仍是**"prefill 步 / decode 步"两相分离**的调度模型（一步要么全 prefill、要么全 decode）；vLLM V1 取消了 prefill/decode 概念，统一为**"每个请求每步推进 `num_computed_tokens` 追赶 `num_tokens`"**的单一抽象，decode 只是 `num_new_tokens=1` 的特例。这是两者最根本、影响最深远的差异。

2. **分块粒度不同**：nano-vllm **只允许 waiting 队首的第一个 seq 分块**（防饥饿的简化规则），其余 seq 必须整块装下才入批；vLLM V1 对**任意请求**都用 `min(num_remaining, token_budget)` 切分，唯一约束是"全批至多 1 个 partial 请求"（受 persistent batch 限制），且 partial 请求强制排在批尾。

3. **prefill 与 decode 能否同批**：nano-vllm **不能**——`schedule()` 中 prefill 命中后立即 `return ..., True` 提前返回，decode 请求要等下一步；vLLM V1 **先调度 running（含 decode）再调度 waiting**，长 prompt 的 chunk 与众多 decode 请求**装进同一个 forward**，这正是 chunked prefill 提升吞吐/降低 decode 抖动的关键收益来源。

4. **Attention 内核路径**：nano-vllm 走**双路**——prefill 用 `flash_attn_varlen_func`、decode 用 `flash_attn_with_kvcache`，靠 `context.is_prefill` 分发；vLLM V1 **单路** `flash_attn_varlen_func`，全批共用 `query_start_loc`/`seqused_k`/`block_table`，prefill 和 decode 的 token 在同一 varlen 调用里处理。

5. **KV 显存分配时机**：nano-vllm 在**首个 chunk 就为整个 prompt 一次性分配全部 KV 块**（文档 7.x 明确标注为"以简洁换内存"的取舍）；vLLM V1 在每个 chunk 调度时**按需 `allocate_slots(num_new_tokens)` 增量分配**，长 prompt 不会提前占满显存。

**核心结论**：nano-vllm 的 chunked prefill 是一个**正确但教学化的简化实现**——它实现了"切分长 prompt + 中途不产 token"的核心语义，但保留了 prefill/decode 两相分离的旧式（V0 风格）骨架，因此放弃了 chunked prefill 最大的工程价值（prefill chunk 与 decode 混批）。vLLM V1 则把 chunked prefill 内化为调度器的**默认且唯一**的工作方式。

---

## 二、背景：Chunked Prefill 要解决什么问题

Prefill 计算量随 prompt 长度线性增长。一条数千 token 的长 prompt 若一次算完，会独占整个 batch 预算，把同批 decode 请求（每步只算 1 token、本该毫秒级返回）的延迟显著拖高，造成 **inter-token latency 抖动**。同时单步可处理 token 数有物理上限（`max_num_batched_tokens`）。

Chunked Prefill 的思想：**把长 prompt 的 prefill 切成多个 chunk，分摊到连续若干步**，每步只消耗预算内 token，使长 prompt 不再"霸占"算力，并让省下的预算装入 decode 请求 → 平滑延迟、提升 GPU 利用率。

vLLM 在 V0 通过 `enable_chunked_prefill` 开关引入该特性（`config.py:1395` 仍可见此 flag）；**到 V1 直接成为内建的、不可关闭的调度范式**。

### 注意
Chunked Prefill 中，每个 Chunk 都会完整执行 Transformer（包括 Q/K/V、Attention、MLP），并立即生成对应的 KV Cache；它只是把一个超长 Prompt 的 Prefill 拆成多次增量 Prefill，而不是先只算 KV、最后再统一算 Attention。

但Chunk Prefill 的前几个 chunk 确实不需要产生 logits（不需要 lm_head），但它们仍然必须完整经过所有 Transformer Layer。

---

## 三、nano-vllm 的实现剖析

### 3.1 调度层（`scheduler.py:39-117`）

`schedule()` 返回 `(scheduled_seqs, is_prefill)`，结构上是**先 prefill 后 decode 的两段式**：

```python
# ── Prefill 段 ──
while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
    seq = self.waiting[0]
    remaining = self.max_num_batched_tokens - num_batched_tokens
    ...
    if not seq.block_table:                      # 新请求
        num_cached_blocks = block_manager.can_allocate(seq)   # 探测前缀缓存
        num_tokens = seq.num_tokens - num_cached_blocks * block_size
    else:                                         # chunked 续算
        num_tokens = seq.num_tokens - seq.num_cached_tokens

    if remaining < num_tokens and scheduled_seqs:   # ★只有队首才能分块
        break

    seq.num_scheduled_tokens = min(num_tokens, remaining)
    ...
if scheduled_seqs:
    return scheduled_seqs, True                  # ★prefill 命中即提前返回

# ── Decode 段（仅当本步无 prefill 时才执行）──
while self.running ...
```

**三条关键设计**：
1. **只让waiting队首分块**（`scheduler.py:77`）：`remaining < num_tokens and scheduled_seqs` —— 仅当 `scheduled_seqs` 为空（本步第一个被调度者）时才允许切分；后续 seq 必须能整块装下。理由：防止预算被多个半截 prompt 瓜分导致谁都进不了 decode（饥饿）。另外一种不切分情况：本步第一个被调度者已经满足 remaining >= num_tokens。
2. **prefill 优先且独占整步**：只要 waiting 里能调出 prefill，本步就是纯 prefill 步，decode 被推迟。
3. **续算靠 `num_cached_tokens` 串联**：分块进度与前缀缓存命中量**复用同一字段** `num_cached_tokens`，下一步 `num_tokens - num_cached_tokens` 即剩余待算。

### 3.2 "分块中途不产 token"（`scheduler.py:142`）

```python
seq.num_cached_tokens += seq.num_scheduled_tokens
if is_prefill and seq.num_cached_tokens < seq.num_tokens:
    continue                       # 仍在 prompt 中间 → 丢弃采样结果
seq.append_token(token_id)         # 仅最后一个 chunk 才追加
```

模型对中间 chunk 也会输出 logits，但只有 prompt 最后一个 token 的输出才是真正要采样的"下一个 token"，故中间步 `continue` 丢弃。

### 3.3 KV 写入与 slot_mapping（`model_runner.py:165-211`）

`prepare_prefill` 中 `start = seq.num_cached_tokens`，只为本 chunk 的 `[start, end)` 区间构建 `slot_mapping`、`positions`、`input_ids`；`cu_seqlens_q` 记本 chunk query 长度，`cu_seqlens_k = end`（含已缓存前缀的完整 K 长度）。当 `cu_seqlens_k[-1] > cu_seqlens_q[-1]`（说明有前缀/已算 chunk 在 cache 里）时才构建 `block_tables`，让 attention 从分页 cache 读历史。

### 3.4 Attention（`attention.py:118-189`）

双路分发：
- **prefill**：`flash_attn_varlen_func(..., causal=True, block_table=context.block_tables)`；有前缀/续算时 `k_fa,v_fa = k_cache,v_cache`（从 cache 读），否则用本步 `k,v`。
- **decode**：`flash_attn_with_kvcache(...)`。
- **CPU/无 flash-attn fallback**：`_sdpa_prefill` 逐序列按 `cu_seqlens` 切开计算，`seqlen_k > seqlen_q` 时用右下对齐的 `tril(seqlen_k - seqlen_q)` mask 处理续算/前缀场景（`attention.py:230-234`）。

### 3.5 显存取舍

文档明确记载：**"Chunked prefill 在首个 chunk 即为整个 prompt 分配全部 KV 块"** → 长 prompt 提前占用显存，理由是"以简洁换内存：避免逐 chunk 增量分配的复杂度"。代码上体现为 `block_manager.allocate(seq, num_cached_blocks)` 在首个 chunk 一次性按 `seq.num_blocks` 全量分配。

---

## 四、vLLM V1 的实现剖析

### 4.1 统一调度抽象（`scheduler.py:93-101` 注释原文）

> There's no "decoding phase" nor "prefill phase" in the scheduler. Each request just has the `num_computed_tokens` and `num_tokens` ... the scheduler tries to assign tokens so that each request's `num_computed_tokens` can catch up its `num_tokens`. This is general enough to cover chunked prefills, prefix caching, and the "jump decoding" optimization.

**没有 prefill/decode 之分**。统一抽象，decode = `num_new_tokens` 恰为 1 的请求。

### 4.2 调度顺序：先 running 后 waiting（关键差异）

```python
# 1) 先调度 RUNNING（含正在 decode 的 + 上一步未算完的 partial）
while req_index < len(self.running):
    num_new_tokens = min(request.num_tokens - request.num_computed_tokens, token_budget)
    new_blocks = kv_cache_manager.append_slots(request, num_new_tokens)  # 增量分配
    ...
    token_budget -= num_new_tokens

# 2) 再用剩余预算调度 WAITING（新 prompt 的首个 chunk）
if not preempted_reqs:
    while self.waiting:
        if has_partial_request: break          # 全批至多 1 个 partial
        ...
        num_new_tokens = min(request.num_tokens - num_computed_tokens, token_budget)
        new_blocks = kv_cache_manager.allocate_slots(request, num_new_tokens, computed_blocks)
```

decode 请求和 prefill chunk 共享同一个 `token_budget`，**装进同一个 SchedulerOutput → 同一个 forward**。这正是 nano-vllm 缺失的能力。

### 4.3 分块约束：persistent batch 限定单 partial

vLLM V1 的约束不是"只有队首能分块"，而是 **"全批至多 1 个请求处于 partial 状态，且必须排在批尾"**（`scheduler.py:116-121, 173, 250`）：

```python
has_partial_request = (request.num_computed_tokens + num_new_tokens < request.num_tokens)
```

任何请求都可被切分（受 budget 约束），但一旦某请求 partial，`has_partial_request` 置位，后续 waiting 请求停止调度。这是 V1 model runner 的 **persistent batch** 实现约束（注释标注为 TODO 待移除），而非饥饿考虑。

### 4.4 前缀缓存边界处理（`scheduler.py:204-214`）

当 prompt 长度整除 block_size 且全部命中缓存时 `num_new_tokens==0`，V1 **强制回退最后一个 block 重算**（`num_computed_tokens -= block_size; num_new_tokens = block_size`），因为 `allocate_slots` 假设 `num_computed_tokens` 是 block_size 整数倍。nano-vllm 用 256 的大 block 且分配粒度不同，未显式处理此边界。

### 4.5 Attention：单路 varlen（`v1/attention/backends/flash_attn.py:214-221`）

全批一次 `flash_attn_varlen_func`，`cu_seqlens_q=query_start_loc`、`seqused_k=seq_lens`、`block_table` 恒存在。prefill chunk 的多 query token 与 decode 的单 query token 在**同一调用**内由 varlen 机制统一处理；前缀场景还有 `cascade attention`（`prefix_output`/`suffix_output` 两段 + LSE 合并，`flash_attn.py:355-379`）。

---

## 五、对比分析

| 维度 | nano-vllm（phase4） | vLLM V1（0.1.dev3） |
|---|---|---|
| **调度抽象** | prefill 步 / decode 步两相分离 | 统一 `num_computed_tokens → num_tokens`，无相之分 |
| **一步内能否混 prefill+decode** | ❌ 不能（prefill 命中即提前 return） | ✅ 先 running 后 waiting，同批混合 |
| **谁能被分块** | 仅 waiting 队首 1 个 seq | 任意请求，约束为全批 ≤1 partial 且在批尾 |
| **分块约束动机** | 防饥饿（简化规则） | persistent batch 实现限制（TODO 待移除） |
| **KV 块分配时机** | 首 chunk 一次性全量分配 | 每 chunk 增量 `allocate/append_slots` |
| **Attention 内核** | 双路：varlen(prefill)+with_kvcache(decode) | 单路：varlen 统一 + cascade |
| **续算进度载体** | `num_cached_tokens`（与前缀缓存复用） | `num_computed_tokens`（与前缀缓存复用） |
| **中途不产 token** | `postprocess` 中 `continue` 丢弃采样 | model runner 仅对 `num_computed==num_tokens` 的请求采样 |
| **block_size 典型值** | 256 | 16（默认） |
| **代码规模** | scheduler ~150 行 | scheduler ~637 行（不含 kv_cache_manager 等） |

### 5.1 收益差异的本质

Chunked Prefill 的核心工程价值有两条：(a) 把长 prompt 切小，避免单步过大；(b) **用切出来的预算余量装 decode，让两类请求混批**，从而平滑 ITL、抬高 GPU 占用。

nano-vllm 只实现了 (a)，**放弃了 (b)**——因为它 prefill 步独占、decode 步独占。所以在 nano-vllm 里，"分块"主要起到**限制单步 token 数 / 配合前缀缓存续算**的作用，而非经典 chunked prefill 的"混批平滑延迟"。这在文档中虽未明说，但从 `return scheduled_seqs, True` 的提前返回可严格推出。

### 5.2 nano-vllm 的合理简化

- **只让队首分块**：用一条 `and scheduled_seqs` 就解决了饥饿，逻辑极简、易测（`tests/test_scheduler.py` 可覆盖），代价是调度灵活性。
- **首 chunk 全量分配 KV**：省去 `append_slots` 的增量逻辑与跨 chunk 块表维护，代价是长 prompt 提前占显存（对教学/中小规模可接受）。
- **双路 attention**：decode 用 `flash_attn_with_kvcache` 是 V0 时代的成熟高效路径，避免为 decode 也构造 varlen 元数据，代码更直观。

### 5.3 nano-vllm 的潜在问题/改进点

1. **无法 prefill+decode 混批** → 长 prompt 到来时，所有 decode 请求会被推迟整步。若要对齐 vLLM V1 收益，需重构为"先调度 running decode，再用余量调度 waiting chunk"的单段循环，并让 attention 走统一 varlen。这是结构性改动。
2. **首 chunk 全量分配** 在 `max_num_seqs` 较大 + 长 prompt 并发时可能过早 OOM/触发抢占；可改为按 chunk 增量分配缓解。
3. **`block_size=256` 偏大**：前缀缓存命中粒度粗（必须 256 token 对齐才命中一块），且首块重算等边界 vLLM 已处理而 nano-vllm 未显式处理——长 prompt 边界正确性值得加测试。

---

## 六、综合结论与建议

**判断**：nano-vllm 的 chunked prefill 是一份**语义正确、结构清晰的最小可用实现**，准确复刻了"切分长 prompt + 中途不产 token + 与前缀缓存共用进度字段"三大核心机制，非常适合教学与理解。但它停留在 **V0 式 prefill/decode 两相分离**骨架内，因而**未能兑现 chunked prefill 最重要的工程红利——prefill chunk 与 decode 的同批混合**。vLLM V1 则把 chunked prefill 升格为调度器的唯一范式，decode 沦为 `num_new_tokens=1` 的退化情形。

**给当前工程的建议**（按性价比排序）：
1. **（高价值，结构性）** 若 phase5 目标是逼近真实吞吐：把 `schedule()` 改为单段循环——先消费 running 的 decode（各 1 token），再用 `token_budget` 余量给 waiting 队首切 chunk，allow 同批返回；attention 相应改为统一 `flash_attn_varlen_func` + 全程 `block_table`，废弃 decode 专用分支。
2. **（中价值）** 把首 chunk 全量 KV 分配改为按 chunk 增量分配，降低长 prompt 并发下的显存峰值与抢占率。
3. **（低成本）** 为"prompt 长度整除 block_size 且全命中前缀"的边界补测试，对齐 vLLM 的最后一块重算逻辑，避免 `num_scheduled_tokens==0` 的隐患。
4. **（文档）** 在 `detailed_design.md` 7.2 显式注明"当前 chunked prefill 不做 prefill/decode 混批"，避免读者误以为已具备 vLLM 同等收益。

---

## 七、调研局限性与待补充方向

1. **vLLM 版本偏早**：本地检出为 `0.1.dev3`，scheduler 仍是单文件、含 persistent-batch 单 partial 约束。**最新 vLLM 已将 `sched/` 拆分并可能移除该约束、引入 `long_prefill_token_threshold`**（按 prompt 长度而非纯 budget 决定切分）。结论中"任意请求可分块"的描述对新版可能更彻底。建议对照最新 main 复核。
2. **未做性能实测**：吞吐/ITL 差异为基于代码路径的逻辑推断，未在相同硬件跑 benchmark 量化"混批"的实际收益。
3. **未覆盖 V0 chunked prefill 细节**：仅确认 `enable_chunked_prefill` flag 存在，未深入 V0 的 `_schedule_chunked_prefill` 与 V1 的差异（V0 也有混批，但实现路径不同）。
4. **cascade attention / 多 partial** 等 V1 进阶机制仅点到为止，未展开。

---

## 参考来源

- nano-vllm 源码：`nanovllm/engine/scheduler.py`、`model_runner.py`、`engine/sequence.py`、`layers/attention.py`、`utils/context.py`（commit f8d495d, phase4）
- nano-vllm 设计文档：`docs/detailed_design.md` §7.1 前缀缓存、§7.2 Chunked Prefill、§限制表
- vLLM V1 源码：`vllm/v1/core/scheduler.py`（`schedule()` L93-265）、`vllm/v1/attention/backends/flash_attn.py`（L157-379）、`vllm/config.py`（L1395-1489 `enable_chunked_prefill`）
- 本地 vLLM 版本：`0.1.dev3+gcd90accbd`
