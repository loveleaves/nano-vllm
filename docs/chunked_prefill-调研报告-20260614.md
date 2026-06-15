# Chunked Prefill 设计与实现对比调研报告：nano-vllm vs vLLM

**调研日期**：2026-06-14（2026-06-15 重写，对比基准升级到最新 vLLM main）
**调研意图**：技术向 —— 剖析 nano-vllm（当前工程，phase4）与开源 vLLM（V1，最新 main）的 Chunked Prefill 设计与实现，定位架构差异、设计取舍，为后续优化/学习提供依据
**对比基准**：
- nano-vllm：`nanovllm/engine/scheduler.py`、`model_runner.py`、`layers/attention.py`（commit `f8d495d`）
- vLLM V1（最新 main）：`/home/cb/work/vllm/vllm`，commit `68f5e565c`。调度器已从单文件拆分为 `vllm/v1/core/sched/`（`scheduler.py` 2372 行 + `interface.py`/`request_queue.py`/`output.py`/`async_scheduler.py`），attention 走 `vllm/v1/attention/backends/flash_attn.py`

> **本次重写要点**：上一版报告对比的是旧检出 `/home/cb/work/vllm/test/vllm`（`0.1.dev3`，scheduler 单文件、含 "全批至多 1 个 partial 请求" 的 persistent-batch 约束）。**最新 main 已彻底移除该约束**：分块改为纯预算驱动 + 每请求 `long_prefill_token_threshold` 上限，可同时存在任意多个 partial 请求。`max_num_partial_prefills` 等旧开关在 V1 scheduler 中已不再被读取（V0 遗留）。详见 §四、§五。

---

## 一、执行摘要

1. **核心架构分歧（未变）**：nano-vllm 仍是 **“prefill 步 / decode 步”两相分离**的调度模型（一步要么全 prefill、要么全 decode）；vLLM V1 取消 prefill/decode 概念，统一为 **“每个请求每步推进 `num_computed_tokens` 追赶 `num_tokens_with_spec`”** 的单一抽象，decode 只是 `num_new_tokens=1` 的特例。这是两者最根本的差异。

2. **分块粒度（vLLM 侧已升级）**：nano-vllm **只允许 waiting 队首的第一个 seq 分块**，其余 seq 必须整块装下才入批；vLLM V1 最新 main 对**任意请求**用 `min(num_remaining, token_budget)` 切分，并叠加**每请求上限 `long_prefill_token_threshold`**（默认 `max_model_len * 0.04`），**不再有“全批至多 1 partial”的约束**——这是相对旧版报告的关键修正。

3. **prefill 与 decode 能否同批（未变）**：nano-vllm **不能**——`schedule()` 中 prefill 命中后立即 `return ..., True` 提前返回；vLLM V1 **先调度 running（含 decode）再调度 waiting（新 prompt chunk）**，长 prompt 的 chunk 与众多 decode 请求装进**同一个 forward**，这是 chunked prefill 提升吞吐 / 平滑 ITL 的关键收益来源。

4. **Attention 内核路径（未变）**：nano-vllm 走**双路**——prefill 用 `flash_attn_varlen_func`、decode 用 `flash_attn_with_kvcache`，靠 `context.is_prefill` 分发；vLLM V1 **单路** `flash_attn_varlen_func`（`cu_seqlens_q=query_start_loc`、`seqused_k=seq_lens`、恒带 `block_table`），并对长公共前缀启用 **cascade attention**（`use_cascade`，两段 + LSE 合并）。

5. **KV 显存分配时机（未变）**：nano-vllm 在**首个 chunk 就为整个 prompt 一次性分配全部 KV 块**（“以简洁换内存”的取舍）；vLLM V1 每个 chunk 调度时 `allocate_slots(request, num_new_tokens)` **增量分配**，失败则进入 `while True` 抢占重试循环。

**核心结论**：nano-vllm 的 chunked prefill 是一个**正确但教学化的简化实现**——实现了“切分长 prompt + 中途不产 token + 与前缀缓存共用进度字段”的核心语义，但保留了 prefill/decode 两相分离的旧式骨架，因此放弃了 chunked prefill 最大的工程价值（prefill chunk 与 decode 混批）。vLLM V1 最新 main 把 chunked prefill 内化为调度器**默认且唯一**的工作方式，且相比一年前已**移除单 partial 限制、引入 long-prompt 阈值**，混批粒度更细、长 prompt 控制更精。

---

## 二、背景：Chunked Prefill 要解决什么问题

Prefill 计算量随 prompt 长度线性增长。一条数千 token 的长 prompt 若一次算完，会独占整个 batch 预算，把同批 decode 请求（每步只算 1 token、本该毫秒级返回）的延迟显著拖高，造成 **inter-token latency（ITL）抖动**。同时单步可处理 token 数有物理上限（`max_num_batched_tokens`）。

Chunked Prefill 的思想：**把长 prompt 的 prefill 切成多个 chunk，分摊到连续若干步**，每步只消耗预算内 token，使长 prompt 不再“霸占”算力，并让省下的预算装入 decode 请求 → 平滑延迟、提升 GPU 利用率。

vLLM 在 V0 通过 `enable_chunked_prefill` 开关引入；**到 V1 成为内建默认范式**（`SchedulerConfig.enable_chunked_prefill` 默认 `True`），并新增 `long_prefill_token_threshold` 控制“多长算长 prompt、单请求每步最多吃多少”。

### 注意（语义澄清）
Chunked Prefill 中，每个 chunk 都会完整执行 Transformer（Q/K/V、Attention、MLP）并立即生成对应 KV Cache；它只是把超长 prompt 的 prefill 拆成多次增量 prefill，而不是先只算 KV、最后再统一算 Attention。前几个 chunk 不需要产生 logits（不走 lm_head 采样），但仍必须完整经过所有 Transformer 层。

---

## 三、nano-vllm 的实现剖析

### 3.1 调度层（`scheduler.py` `schedule()`）

`schedule()` 返回 `(scheduled_seqs, is_prefill)`，结构上是**先 prefill 后 decode 的两段式**：

```python
# ── Prefill 段 ──
while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
    seq = self.waiting[0]
    remaining = self.max_num_batched_tokens - num_batched_tokens
    if remaining == 0:
        break
    if not seq.block_table:                       # 新请求
        num_cached_blocks = self.block_manager.can_allocate(seq)   # 探测前缀缓存
        if num_cached_blocks == -1:
            break
        num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
    else:                                          # chunked 续算
        num_tokens = seq.num_tokens - seq.num_cached_tokens

    if remaining < num_tokens and scheduled_seqs:  # ★只有队首才能分块
        break
    seq.num_scheduled_tokens = min(num_tokens, remaining)
    ...
if scheduled_seqs:
    return scheduled_seqs, True                    # ★prefill 命中即提前返回
# ── Decode 段（仅当本步无 prefill 时执行）──
```

**三条关键设计**：
1. **只让 waiting 队首分块**（`remaining < num_tokens and scheduled_seqs`）：仅当 `scheduled_seqs` 为空（本步第一个被调度者）时才允许切分；后续 seq 必须能整块装下。理由：防止预算被多个半截 prompt 瓜分导致谁都进不了 decode（饥饿）。
2. **prefill 优先且独占整步**：只要 waiting 能调出 prefill，本步就是纯 prefill 步，decode 被推迟。
3. **续算靠 `num_cached_tokens` 串联**：分块进度与前缀缓存命中量**复用同一字段** `num_cached_tokens`，下一步 `num_tokens - num_cached_tokens` 即剩余待算。

### 3.2 “分块中途不产 token”（`postprocess`）

```python
seq.num_cached_tokens += seq.num_scheduled_tokens
seq.num_scheduled_tokens = 0
if is_prefill and seq.num_cached_tokens < seq.num_tokens:
    continue                       # 仍在 prompt 中间 → 丢弃采样结果
seq.append_token(token_id)         # 仅最后一个 chunk 才追加
```

模型对中间 chunk 也会输出 logits，但只有 prompt 最后一个 token 的输出才是要采样的“下一个 token”，故中间步 `continue` 丢弃。

### 3.3 KV 写入与 slot_mapping（`model_runner.py` `prepare_prefill`）

`start = seq.num_cached_tokens`，只为本 chunk 的 `[start, end)` 区间构建 `slot_mapping`/`positions`/`input_ids`；`cu_seqlens_q` 记本 chunk query 长度，`cu_seqlens_k = end`（含已缓存前缀的完整 K 长度）。当 `cu_seqlens_k[-1] > cu_seqlens_q[-1]`（有前缀/已算 chunk 在 cache 里）时才构建 `block_tables`，让 attention 从分页 cache 读历史。

### 3.4 Attention（`attention.py` `forward`）

双路分发：
- **prefill**：`flash_attn_varlen_func(..., causal=True, block_table=context.block_tables)`；有前缀/续算时 `k_fa,v_fa = k_cache,v_cache`（从 cache 读），否则用本步 `k,v`。
- **decode**：`flash_attn_with_kvcache(...)`。
- **CPU/无 flash-attn fallback**：`_sdpa_prefill` 逐序列按 `cu_seqlens` 切开计算，`seqlen_k > seqlen_q` 时用右下对齐的 `tril(seqlen_k - seqlen_q)` mask 处理续算/前缀场景。

### 3.5 显存取舍

**首个 chunk 即为整个 prompt 分配全部 KV 块**（`block_manager.allocate(seq, num_cached_blocks)` 按 `seq.num_blocks` 全量分配）→ 长 prompt 提前占用显存，理由是“以简洁换内存：避免逐 chunk 增量分配的复杂度”。

---

## 四、vLLM V1（最新 main）的实现剖析

### 4.1 统一调度抽象（`scheduler.py:338-347` 注释原文）

> There's no "decoding phase" nor "prefill phase" in the scheduler. Each request just has the `num_computed_tokens` and `num_tokens_with_spec`. … the scheduler tries to assign tokens to the requests so that each request's `num_computed_tokens` can catch up its `num_tokens_with_spec`. This is general enough to cover chunked prefills, prefix caching, speculative decoding, and the "jump decoding" optimization in the future.

`num_tokens_with_spec = len(prompt) + len(output) + len(spec_tokens)`。**没有 prefill/decode 之分**，decode = `num_new_tokens` 恰为 1 的请求；spec decode 自然纳入（一次推进多个 token）。

### 4.2 调度顺序：先 RUNNING 后 WAITING（关键差异，未变）

```python
token_budget = self.max_num_scheduled_tokens
# 1) 先调度 RUNNING（含正在 decode 的 + 上一步未算完的 partial）
while req_index < len(self.running) and token_budget > 0:
    num_new_tokens = (request.num_tokens_with_spec
                      + request.num_output_placeholders
                      - request.num_computed_tokens)
    if 0 < long_prefill_token_threshold < num_new_tokens:   # ★每请求长度上限
        num_new_tokens = long_prefill_token_threshold
    num_new_tokens = min(num_new_tokens, token_budget)
    new_blocks = self.kv_cache_manager.allocate_slots(request, num_new_tokens, ...)  # 增量+抢占重试
    ...
    token_budget -= num_new_tokens
# 2) 再用剩余预算调度 WAITING（新 prompt 的首个 chunk）
while (self.waiting or self.skipped_waiting) and token_budget > 0:
    if len(self.running) == self.max_num_running_reqs: break
    num_new_tokens = request.num_tokens - num_computed_tokens
    if 0 < long_prefill_token_threshold < num_new_tokens:
        num_new_tokens = long_prefill_token_threshold
    if not enable_chunked_prefill and num_new_tokens > token_budget: break  # 关闭分块时整块或不调
    num_new_tokens = min(num_new_tokens, token_budget)
```

decode 请求和 prefill chunk 共享同一个 `token_budget`，**装进同一个 SchedulerOutput → 同一个 forward**。这正是 nano-vllm 缺失的能力。

### 4.3 ★分块约束已改写：从“单 partial”到“纯预算 + 长度阈值”

**这是相对旧版报告（基于 `test/vllm` `0.1.dev3`）的核心修正**：

| | 旧版 vLLM V1（`0.1.dev3`） | 最新 main（`68f5e565c`） |
|---|---|---|
| 谁能 partial | 全批至多 1 个 partial，且必须排批尾 | **无 partial 数量限制**，任意多请求可同时 partial |
| 约束机制 | `has_partial_request` 布尔（persistent-batch 实现限制，TODO 待移除） | 已删除；`grep has_partial_request` 在 V1 scheduler 中**零命中** |
| 单请求每步上限 | 无（只受 budget） | **`long_prefill_token_threshold`**（默认 `int(max_model_len*0.04)`），running/waiting 两段都套用 |
| 旧开关现状 | — | `max_num_partial_prefills` / `max_long_partial_prefills` 仍在 `config/scheduler.py` 定义（默认 1），但 **V1 `scheduler.py` 不再读取**，属 V0 遗留 |

设计意图变化：旧版限制单 partial 是受 model runner 的 persistent batch 实现所迫；新版去掉该限制后，长 prompt 的“霸占”改由 `long_prefill_token_threshold` 这一**按 prompt 长度（而非纯 budget）切分**的阈值来约束——长 prompt 每步最多吃 4% × max_model_len，给短 prompt / decode 留出更多混批空间，更贴近 chunked prefill 的本意。

### 4.4 WAITING 段的丰富职责（V1 远超 nano-vllm）

同一个 waiting 循环里，最新 main 还内联处理（nano-vllm 全无）：
- **前缀缓存**：`kv_cache_manager.get_computed_blocks(request)` 拿本地命中块；`num_computed_tokens` 据此跳过。
- **外部 KV / P-D 分离**：`connector.get_num_new_matched_tokens`（远端 KV 命中），`WAITING_FOR_REMOTE_KVS` 状态、`load_kv_async`。
- **多模态 encoder**：`_try_schedule_encoder_inputs` 限制 encoder 算力预算、避免切坏一个多模态 item。
- **Mamba/混合模型**：`_mamba_block_aligned_split` 把 chunk 对齐到 block_size 整数倍（线性注意力 state 缓存需要）。
- **LoRA**：`max_loras` 约束，超限的请求进 `skipped_waiting` 队列跳过。
- **调度策略**：`create_request_queue(self.policy)`（FCFS / priority），`skipped_waiting` 二级队列。

### 4.5 前缀缓存 / block 对齐边界（`scheduler.py:292-334` `_mamba_block_aligned_split` 等）

当 prompt 长度全部命中缓存导致 `num_new_tokens==0` 时，V1 仍需保证 `num_computed_tokens` 落在 block 边界（hybrid/mamba 模型尤甚），会回退到 block 对齐切分。nano-vllm 用 256 的大 block，分配粒度不同，未显式处理此类边界。

### 4.6 Attention：单路 varlen + cascade（`v1/attention/backends/flash_attn.py`）

全批一次 `flash_attn_varlen_func`（`cu_seqlens_q=query_start_loc`、`seqused_k=seq_lens`、恒带 `block_table`），prefill chunk 的多 query token 与 decode 的单 query token 在**同一调用**内由 varlen 机制统一处理；KV 写入用 `reshape_and_cache_flash`。当存在较长**公共前缀**时启用 **cascade attention**（`use_cascade = common_prefix_len > 0`）：把“共享前缀段”与“各自后缀段”分两次算再用 LSE 合并，省去前缀重复读。

---

## 五、对比分析

| 维度 | nano-vllm（phase4） | vLLM V1（最新 main `68f5e565c`） |
|---|---|---|
| **调度抽象** | prefill 步 / decode 步两相分离 | 统一 `num_computed_tokens → num_tokens_with_spec`，无相之分 |
| **一步内能否混 prefill+decode** | ❌ 不能（prefill 命中即提前 return） | ✅ 先 running 后 waiting，同批混合 |
| **谁能被分块** | 仅 waiting 队首 1 个 seq | 任意请求（**无 partial 数量限制**） |
| **单请求每步上限** | 无（受 budget） | `long_prefill_token_threshold`（默认 `max_model_len*0.04`） |
| **分块约束动机** | 防饥饿（简化规则） | 限制长 prompt 霸占 + 给混批留预算（旧版的单 partial 限制已删除） |
| **KV 块分配时机** | 首 chunk 一次性全量分配 | 每 chunk 增量 `allocate_slots`，失败抢占重试 |
| **Attention 内核** | 双路：varlen(prefill)+with_kvcache(decode) | 单路 varlen + cascade |
| **续算进度载体** | `num_cached_tokens`（与前缀缓存复用） | `num_computed_tokens`（与前缀/spec/外部 KV 复用） |
| **中途不产 token** | `postprocess` 中 `continue` 丢弃采样 | runner 仅对 `num_computed==num_tokens` 的请求采样 |
| **前缀缓存/P-D/多模态/Mamba/LoRA** | 仅本地前缀缓存 | 全部内联在 waiting 循环 |
| **block_size 典型值** | 256 | 16（默认） |
| **代码规模** | scheduler ~150 行 | `sched/scheduler.py` 2372 行（+ kv_cache_manager 等） |

### 5.1 收益差异的本质（未变）

Chunked Prefill 的核心工程价值有两条：(a) 把长 prompt 切小，避免单步过大；(b) **用切出来的预算余量装 decode，让两类请求混批**，平滑 ITL、抬高 GPU 占用。

nano-vllm 只实现了 (a)，**放弃了 (b)**——因为它 prefill 步独占、decode 步独占。所以在 nano-vllm 里，“分块”主要起到**限制单步 token 数 / 配合前缀缓存续算**的作用，而非经典 chunked prefill 的“混批平滑延迟”。这从 `return scheduled_seqs, True` 的提前返回可严格推出。

### 5.2 vLLM 侧的演进（本次重写新增）

最新 main 相比一年前的 `0.1.dev3` 在 chunked prefill 上的演进，恰好印证“工程价值靠混批粒度”：
- **删除单 partial 限制** → 允许任意多请求同时处于 prefill 中段，混批自由度更高；
- **引入 `long_prefill_token_threshold`** → 把“切多大”从纯 budget 改为**与 prompt 长度挂钩**，长 prompt 被强制细切，短 prompt/decode 抢到更多混批名额（配合 `max_long_partial_prefills` 的“短 prompt 插队”语义，虽然该 V0 开关在 V1 已不直接生效）；
- **调度器拆分 `sched/`** → 把策略队列（FCFS/priority）、async 调度、P-D 分离解耦，scheduler 主体更聚焦“预算分配”。

### 5.3 nano-vllm 的合理简化

- **只让队首分块**：一条 `and scheduled_seqs` 解决饥饿，逻辑极简、易测，代价是调度灵活性。
- **首 chunk 全量分配 KV**：省去增量 `allocate_slots` 与跨 chunk 块表维护，代价是长 prompt 提前占显存（教学/中小规模可接受）。
- **双路 attention**：decode 用 `flash_attn_with_kvcache` 是成熟高效路径，避免为 decode 也构造 varlen 元数据，代码更直观。

### 5.4 nano-vllm 的潜在问题/改进点

1. **无法 prefill+decode 混批** → 长 prompt 到来时，所有 decode 请求被推迟整步。若要对齐 vLLM 收益，需重构为“先调度 running decode，再用余量调度 waiting chunk”的单段循环，并让 attention 走统一 varlen。结构性改动。
2. **首 chunk 全量分配** 在 `max_num_seqs` 较大 + 长 prompt 并发时可能过早 OOM/触发抢占；可改为增量分配缓解。
3. **缺少长 prompt 阈值** → 现状靠 `max_num_batched_tokens` 间接限制单步，但单个超长 prompt 仍可吃满整个预算独占一步；可仿 `long_prefill_token_threshold` 给单 seq 每步加上限。
4. **`block_size=256` 偏大**：前缀缓存命中粒度粗（须 256 token 对齐），且“prompt 整除 block_size 且全命中前缀”的边界 vLLM 有专门处理而 nano-vllm 未显式处理——值得加测试。

---

## 六、综合结论与建议

**判断**：nano-vllm 的 chunked prefill 是一份**语义正确、结构清晰的最小可用实现**，准确复刻了“切分长 prompt + 中途不产 token + 与前缀缓存共用进度字段”三大核心机制，非常适合教学与理解。但它停留在 **V0 式 prefill/decode 两相分离**骨架内，因而**未兑现 chunked prefill 最重要的工程红利——prefill chunk 与 decode 的同批混合**。vLLM V1 最新 main 则把 chunked prefill 升格为调度器唯一范式，并已**移除单 partial 限制、引入 long-prompt 阈值**，混批更自由、长 prompt 控制更精。

**给当前工程的建议**（按性价比排序）：
1. **（高价值，结构性）** 若 phase5 目标是逼近真实吞吐：把 `schedule()` 改为单段循环——先消费 running 的 decode（各 1 token），再用 `token_budget` 余量给 waiting 切 chunk，允许同批返回；attention 相应改为统一 `flash_attn_varlen_func` + 全程 `block_table`，废弃 decode 专用分支。
2. **（中价值）** 把首 chunk 全量 KV 分配改为按 chunk 增量分配，降低长 prompt 并发下的显存峰值与抢占率。
3. **（中价值，新增）** 引入类似 `long_prefill_token_threshold` 的单 seq 每步上限，避免单条超长 prompt 独占整步预算。
4. **（低成本）** 为“prompt 长度整除 block_size 且全命中前缀”的边界补测试，对齐 vLLM 的块对齐回退逻辑，避免 `num_scheduled_tokens==0` 隐患。
5. **（文档）** 在 `detailed_design.md` 7.2 显式注明“当前 chunked prefill 不做 prefill/decode 混批”，避免读者误以为已具备 vLLM 同等收益。

---

## 七、调研局限性与待补充方向

1. **未做性能实测**：吞吐/ITL 差异为基于代码路径的逻辑推断，未在相同硬件跑 benchmark 量化“混批”的实际收益。
2. **`max_num_partial_prefills` 的历史路径未深挖**：已确认它在 V1 `sched/scheduler.py` 不被读取（V0 遗留），但其在 V0 引擎与某些回退路径下是否仍生效，未逐一追溯。
3. **cascade attention / spec decode / P-D 分离 / Mamba 对齐切分**等 V1 进阶机制仅点到为止，未展开各自的正确性与性能细节。
4. **未对比 V0 的 `_schedule_chunked_prefill`**：V0 也有混批，但实现路径与 V1 不同，本报告聚焦 V1。
5. **版本时效**：基于 commit `68f5e565c`；vLLM 迭代极快，`long_prefill_token_threshold` 默认系数（当前 0.04）与调度细节可能继续变动，引用时请以实际检出为准。

---

## 参考来源

- nano-vllm 源码：`nanovllm/engine/scheduler.py`、`model_runner.py`、`engine/sequence.py`、`layers/attention.py`、`utils/context.py`（commit `f8d495d`, phase4）
- nano-vllm 设计文档：`docs/detailed_design.md` §7.1 前缀缓存、§7.2 Chunked Prefill
- vLLM V1 源码（最新 main，`/home/cb/work/vllm/vllm`）：
  - `vllm/v1/core/sched/scheduler.py`（`schedule()` L336-265+，RUNNING L372-556，WAITING L558-)
  - `vllm/config/scheduler.py`（`enable_chunked_prefill` L84、`long_prefill_token_threshold` L80/L245、`max_num_partial_prefills` L70）
  - `vllm/v1/attention/backends/flash_attn.py`（varlen + cascade，`use_cascade` L471/L510、`reshape_and_cache_flash` L874）
- 本地 vLLM 版本：commit `68f5e565c`（"[PD][Nixl] Mamba prefix caching mode support"）
- 关联报告：`docs/prefix_caching-调研报告-20260614.md`、`docs/cuda_graph-调研报告-20260614.md`、`docs/flashattention_triton-调研报告-20260614.md`
