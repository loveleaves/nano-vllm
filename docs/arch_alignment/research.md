# nano-vllm 向 vLLM V1 核心执行架构对齐 — 技术调研报告

> 范围：A（统一连续批）+ B（显式 AttentionMetadata）。排除多后端、进程解耦、外围特性。
> 参照版本：vLLM 0.15.1（V1 引擎，本地检出 `/home/cb/work/vllm/vllm` @ 501d4023a）。

## 摘要

vLLM V1 的"统一连续批"不是单点改动，而是**三层一致的去阶段化**：

1. **调度层**：`Scheduler.schedule()` 不区分 prefill/decode，只让每个 request 的 `num_computed_tokens` 追赶 `num_tokens`，输出 per-request `num_scheduled_tokens: dict`。
2. **输入构造层**：`prepare_inputs` 用 per-request `num_scheduled_tokens` 拼出**一个混合批**的 `query_start_loc`(=cu_seqlens_q)、`seq_lens`、`slot_mapping`、`block_table`，prefill chunk（query_len>1）与 decode（query_len=1）混排在同一批。
3. **Attention kernel 层**（关键）：prefill 与 decode 走**同一个** `flash_attn_varlen_func`，通过 `block_table` + `seqused_k`(=seq_lens) 寻址分页 KV。decode 仅是 query_len=1 的退化情形。**不再有 `flash_attn_with_kvcache` 专用 decode 分支。**

元数据通过显式 `FlashAttentionMetadata` 数据类逐层传入，无任何全局单例。

结论：nano-vllm 对齐 A+B 的核心是——(1) 调度器返回 dict 并合并两段循环；(2) 合并 `prepare_prefill/decode` 为单一 `prepare_inputs`；(3) **统一 attention 为单一 varlen 调用**；(4) 把 `Context` 降级为 `AttentionMetadata` 显式入参。

---

## 参照实现对比

### 参照对象 1：`vllm/v1/core/sched/scheduler.py` — 去阶段化调度

源码注释（L314-323）原文：
> "There's no 'decoding phase' nor 'prefill phase' in the scheduler. Each request just has the num_computed_tokens and num_tokens_with_spec ... At each step, the scheduler tries to assign tokens to the requests so that each request's num_computed_tokens can catch up its num_tokens."

**算法骨架**（剥离 spec/encoder/mamba/PD 后）：

```
token_budget = max_num_batched_tokens
num_scheduled_tokens: dict[req_id, int] = {}

# 1) 先调度 RUNNING（已在跑的，含正在 chunked-prefill 和正在 decode 的）
for request in running:
    if token_budget <= 0: break
    num_new = request.num_tokens - request.num_computed_tokens   # decode 时=1，chunk 时>1
    num_new = min(num_new, token_budget)
    new_blocks = allocate_slots(request, num_new)   # 不够则抢占 running 末尾
    num_scheduled_tokens[request.id] = num_new
    token_budget -= num_new

# 2) 再调度 WAITING（新请求 / 被抢占恢复的）
while waiting and token_budget > 0:
    request = waiting[0]
    if request.num_computed_tokens == 0:
        cached_blocks, num_cached = get_computed_blocks(request)  # 前缀缓存命中
    num_new = request.num_tokens - num_computed_tokens
    num_new = min(num_new, token_budget)        # chunked prefill
    allocate_slots(...)
    move request waiting->running
    num_scheduled_tokens[request.id] = num_new
    token_budget -= num_new

return SchedulerOutput(num_scheduled_tokens=..., total=sum(...))
```

**与 nano-vllm 现状的核心差异**：
| 点 | nano-vllm | V1 |
|---|---|---|
| 返回值 | `(list[Seq], is_prefill: bool)` | `num_scheduled_tokens: dict` |
| 阶段 | prefill 批与 decode 批互斥 | running(含 decode+续 chunk) 与 waiting(新 prefill) 同批 |
| chunk 限制 | 仅队首 seq 可分块 | 任意 request 按全局 budget 切 |
| "本步要算几个 token" | seq.num_scheduled_tokens（prefill）或恒 1（decode） | 统一 `num_tokens - num_computed_tokens` 截断到 budget |

**注意**：nano-vllm 已有 `num_cached_tokens` / `num_scheduled_tokens` 字段（语义≈V1 的 `num_computed_tokens` / 本步 token 数），数据模型已接近，主要差在调度循环结构与返回类型。

### 参照对象 2：`vllm/v1/attention/backends/flash_attn.py` — 统一 attention

`FlashAttentionMetadata` 关键字段（L185-201）：
```python
num_actual_tokens: int       # 本步总 token 数（= total_num_scheduled_tokens）
max_query_len: int           # 批内最大 query 长度（decode-only 时=1）
query_start_loc: Tensor      # [num_reqs+1] 累计 query 长度 == nano 的 cu_seqlens_q
max_seq_len: int             # 批内最大 KV 总长
seq_lens: Tensor             # [num_reqs] 每个 req 的 KV 总长 == flash 的 seqused_k
block_table: Tensor          # [num_reqs, max_blocks] 分页地址
slot_mapping: Tensor         # [num_actual_tokens] 本步 token 写入 KV 的绝对 slot
```

`forward()` 的统一调用（剥离 cascade/sink/MLA）：
```python
flash_attn_varlen_func(
    q=query[:num_actual_tokens], k=key_cache, v=value_cache,
    cu_seqlens_q=query_start_loc,
    max_seqlen_q=max_query_len,
    seqused_k=seq_lens,          # 每 req 的 KV 实际长度
    max_seqlen_k=max_seq_len,
    block_table=block_table,     # 分页 KV 寻址
    softmax_scale=scale, causal=True, ...
)
```
**一次调用覆盖 prefill+decode+前缀缓存。** key/value 直接传整个 paged cache，靠 `block_table`+`seqused_k` 定位。

**对 nano-vllm 的启示**：现 `attention.py` 有两条分支：
- prefill：`flash_attn_varlen_func`（前缀缓存时已传 `block_table`，但无前缀时传裸 k/v）
- decode：`flash_attn_with_kvcache`

→ 可统一为：**始终把 K/V 先写入 paged cache（已有 `store_kvcache`），再用单一 `flash_attn_varlen_func(block_table=..., seqused_k=...)`**。decode 即 `cu_seqlens_q` 步长为 1 的退化。这样消除 `is_prefill` 在 attention 层的分叉。SDPA fallback 同理可统一（按 query_start_loc 逐 req，causal 右下对齐已在 `_sdpa_prefill` 实现，decode 是其特例）。

### 参照对象 3：`vllm/v1/worker/gpu_model_runner.py` — 混合批输入构造

V1 维护持久化 `InputBatch`，`_prepare_inputs` 用 `num_scheduled_tokens`（per-req）做：
- `cu_num_tokens = cumsum(num_scheduled_per_req)` → query_start_loc（L1250）
- 每 req 取 `positions = range(num_computed, num_computed+num_sched)`
- `slot_mapping`：每个新 token 按 block_table 映射到绝对 slot
- `seq_lens[i] = num_computed_tokens[i] + num_scheduled_tokens[i]`

**对 nano-vllm 的启示**：nano 的 `prepare_prefill` 已几乎构造了全部这些量（cu_seqlens_q/k、slot_mapping、block_tables）。统一后 `prepare_decode` 可并入：decode req 的 `num_scheduled_tokens=1`、`seqlen_q=1`、`seqlen_k=len(seq)`。**两函数合一只需让 prefill 的循环接受 num_scheduled_tokens=1 的 req 即可**，逻辑高度复用。

### 参照对象 4：元数据传递方式（全局单例 vs 显式入参）

V1：`AttentionMetadata` 由 builder 构造，存入 `forward_context`（线程局部、按 forward 作用域 set/reset），attention layer 经 `get_forward_context().attn_metadata` 取——**但这是按层注册的、带类型的、可多组共存**，并非 nano 的进程级裸全局变量。更彻底的做法（也更适合教学库）是 `model.forward(input_ids, positions, attn_metadata)` 显式逐层下传。

**对 nano-vllm 的启示**：选择**显式入参**路线（最清晰、可单测、无重入问题）。`qwen3.py` 各 `forward` 增加 `attn_metadata` 参数透传给 `Attention.forward(q,k,v,attn_metadata)`。`utils/context.py` 的 `Context` 重命名/迁移为 `AttentionMetadata` 纯数据类，删除 `_CONTEXT` 全局与 `set/get/reset_context`。

---

## 当前代码库分析

涉及模块与改动面：

| 文件 | 当前职责 | 对齐改动 |
|---|---|---|
| `engine/scheduler.py` | `schedule()→(seqs,is_prefill)`，两段互斥循环 | 改返回 `(seqs, num_scheduled_tokens_dict)` 或在 seq 上带本步 token 数；合并 running/waiting 为"先 running 后 waiting"单遍，去 is_prefill |
| `engine/model_runner.py` | `prepare_prefill`/`prepare_decode` 分离；`run(seqs,is_prefill)`；`run_model` 按 is_prefill 分流 | 合并为 `prepare_inputs(seqs)` 构造混合批 + `AttentionMetadata`；`run(seqs)` 去 is_prefill；graph 仅在"全 decode 批"时触发 |
| `layers/attention.py` | prefill/decode 两分支 + 两套 fallback | 统一为单一 varlen 调用（始终 paged）；SDPA fallback 统一按 query_start_loc 逐 req |
| `utils/context.py` | 全局 `Context` 单例 + set/get/reset | 改为 `AttentionMetadata` 数据类，删全局 |
| `models/qwen3.py` | `forward(input_ids, positions)` | 透传 `attn_metadata` 到每层 `Attention` |
| `engine/llm_engine.py` | `step()` 用 is_prefill 算吞吐 | 适配新调度返回；吞吐统计改用 dict 求和 |
| `engine/sequence.py` | 有 `num_cached_tokens`/`num_scheduled_tokens`/`is_prefill` | 语义对齐；`is_prefill` 字段可保留为派生属性或移除 |
| `tests/test_scheduler.py` 等 | 断言 `(seqs,is_prefill)` 二元返回 | 重写为断言 dict + 混合批用例 |

**已具备的有利条件**：
- `flash_attn_varlen_func` 已在用且已支持 `block_table` 参数（attention.py L149）——统一 kernel 无需引入新依赖。
- `slot_mapping`/`cu_seqlens`/`block_tables` 构造逻辑已存在，复用即可。
- `num_cached_tokens`/`num_scheduled_tokens` 数据模型已接近 V1 的 `num_computed_tokens`。
- CUDA graph 已按 batch size 分桶——保留，仅约束触发条件为"批内全部 query_len==1"。

**主要风险点**：
1. **CUDA Graph 与混合批冲突**：graph 按"每 seq 1 token、固定 bs"录制。混合批含变长 chunk → 不能 replay。**缓解**：`run_model` 仅当批内所有 req 的 `num_scheduled_tokens==1`（纯 decode 步）才走 graph，否则 eager。这与 V1 思路一致（uniform decode 才进 graph）。
2. **`flash_attn_with_kvcache` → `flash_attn_varlen_func` 数值一致性**：统一后 decode 也走 varlen，需验证输出与原 kvcache 路径逐 token 一致。**缓解**：phase7 用 example.py 贪心逐 token diff。
3. **seqused_k 接口可用性**：需确认所装 flash_attn 版本 `flash_attn_varlen_func` 支持 `seqused_k`（部分版本叫法不同）。**缓解**：phase6 首个 task 先写 attention 单测探测接口签名。
4. **抢占语义**：V1 在 running 内按预算抢占；nano 现有 decode 段抢占逻辑需迁移到统一循环且不破坏前缀缓存。

---

## 结论与选型建议

**采纳方案：三层统一 + 显式元数据。**

1. **调度器**：保留 nano 的 `Sequence`/`BlockManager`，改 `schedule()` 为单遍"先 running 后 waiting"，返回 `(scheduled_seqs, num_scheduled_tokens: dict[seq_id,int])`，删 `is_prefill`。chunked prefill 放开到任意 req（按全局 budget）。
2. **输入构造**：合并 `prepare_inputs(seqs, num_scheduled)`，统一产出 `AttentionMetadata`。
3. **Attention**：统一为单一 `flash_attn_varlen_func(block_table, seqused_k)`，始终先 `store_kvcache`。SDPA fallback 统一逐 req。
4. **元数据**：`AttentionMetadata` 显式入参，删全局 `Context`。
5. **CUDA Graph**：保留，触发条件收紧为"纯 decode 批"。

**不采纳**：forward_context 线程局部方案（比显式入参更复杂，教学价值低）；多后端 builder 抽象（超范围 C）。

**验证策略**：每完成一层用 example.py 贪心输出与 main 分支 baseline 逐 token 对比，保证范式重构零行为变化（PRD 验收 5）。

下一阶段（Phase 3）将据此产出 `design.md`：给出 `AttentionMetadata` 字段定义、`schedule()`/`prepare_inputs()`/`Attention.forward()` 的接口签名骨架与数据流图。
