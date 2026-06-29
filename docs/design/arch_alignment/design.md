# nano-vllm 向 vLLM V1 核心执行架构对齐 — 详细设计文档

> 范围：A（统一连续批）+ B（显式 AttentionMetadata）。基于 `research.md` 选型。

## Motivation

消除 nano-vllm 的两处与 vLLM V1 的本质代差：

1. **阶段式调度**：`schedule()→(seqs, is_prefill)` 使 prefill 批与 decode 批互斥，decode 序列被迫等 prefill 整批完成 → GPU 气泡。
2. **全局元数据单例**：`set_context/get_context` 进程级裸全局变量，不可重入、不可单测构造、与 graph/未来多后端冲突。

目标：重构为 **统一连续批（prefill chunk 与 decode token 同批混排）+ 显式 `AttentionMetadata` 逐层传参**，行为对齐 V1 且 Qwen3 端到端输出零变化。

---

## Architecture

### 数据流（重构后）

```
LLMEngine.step()
  │
  ├─ scheduler.schedule() ──────────────► (scheduled_seqs, num_scheduled: dict[seq_id,int])
  │     先 running（decode=1 / 续 chunk>1），后 waiting（新 prefill chunk）
  │     单遍按 token_budget 分配；无 is_prefill
  │
  ├─ model_runner.run(scheduled_seqs) ──► token_ids
  │     │
  │     ├─ prepare_inputs(seqs) ─────────► (input_ids, positions, AttentionMetadata)
  │     │     每 seq: q 段 = [num_cached, num_cached+num_scheduled)
  │     │     拼 query_start_loc / seq_lens / slot_mapping / block_table
  │     │
  │     ├─ run_model(input_ids, positions, attn_md)
  │     │     纯 decode 批(所有 num_scheduled==1) → CUDA graph replay
  │     │     否则 → eager
  │     │       └─ model.forward(input_ids, positions, attn_md)  # attn_md 逐层透传
  │     │             └─ Attention.forward(q,k,v, attn_md)        # 单一 varlen 调用
  │     │
  │     └─ sampler(logits, temperatures)
  │
  └─ scheduler.postprocess(seqs, token_ids, num_scheduled)
        chunk 未完成的 seq 不产 token；完成的追加 token + 查终止
```

### 关键变化点对照

| 模块 | 重构前 | 重构后 |
|---|---|---|
| `utils/context.py` | `Context` 单例 + set/get/reset | `AttentionMetadata` 纯数据类，无全局 |
| `scheduler.schedule` | `(seqs, is_prefill)` | `(seqs, num_scheduled: dict)` |
| `model_runner` | `prepare_prefill`+`prepare_decode` | 单一 `prepare_inputs` |
| `attention.forward` | prefill/decode 两分支 + 两 fallback | 单一 varlen 调用（按 block_table 有无分流，非按 is_prefill） |
| `qwen3.py` forward 链 | `(positions, hidden_states)` | 透传 `attn_metadata` |
| CUDA graph | decode 批触发 | 纯 decode 批（query_len 全=1）触发，逻辑不变 |

---

## Interfaces

### 1. `AttentionMetadata`（`utils/context.py` → 重构）

```python
@dataclass
class AttentionMetadata:
    """单步推理的 attention 元数据，显式传入 Attention.forward。取代全局 Context。

    统一字段（prefill chunk 与 decode token 共用，无 is_prefill 分支）：
      query_start_loc — [num_seqs+1] 累计 query 长度（cu_seqlens_q）。decode 时步长恒为 1
      seq_lens        — [num_seqs] 每 seq 的 KV 总长 = num_cached + num_scheduled（flash 的 seqused_k）
      max_query_len   — 批内最大 query 长度。==1 即纯 decode 批（graph 可用）
      max_seq_len     — 批内最大 KV 总长
      slot_mapping    — [total_tokens] 每个本步 token 写入 KV cache 的绝对 slot
      block_table     — [num_seqs, max_blocks] 分页 KV 地址；None 表示无 cache（仅 warmup）
    """
    query_start_loc: torch.Tensor | None = None
    seq_lens: torch.Tensor | None = None
    max_query_len: int = 0
    max_seq_len: int = 0
    slot_mapping: torch.Tensor | None = None
    block_table: torch.Tensor | None = None

    @property
    def is_decode_only(self) -> bool:
        return self.max_query_len == 1
```

> 删除模块级 `_CONTEXT`、`get_context`、`set_context`、`reset_context`。

### 2. `Scheduler.schedule`（`engine/scheduler.py`）

```python
def schedule(self) -> tuple[list[Sequence], dict[int, int]]:
    """统一连续批调度。返回 (scheduled_seqs, num_scheduled[seq_id]->本步 token 数)。

    单遍策略（对齐 V1）：
      1. 先调度 running：每个 seq num_new = num_tokens - num_cached_tokens
         （decode=1；被前缀缓存/chunk 续算时 >1），截断到 token_budget。
         can_append 不足 → 抢占 running 末尾。
      2. 再调度 waiting：can_allocate 探前缀缓存，num_new = num_tokens - num_cached，
         按 budget 切 chunk（任意 req 可分块，不再限队首）。chunk 未完 → 留 waiting。
    无 is_prefill 返回值；prefill/decode 仅是 num_new 不同。
    """
```

`postprocess` 签名相应改为 `(seqs, token_ids, num_scheduled: dict)`：用 `num_scheduled[seq.seq_id]` 取代旧 `is_prefill` 判断 chunk 是否完成（`num_cached + num_sched < num_tokens` → 不产 token）。

### 3. `ModelRunner`（`engine/model_runner.py`）

```python
def prepare_inputs(self, seqs) -> tuple[Tensor, Tensor, AttentionMetadata]:
    """合并 prepare_prefill/decode。每 seq 取 q 段 [num_cached, num_cached+num_scheduled)，
    拼 input_ids/positions/query_start_loc/seq_lens/slot_mapping/block_table。
    decode seq 即 num_scheduled==1 的退化情形，逻辑同一循环复用。"""

def run(self, seqs) -> list[int] | None:        # 删 is_prefill 形参
def run_model(self, input_ids, positions, attn_md):
    """attn_md.is_decode_only 且 bs<=512 且非 enforce_eager → graph replay；否则 eager。"""
```

graph 路径：持久化一个 `AttentionMetadata` 实例包裹静态缓冲（slot_mapping/seq_lens/block_table 等），replay 前 in-place 更新这些张量；`self.model(input_ids, positions, graph_attn_md)` 捕获时即绑定静态张量引用，replay 读更新值。

### 4. `Attention.forward`（`layers/attention.py`）

```python
def forward(self, q, k, v, attn_md: AttentionMetadata) -> Tensor:
    """统一单一 varlen 调用，无 is_prefill 分支。
      1. 始终 store_kvcache(k, v, slot_mapping)（slot=-1 跳过）
      2. block_table 非 None → flash_attn_varlen_func(
             q, k_cache, v_cache,
             cu_seqlens_q=query_start_loc, max_seqlen_q=max_query_len,
             seqused_k=seq_lens,           max_seqlen_k=max_seq_len,
             block_table=block_table, causal=True, softmax_scale=scale)
         覆盖 prefill / decode / 前缀缓存。
      3. block_table 为 None（warmup，无 cache）→ 裸 k/v varlen（旧 prefill 路径）
      4. 非 CUDA / 无 flash_attn → _sdpa_unified（按 query_start_loc 逐 seq，
         seqlen_k>seqlen_q 时右下对齐 causal mask，decode 为 seqlen_q==1 特例）
    """
```

`_sdpa_prefill` → `_sdpa_unified`：现有逐序列 + 右下对齐 mask 逻辑已能覆盖 decode（query_len=1），合并两 fallback 分支。

### 5. `qwen3.py` forward 链透传

```python
Qwen3Attention.forward(self, positions, hidden_states, attn_md)
Qwen3DecoderLayer.forward(self, positions, hidden_states, residual, attn_md)
Qwen3Model.forward(self, input_ids, positions, attn_md)
Qwen3ForCausalLM.forward(self, input_ids, positions, attn_md)
```
`self.attn(q, k, v, attn_md)` 显式下传。

### 6. `LLMEngine.step`（`engine/llm_engine.py`）

```python
seqs, num_scheduled = self.scheduler.schedule()
num_tokens = sum(num_scheduled.values())     # 吞吐统计：>0 prefill-heavy / 用 max_query 判
token_ids = self.model_runner.call("run", seqs)
self.scheduler.postprocess(seqs, token_ids, num_scheduled)
```
吞吐统计：以 `num_scheduled` 是否含 >1 值区分 prefill/decode 显示（沿用 pbar）。

---

## State Machine

`Sequence` 状态机不变（WAITING/RUNNING/FINISHED）。`is_prefill` 字段**保留**（`__getstate__` pickle 优化依赖它），但其值改由调度派生：`is_prefill = num_cached_tokens < num_prompt_tokens`（prompt 未算完即 prefill 阶段）。`SequenceStatus.RUNNING` 的进入时机不变（prompt 全部算完移入 running）。

---

## Risks

| 风险 | 影响 | 缓解 |
|---|---|---|
| `flash_attn_varlen_func` 不支持 `seqused_k` | 统一 kernel 不可用 | Task 1 先写 attention 单测探测签名；不支持则用 `cu_seqlens_k` 等价构造 |
| decode 改走 varlen 与原 `with_kvcache` 数值不一致 | 输出漂移 | Phase 7 example.py 贪心逐 token diff baseline |
| CUDA graph 与混合批冲突 | replay 崩溃/错误 | `run_model` 仅纯 decode 批（max_query_len==1）走 graph；graph_attn_md 包裹静态张量 |
| 抢占语义迁移到统一循环破坏前缀缓存 | 命中率下降/死锁 | 复用现有 `preempt`（保留 block hash）；test_scheduler 加抢占用例 |
| warmup 无 cache 路径 | block_table=None 崩溃 | Attention.forward 对 block_table=None 走裸 k/v 分支 |
| TP 多进程 pickle 兼容 | rank>0 崩溃 | num_scheduled 走 seq.num_scheduled_tokens 字段（已在 __getstate__），dict 仅 rank0 用 |

## Test Plan

- **Unit**：
  - `test_context.py`（改）→ `AttentionMetadata` 数据类构造、`is_decode_only`。
  - `test_scheduler.py`（改）→ 断言返回 `(seqs, dict)`；**新增混合批用例**（1 prefill chunk + N decode 同批）；抢占用例。
  - `test_attention.py`（改）→ 单一 varlen 路径；decode=query_len1 特例；SDPA 统一 fallback 数值对齐。
- **Integration**：`model_runner.prepare_inputs` 混合批的 query_start_loc/slot_mapping/seq_lens 正确性（CPU 构造断言）。
- **E2E**：`example.py` Qwen3 贪心输出与 main 分支 baseline 逐 token 一致（验收 5）；143 单测全绿（验收 4）。

---

## 设计评审自答（Phase 4）

1. **是否合理/有无逻辑漏洞**：调度去阶段化与 V1 注释算法一致；attention 统一为 varlen 已被 V1 生产验证。漏洞点在 warmup 无 cache → 已用 block_table=None 分支覆盖。
2. **扩展性**：显式 `attn_metadata` 入参为后续范围 C（多后端 builder）铺平路——只需把 metadata 构造抽到 builder，forward 签名不变。比全局单例改动量小。
3. **兼容现有接口**：`LLMEngine.generate/add_request` 对外接口不变；`step` 内部签名变。TP 多进程协议靠 `seq.num_scheduled_tokens`（已在 pickle）不变。`is_prefill` 字段保留兼容 `__getstate__`。
4. **性能开销**：统一 varlen 对 decode 理论上与 `with_kvcache` 同量级（同为分页读）；混合批反而减少 GPU 气泡。显式传参无运行时 overhead（仅引用传递）。graph 路径不变，纯 decode 性能不退化。

---

**门控：请确认设计通过（Phase 4 评审）。** 确认后进入 Phase 5 任务拆解 + Phase 6 编码。
