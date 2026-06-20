# 投机解码（T）— 详细设计

> 基于 `research.md`。交付**与 GPU 无关、纯 CPU 可测**的核心算法与组件
> （n-gram proposer + 拒绝采样 + propose/score/verify 编排），并明确 GPU verify 循环的
> 集成边界。理由：GPU 多位置 verify + KV 回滚 + 调度器推进侵入 Scheduler/InputBatch/
> KVCache，是最易引入回归的部分，单列为 UniProc-only 集成点。

## 范围决策

| 部分 | 是否交付 | 说明 |
|---|---|---|
| NgramProposer | ✅ | 历史 n-gram 匹配，最轻量 proposer |
| RejectionSampler（贪心验证） | ✅ | 接受最长前缀 + 修正/奖励 token |
| SpeculativeDecoder 编排 | ✅ | score_fn 注入，完整算法 |
| 贪心等价性 | ✅ | 单测证明 == 逐 token 贪心 |
| **GPU 多位置 verify + KV 自愈 + 调度推进** | ✅ **已集成** | EngineCore `_step_spec` + ModelRunner `verify_spec`，UniProc-only，门控 `speculative_num_tokens` |
| EAGLE / Medusa / 草案模型 | ❌ | 按 proposer 接口可扩 |
| 随机（非贪心）拒绝采样 | ❌ | 接口预留 |

## Architecture

```
spec_decode/
├── ngram_proposer.py   # NgramProposer.propose(token_ids) -> 最多 k 个提议 token
└── spec_decoder.py     # SpeculativeDecoder.step(token_ids, score_fn) -> 接受的 token 序列
sample/
└── rejection_sampler.py # RejectionSampler.verify_greedy(draft, target) -> 接受序列
```

### 算法

```
SpeculativeDecoder.step(token_ids, score_fn):
   draft  = proposer.propose(token_ids)              # k 个候选（或 [] 退化为单步）
   target = score_fn(token_ids, draft)               # 目标模型 k+1 个位置的 argmax
   return rejection_sampler.verify_greedy(draft, target)

verify_greedy(draft, target):   # len(target) == len(draft)+1
   accepted = []
   for i, d in enumerate(draft):
       accepted.append(target[i])
       if target[i] != d: return accepted            # 分歧：修正 token 后停止
   accepted.append(target[len(draft)])               # 全接受：追加奖励 token
   return accepted
```

`NgramProposer.propose`：取尾部长度 n∈[max_n..min_n] 的 n-gram，自后向前找更早的相同
n-gram，返回其后最多 k 个 token；优先用更长的 n-gram（匹配更可信）。

### GPU verify 循环集成（已实现，UniProc-only，门控 `Config.speculative_num_tokens>0`）

实现策略：**正常路径零回归**——spec 走独立 `EngineCore._step_spec`，先跑一次普通
`_step_sync`（推进 prefill / 产 1 个基准 token），再对每个 decode 序列做**多 token 扩展**：

```
_extend_with_spec(seq):                                   # EngineCore
  L0 = seq.num_tokens
  drafts = proposer.propose(seq.token_ids)                # n-gram
  if not drafts: return []                                # 退化普通 decode
  for d in drafts: seq.append_token(d); block_manager.may_append(seq)   # 投机追加 + 分配块
  targets = executor.verify_spec(seq, len(drafts))        # GPU：k+1 位置并行前向 → argmax
  accepted = rejection_sampler.verify_greedy(drafts, targets)
  final, finished, reason = apply_finish(seq, L0, accepted)   # EOS/max_tokens 逐 token 判定
  seq.token_ids = seq.token_ids[:L0] + final; seq.num_tokens = L0+len(final)
  seq.num_cached_tokens = seq.num_tokens - 1              # 与普通 decode 同不变式
  block_manager.truncate_blocks(seq)                      # 仅释放尾部块
```

**KV 自愈（无需显式回滚 KV）**：`verify_spec` 在位置 L0..L0+k-1 写入草案 token 的 KV。
- 被接受的匹配位（draft==target）：KV 正确。
- 修正位（首个分歧，draft≠target）：KV 为草案值（错），但它是新的"最新 token"
  （`num_cached = num_tokens-1`），下一步前向它时**覆写**为正确 KV——与普通 decode 中
  "最新 token KV 下一步才写"完全一致。
- 奖励位（全接受时的 target[k]，位置 L0+k）：verify 未写其 KV，同样下一步写。

**块回滚**：`KVCacheManager.truncate_blocks` 只 pop/deref **尾部**多余块；已分配的前部块
物理位置不变，故被接受 token 的 KV 完好。`num_cached = num_tokens-1` 保证块表不变式
（`len(block_table)*block_size >= num_tokens`），下一步 `may_append` 不重复分配。

**GPU verify_spec**（ModelRunner，eager 单序列）：query = [token@(L0-1), 草案…]（k+1 位置），
构造 AttentionMetadata（slot_mapping/cu_seqlens/block_table），前向后对**全部** k+1 位置取
`lm_head(hidden, attn_md=None)`（不做末位聚合）→ argmax。

门控同 async/swap（UniProc；与 async 互斥）。MultiProcExecutor.verify_spec 抛 NotImplementedError。

## 关键设计点

- **贪心等价性是正确性锚点**：单测用确定性目标 + n-gram proposer 跑完整循环，证明投机序列
  逐 token == 自回归贪心，且步数明显少于 token 数（加速）。
- **退化安全**：n-gram 未命中 → draft=[] → score_fn 返回 1 个 token → 等价普通单步。
- **proposer 可插拔**：NgramProposer 之外，EAGLE/草案模型/后缀自动机按 `propose` 接口接入，
  编排与拒绝采样不变。
