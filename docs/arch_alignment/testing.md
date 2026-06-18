# nano-vllm 向 vLLM V1 核心执行架构对齐 — 测试文档

## 测试环境

| 项 | 值 |
|---|---|
| GPU | NVIDIA RTX 3060 Ti (8GB) |
| torch / CUDA | 2.5.1 + cu121 |
| flash_attn | 2.8.3（无 `seqused_k`，用 `cu_seqlens_k` + `block_table`）|
| 模型 | Qwen3-1.7B（bf16，`~/model/Qwen3-1.7B/`）|
| 单测命令 | `pytest tests/ -q` |

## 测试用例清单

| 测试文件 | 测试类型 | 测试点 | 结果 |
|---|---|---|---|
| `tests/test_context.py` | Unit | `AttentionMetadata` 字段、`is_decode_only`、混合批/纯 decode 判定 | ✅ 5 passed |
| `tests/test_attention.py` | Unit | 统一 SDPA：MHA/GQA prefill、causal 正确性、**跨序列隔离**、**decode 读缓存**、GQA 广播 | ✅ 7 passed |
| `tests/test_embed_head.py` | Unit | LM head 按 `query_start_loc` 取末 token（prefill 取末/decode 恒等/无元数据全输出）| ✅ 11 passed |
| `tests/test_qwen3.py` | Unit | forward 链透传 `attn_md`、per-seq logits | ✅ 5 passed |
| `tests/test_scheduler.py` | Unit | 统一连续批：返回 dict、预算跨 seq 填充、**混合 prefill+decode 同批**、纯 decode 全 1、抢占、chunked prefill | ✅ 20 passed |
| `tests/test_sequence.py` | Unit | `is_prefill` 派生属性、pickle 优化（prefill 全 token / decode 仅 last_token）| ✅ 15 passed |
| 其余（block_manager/linear/sampler/...）| Unit | 未改动模块回归 | ✅ 全过 |
| **合计** | | | **✅ 149 passed, 4 skipped** |

> 4 skipped 为 `test_model_loader.py` 中需真实大权重的用例（环境无关，原本即 skip）。

### 集成 / E2E 测试（Qwen3-1.7B，greedy via temperature=0.01，max_tokens=48）

| 对比项 | 方法 | 结果 |
|---|---|---|
| **基线 eager == 新版 eager** | 同 prompts 逐 token 对比（baseline=phase5 HEAD worktree）| ✅ 逐 token 一致 |
| **基线 eager == 新版 CUDA graph** | 统一 varlen graph 捕获路径 | ✅ 逐 token 一致 |
| **基线 chunked == 新版 chunked (eager)** | `max_num_batched_tokens=32` 强制分块 + 混合批 | ✅ 逐 token 一致 |
| **基线 chunked == 新版 chunked (graph)** | 分块 + graph | ✅ 逐 token 一致 |

> 说明：新版 chunked 与基线 **非** chunked 输出不同——这是分块导致的 bf16 注意力数值差异，
> 基线在相同 `max_num_batched_tokens=32` 下产生 **完全相同** 的差异，故为既有特性而非回归。

## 验收标准对照（来自 PRD）

| 验收标准 | 测试方法 | 实测 | 达标 |
|---|---|---|---|
| 1. `schedule()` 不返回 `is_prefill`；返回 per-request `num_scheduled` dict，单批可混合 prefill+decode | `test_mixed_prefill_decode_batch`（断言同批 `min(ns)==1 且 max(ns)>1`）| 通过 | ✅ |
| 2. 无 `get_context()` 全局读取；`Attention.forward` 接收显式 `attn_md` | `grep` 全仓无残留（仅 `mp.get_context`）；签名 `forward(q,k,v,attn_md)` | 通过 | ✅ |
| 3. `context.py` 全局单例移除 / 降级为数据类 | `AttentionMetadata` 纯 dataclass，无 `_CONTEXT`/set/get/reset | 通过 | ✅ |
| 4. 全部单测通过 | `pytest tests/` | 149 passed, 4 skipped | ✅ |
| 5. Qwen3 端到端输出与重构前一致 | baseline worktree 逐 token diff（eager+graph+chunked）| 全部一致 | ✅ |

## 关键风险验证结果

| 设计风险 | 验证 | 结论 |
|---|---|---|
| flash_attn 2.8.3 无 `seqused_k` | 改用 `cu_seqlens_k`（累计形式）| 已规避 |
| decode 改走 varlen 与 `with_kvcache` 数值不一致 | 经验脚本：paged varlen vs with_kvcache `max diff = 0.0` | 数值一致 |
| CUDA graph 与 varlen 的 `max_seqlen_k` 整型烘焙 | 经验脚本：高估 `max_seqlen_k=4096` 偏差 4e-4（bf16 噪声）；E2E graph 与 eager 逐 token 一致 | 高估安全 |
| warmup 无 KV cache | `block_table is None` 走裸 k/v 分支；warmup 正常 | 已覆盖 |
| 抢占破坏前缀缓存 | `test_preempt_running_seq_when_no_free_blocks` + 既有前缀缓存测试 | 通过 |

## 已知局限

1. **范围限定 A+B**：未做多后端 `AttentionBackend` 抽象（范围 C）、EngineCore 进程解耦（范围 D）、外围特性（LoRA/量化/spec decode/PP/EP）。
2. **flash_attn 版本约束**：统一单一 varlen 调用依赖 `cu_seqlens_k`+`block_table`（2.8.3 可用）；升级到支持 `seqused_k` 的版本可进一步简化（去掉累计构造）。
3. **CUDA graph 触发面**：仅纯 decode 批（`max_query_len==1`）走 graph，混合批走 eager——与 vLLM "uniform decode 才进 graph" 策略一致，但混合批未享受 graph 加速。
4. **greedy 验证方式**：采样器为 Gumbel-max 无原生 greedy，E2E 用 `temperature=0.01`（softmax 近 one-hot）近似确定性 argmax 对比。
