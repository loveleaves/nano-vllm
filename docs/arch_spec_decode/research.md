# 投机解码（T）对齐 — V1 现状调研

> 对照基准：本机 `/home/cb/work/vllm/vllm` @ tag `v0.15.1`。
> 第一梯队单机功能之 ③：纯单 GPU 的**延迟**优化，nano 最大的"名气功能"空白。

## V1 组件

| 文件 | 职责 |
|---|---|
| `v1/spec_decode/ngram_proposer.py` | n-gram 草案器（无草案模型，历史匹配提议） |
| `v1/spec_decode/eagle.py` / `medusa.py` / `draft_model.py` | EAGLE / Medusa / 独立草案模型 proposer |
| `v1/spec_decode/suffix_decoding.py` | 后缀自动机 proposer |
| `v1/sample/rejection_sampler.py` | 拒绝采样：验证草案、接受最长正确前缀 + 修正/奖励 token |
| `v1/spec_decode/metadata.py` | 投机元数据（每请求草案 token、接受数） |

## V1 机制

1. **propose**：proposer 廉价提出 k 个候选 token（n-gram 查表 / 草案模型前向）。
2. **verify**：目标模型对"当前 token + k 个草案"共 k+1 个位置**并行前向**，得每位 logits。
3. **reject**：逐位比对草案与目标；贪心下接受最长匹配前缀，首个分歧处用目标 token 修正，
   全接受则追加 1 个奖励 token。一步产出 1..k+1 个 token。
4. **rollback**：拒绝位置的 KV 作废，调度器按接受数推进、回收多写的块。

关键性质：**贪心投机 == 逐 token 贪心**（只加速、不改变输出）。

## 与 nano 的关系

- nano 的 **async 调度**（占位 token + 采样留 GPU 跨步前向，M 轮）与投机的"多步/多位置"
  思路相邻，是别的精简实现难做、而 nano 已铺好的地基。
- nano 的 **结构化采样层**（J 轮）已具 logprobs / 向量化采样，便于挂拒绝采样。

## 与 nano 的差距（本轮范围）

| V1 特性 | 是否对齐 | 说明 |
|---|---|---|
| n-gram proposer | ✅ | `spec_decode/ngram_proposer.py`（纯 CPU 可测） |
| 拒绝采样（贪心） | ✅ | `sample/rejection_sampler.py`（贪心；随机版接口预留） |
| propose→score→verify 编排 | ✅ | `SpeculativeDecoder`（score_fn 注入，与 GPU 无关） |
| EAGLE / Medusa / 草案模型 / 后缀自动机 | ❌ | 先做最轻的 n-gram；其余按 proposer 接口可扩 |
| GPU 多位置并行 verify + KV 回滚 + 调度器推进 | ⚠️ | 作为集成边界单列（侵入 Scheduler/InputBatch/KV，UniProc-only），本轮交付算法与组件 |
