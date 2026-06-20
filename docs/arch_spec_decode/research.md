# 投机解码（T）对齐 — V1 现状调研

> 对照基准：本机 `/home/cb/work/vllm/vllm` @ tag `v0.15.1`。
> 第一梯队单机功能之 ③：纯单 GPU 的**延迟**优化，nano 最大的"名气功能"空白。

## 背景：什么是投机解码

**问题**：LLM 自回归解码每次只产 1 个 token，每个 token 都要完整跑一遍模型前向。decode 阶段
是**显存带宽受限**的——大部分时间在把几 GB 的权重从显存搬进计算单元，而单 token 前向的实际
计算量很小。也就是说"再多算几个 token"几乎不额外花时间，但"多跑几次前向"很贵。

**核心思想**：用一个**廉价**的方法先**猜** k 个后续 token（草案），再让**目标模型一次前向并行
验证**这 k 个位置是否正确。验证一次的代价 ≈ 单次 decode，却可能一口气确认多个 token——把
"k 次串行前向"压成"1 次并行前向 + 1 次廉价提议"。

**关键性质**：**不改变输出**。贪心投机解码产出的 token 序列与逐 token 贪心**逐位相同**；它只是
把多步前向合并，是纯粹的延迟优化，不是近似。验证步保证：凡与目标模型不一致的草案 token 一律
被拒绝、用目标 token 修正。

**作用 / 收益**：降低 decode 延迟（命中率高时 1.5~3×），尤其利好"可预测/重复"文本（代码、
JSON、引用、列表）。对单 GPU 单请求也有效——是纯单机的延迟优化，无需更多硬件。

**算法流程（一步）**：
```
1. propose：proposer 廉价提出 k 个候选 token d0..d_{k-1}
            （n-gram：在历史里找相同前缀后面跟了什么；或用小"草案模型"前向）
2. verify ：目标模型对"当前 token + k 个草案"共 k+1 个位置**并行**前向，得每位的预测（贪心取 argmax）
3. reject ：逐位比对草案 vs 目标——
            接受最长正确前缀；首个分歧处用目标 token 修正并停止；全中则额外接受 1 个"奖励 token"
            （奖励 = 目标在第 k+1 位的预测，本就免费算出来了）
4. 一步前向 → 接受 1..k+1 个 token；被拒绝位置的 KV 作废，长度只推进到接受数
```
单步最少进 1 token（草案全错也有奖励/修正位），最多进 k+1 token（草案全中）。

**proposer 谱系**：n-gram（本实现，零成本、无草案模型）< 后缀自动机 < 小草案模型 < EAGLE/Medusa
（轻量预测头，命中率最高）。本实现选最轻的 n-gram 演示完整机制。

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
