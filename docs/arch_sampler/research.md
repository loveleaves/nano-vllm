# 采样层对齐 — V1 现状调研

> 对照基准：本机 `/home/cb/work/vllm/vllm` @ tag `v0.15.1`（V1 架构）。

## V1 `v1/sample/` 组件

| 文件 | 职责 | nano 对应 |
|---|---|---|
| `sample/metadata.py::SamplingMetadata` | 一批序列的采样配置（temperature/top-p/top-k/penalties/历史 token/logprobs/generators/bad_words/allowed_ids/logitsprocs/spec） | `layers/sample/metadata.py`（最小子集） |
| `sample/sampler.py::Sampler` | 主流程：logprobs 留存 → float32 → allowed/bad_words → logitsprocs → penalties → sample（greedy/温度+topk/topp）→ gather_logprobs → SamplerOutput | `layers/sample/sampler.py` |
| `sample/ops/topk_topp_sampler.py` | top-k/top-p 掩码 + 加权随机采样（flashinfer / 原生 sort 两路；random_sample 用 Exponential gumbel） | `layers/sample/ops/topk_topp.py`（仅原生路径） |
| `sample/ops/penalties.py` + `model_executor/layers/utils.py::apply_penalties` | 重复/频率/存在惩罚（repetition 走 CUDA 自定义 op） | `layers/sample/ops/penalties.py`（纯 torch repetition） |
| `sample/ops/logprobs.py` + Sampler.gather_logprobs | logprobs 计算与 top-k 收集、排名 | `layers/sample/ops/logprobs.py` |
| `sample/ops/bad_words.py` / `logits_processor/` / `rejection_sampler.py` | bad_words、min-tokens/logit-bias/min-p 处理器、投机拒绝采样 | ❌ 不引入 |
| `v1/outputs.py::SamplerOutput` / `LogprobsTensors` | 采样输出张量契约 | `layers/sample/outputs.py` |

## 关键观察

1. **结构化元数据 + 一次性向量化**：V1 把整批的采样配置打成 `SamplingMetadata`，Sampler
   对整批 logits 向量化处理；`all_greedy`/`all_random`/`no_penalties`/`top_p is None` 等
   批级标志用于整段跳过，使常见路径（纯 greedy / 纯温度）零额外开销。
2. **逐行 greedy/random 混合**：`sample` 先算 greedy（argmax），再算 random，最后
   `torch.where(temperature < eps, greedy, random)` 按行择一——一个批里可同时有 greedy
   与随机请求。
3. **logprobs 取惩罚/温度前的原始 logits**（与 V0 不同），top-k + 采样 token 一并 gather，
   附带排名。
4. **repetition penalty 看 prompt∪output**，frequency/presence 只看 output；定义对齐 OpenAI。

## nano 对齐前差距

nano 旧 `layers/sampler.py::Sampler` 只有 `forward(logits, temperatures)`：温度缩放 +
Gumbel-max argmax，**无 greedy（assert temperature>0）/ top-k / top-p / 任何惩罚 / logprobs**。
E2E 靠 `temperature=0.01` 近似 greedy。

## nano 取舍（不引入）

generators（逐请求种子复现）、bad_words、allowed_token_ids、min-p、min-tokens/logit-bias
等 logits 处理器、投机拒绝采样、flashinfer 内核。nano 保持单进程、全精度、Gumbel 原生采样。
