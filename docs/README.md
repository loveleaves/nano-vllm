# Nano-vLLM 文档索引

本目录文档分五类：**核心实现文档**、**V1 架构对齐专题**、**工程优化调研报告**、**模型适配专题**、**通用参考**。

> `phase5` 分支已按 vLLM 0.15.1（V1 架构）逐层对齐分层骨架（轮次 A–L）。读架构请先看
> [01_architecture.md](01_architecture.md) 与 [nano_vs_vllm-架构对比](nano_vs_vllm-架构对比-20260618.md)，
> 各对齐轮次的 research/design/testing 见下方"二"。

## 一、核心实现文档

> 理解本仓库当前实现的首选入口（已更新至 A–L 对齐后的架构）。

| 文档 | 内容 |
|------|------|
| [05_setup.md](05_setup.md) | **环境搭建**：uv 环境、flash-attn 编译、运行 example.py（从这里上手） |
| [01_architecture.md](01_architecture.md) | **架构总览**：目录结构、请求生命周期、GPU 显存布局、Qwen3 模型树、Executor 与多进程架构、EngineCore 主循环、模块依赖图 |
| [02_core_concepts.md](02_core_concepts.md) | **核心技术**：PagedAttention、前缀缓存、统一连续批+Chunked Prefill、张量并行、CUDA Graph、统一 varlen FlashAttention、显式 AttentionMetadata、结构化采样层、Fused Add-Norm |
| [03_data_flow.md](03_data_flow.md) | **数据流与调度**：Prefill/Decode 逐步数据变换、CUDA Graph replay、前缀缓存路径、统一连续批调度（decode+prefill chunk 混排）伪代码 |
| [04_implementation_guide.md](04_implementation_guide.md) | **实现指南**：权重加载、显存估算、pin_memory 异步传输、性能优化全景、从零实现路线图、可扩展点、调试技巧 |
| [06_sleep_mode_design.md](06_sleep_mode_design.md) | Sleep Mode 设计分析：CUDA VMM 机制、三级睡眠、分步唤醒、RLHF 用法 |
| [07_triton_custom_ops.md](07_triton_custom_ops.md) | Triton 定制算子开发指南：编程模型、现有 kernel 解析、SiluAndMul/RMSNorm 示例 |

## 二、V1 架构对齐专题（A–L）

> 把 nano 逐层对齐 vLLM 0.15.1（V1）的设计/取舍/测试记录，每个子目录含 research / design / testing 三件套。

| 文档 | 对齐内容 |
|------|------|
| [nano_vs_vllm-架构对比](nano_vs_vllm-架构对比-20260618.md) | **全局对比**（A–L 后）：逐子系统"已对齐 / 仍有差距"，建议先读 |
| [arch_alignment/](arch_alignment/) | **A+B** 统一连续批调度 + 显式 AttentionMetadata |
| [arch_backend_worker/](arch_backend_worker/) | **C+D** 多后端 AttentionBackend 抽象 + Worker/RPC 解耦 |
| [arch_engine/](arch_engine/) | **E** 引擎层组件拆分（Processor/EngineCore/OutputProcessor）+ AsyncLLM 异步通路 |
| [arch_sched/](arch_sched/) | **F** 调度器子包（SchedulerInterface + 结构化 SchedulerOutput + 可插拔队列） |
| [arch_kvcache/](arch_kvcache/) | **G** KV cache 三层（BlockPool / KVCacheManager / KVCacheSpec） |
| [arch_executor/](arch_executor/) | **H** Executor 抽象（UniProc / MultiProc，Worker 瘦身） |
| [arch_inputbatch/](arch_inputbatch/) | **I** 持久化 InputBatch（增量行 + 行回收 + 增量块表 + 输出按 req_id 对齐） |
| [arch_sampler/](arch_sampler/) | **J** 结构化采样层（SamplingMetadata + Sampler + ops：greedy/top-k/top-p/penalties/logprobs） |
| [arch_worker_isolation/](arch_worker_isolation/) | **K** Worker/Executor 进程隔离（rank0 也进子进程 + ResultChannel 回传 + 块数 RPC） |
| [arch_attn_registry/](arch_attn_registry/) | **L** Attention 后端注册表 + 能力选择（AttentionBackendEnum + register_backend） |

## 三、工程优化专题调研报告（nano-vllm vs vLLM）

> 针对各项工程优化，与开源 vLLM 的设计/实现逐项对比。

| 文档 | 内容 |
|------|------|
| [prefix_caching-调研报告-20260614.md](prefix_caching-调研报告-20260614.md) | 前缀缓存：链式块哈希、引用计数共享 |
| [chunked_prefill-调研报告-20260614.md](chunked_prefill-调研报告-20260614.md) | Chunked Prefill：长 prompt 分块处理 |
| [tensor_parallelism-调研报告-20260614.md](tensor_parallelism-调研报告-20260614.md) | 张量并行：Column/Row/QKV 并行与多进程通信 |
| [cuda_graph-调研报告-20260614.md](cuda_graph-调研报告-20260614.md) | CUDA Graph：decode 按 batch size 录制复放 |
| [flashattention_triton-调研报告-20260614.md](flashattention_triton-调研报告-20260614.md) | FlashAttention 与 Triton：注意力与 KV 写入 |

## 四、模型适配专题（Qwen3.5 dense / MoE）

> 在 nano-vllm 上适配新模型架构的设计/调研/测试。完整索引见 [model_adaptation/README.md](model_adaptation/README.md)。

| 文档 | 内容 |
|------|------|
| [model_adaptation/vllm_model_adapt.md](model_adaptation/vllm_model_adapt.md) | vLLM 模型适配实战指南 |
| [model_adaptation/bug-prefill-cross-seq-attention.md](model_adaptation/bug-prefill-cross-seq-attention.md) | Bug 定位：Prefill 跨序列注意力污染 |
| [model_adaptation/qwen35_adaptation/](model_adaptation/qwen35_adaptation/) | Qwen3.5-2B（dense）research/design/testing |
| [model_adaptation/qwen35_moe_adaptation/](model_adaptation/qwen35_moe_adaptation/) | Qwen3.5-35B-A3B（MoE）research/design/testing |

## 五、通用参考

> 与本仓库当前代码无强绑定的通用技术资料。

| 文档 | 内容 |
|------|------|
| [dispatch.md](dispatch.md) | GPU 执行模式与调度机制（Eager / CUDA Graph / 整图下发，跨框架对比） |
| [op_adapt.md](op_adapt.md) | 大模型算子适配手册（算子分类、开发流程、融合/量化/并行优化） |
