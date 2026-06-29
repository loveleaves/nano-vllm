# nano-vllm vs vLLM 0.15.1 (V1) — 全局架构对比

> 对照基准：本机 `/home/cb/work/vllm/vllm` @ tag `v0.15.1`（V1 架构）。
> nano-vllm 经 A–N 轮对齐（见 `docs/arch_*/`、记忆 `project_v1_arch_alignment`），
> 核心热路径模块布局已**刻意镜像 vLLM v1**。因此"区别"主要不在结构对应关系，
> 而集中在三个层面：**进程拓扑、广度、深度**。

---

## 一、最根本区别：进程拓扑

这是两者最本质的架构分歧。

### vLLM V1：EngineCore 独立进程 + 多进程 Worker

```
[前端进程] LLMEngine / AsyncLLM
   │  Processor(tokenize) → EngineCoreRequest
   │  ZMQ (msgpack)  ← v1/engine/core_client.py / coordinator.py
   ▼
[EngineCore 进程] Scheduler + Executor        ← 独立进程，事件循环
   │  collective_rpc
   ▼
[Worker 进程 × N] rank0..rankN-1   (NCCL / PP / EP)
```

- 前端（API / tokenize / detokenize）与调度内核（EngineCore）**跨进程**，ZMQ 通信，
  前端不被 GPU 阻塞；支持 DP coordinator、多 EngineCore 实例。

### nano-vllm：EngineCore 可选独立进程（P 轮），Worker 隔离

```
[前端进程] LLMEngine/AsyncLLM + Processor + OutputProcessor + EngineCoreClient
   │  InprocClient: 同进程直调（默认）   │  MPClient: mp.Queue（ADD/ABORT/OUTPUTS, pickle）
   ▼                                    ▼
[EngineCore 进程] busy-loop：Scheduler + Executor
   │  (UniProc: rank0 内联)            (MultiProc: ShmTransport / ResultChannel)
   ▼
[Worker 进程 × N]   共享内存广播
```

- **P 轮起**引入 `EngineCoreClient` 抽象：`InprocClient`（默认，同进程直调，零回归）
  与 `MPClient`（EngineCore 独立子进程 busy-loop，前端经 mp.Queue 收发），由
  `config.multiproc_engine_core` 选择。进程拓扑由此对齐 V1 三级。
- 传输用 **stdlib multiprocessing.Queue 替代 ZMQ**（与 Worker 层用 SharedMemory 替代 ZMQ
  一致）；未做 DP coordinator / 多 EngineCore / 后台 socket IO 线程。
- 进程隔离到 **Worker 层**（K 轮）用单槽共享内存（ShmTransport + ResultChannel）。

> **影响**：开启 MPClient 后，前端（tokenize/detokenize/HTTP）与 GPU 调度内核分进程、
> 前端保持 CUDA-free，已具备 V1"前后端解耦内核"的拓扑骨架；仍未做 ZMQ/DP/多核心，
> 故大规模服务伸缩能力有限，定位仍是教学 / 单机推理。

---

## 二、模块级对应关系（结构高度同构）

A–N 轮对齐后，核心路径几乎一一对应：

| 维度 | vLLM V1 | nano-vllm | 对齐度 |
|---|---|---|---|
| 引擎拆分 | `v1/engine/{llm_engine,core,core_client,processor,output_processor,detokenizer,async_llm}` | `engine/{llm_engine,core,core_client,processor,output_processor,detokenizer,async_llm}` | ✅ 同构（P 轮补 core_client；少 coordinator/parallel_sampling） |
| 调度器 | `v1/core/sched/{interface,output,request_queue,scheduler,async_scheduler}` | `engine/sched/{interface,output,request_queue,scheduler}` + async 内嵌 EngineCore | ✅ 同构 |
| KV cache | `v1/core/{block_pool,kv_cache_manager,single_type_kv_cache_manager,kv_cache_coordinator}` | `engine/kv_cache/{block_pool,kv_cache_manager,interface}` | ⚠️ 同构但**无 Coordinator / 多组异构** |
| Executor | `v1/executor/{abstract,uniproc,multiproc,ray_*}` | `engine/executor/{abstract,uniproc,multiproc}` | ✅ 同构（**无 Ray**） |
| Worker / Runner | `v1/worker/{gpu_worker,gpu_model_runner,gpu_input_batch,block_table}`(+cpu/tpu/xpu) | `engine/{worker,model_runner,input_batch,block_table}` | ⚠️ 仅 GPU 单后端 |
| Attention | `v1/attention/{backend,selector}` + `attention/backends/*` | `attention/{backend,selector,registry,flash_attn,torch_sdpa}`（顶层包，Q 轮提出 layers/） | ⚠️ 注册表对齐，**仅 2 后端** |
| 采样 | `v1/sample/{metadata,sampler,rejection_sampler,ops/*}` | `sample/{metadata,sampler,outputs,ops/*}`（顶层包，Q 轮提出 layers/） | ⚠️ 无 rejection_sampler(spec) |
| 模型注册 | `model_executor/models/registry.py`（数百架构 + 子进程探测 + 磁盘缓存） | `models/registry.py`（1 架构 + 惰性导入） | ⚠️ N 轮对齐机制，**无元信息缓存** |

**结论**：单批推理的"骨架"已对齐 V1。差距集中在下面两节。

---

## 三、广度差距（vLLM 有、nano 整类缺失）

构成最大的架构面积差：

| 能力域 | vLLM V1 | nano-vllm |
|---|---|---|
| **多模态** | `multimodal/`、encoder_cache_manager、MM input | ❌ 纯文本 |
| **LoRA / Adapter** | `lora/`、lora_model_runner_mixin | ❌ |
| **投机解码** | `v1/spec_decode/`、rejection_sampler、EAGLE/Medusa | ⚠️ T 轮：n-gram proposer + 拒绝采样 + **GPU verify 循环已集成**（EngineCore `_step_spec` + ModelRunner `verify_spec`，KV 自愈，UniProc，贪心等价）；无 EAGLE/Medusa |
| **结构化输出** | `v1/structured_output/`、reasoning、tool_parsers | ⚠️ S 轮：Logits Processor 框架 + 引导解码（ChoiceGrammar，UniProc）；无 xgrammar/正则/JSON |
| **PP / EP / DP** | pipeline / expert / data 并行、DP coordinator | ❌ 仅 TP |
| **量化** | 多种 quant、weight_loader 支持量化分片 | ❌ 仅全精度 safetensors |
| **多平台** | CUDA / CPU / TPU / XPU / ROCm（cpu/tpu/xpu_model_runner） | ❌ 仅 CUDA |
| **服务入口** | `entrypoints/`（OpenAI server、gRPC、Anthropic…） | ⚠️ O 轮补 OpenAI 兼容 server（/v1/completions+chat+models，流式 SSE）；无 gRPC/Anthropic/tools |
| **分布式后端** | Ray + MultiProc | ❌ 仅 MultiProc（共享内存） |
| **KV 异构 / 卸载** | KVCacheCoordinator 多组、kv_offload、kv_connector | ❌ 单组同构 full-attention |
| **模型库** | 数百架构、transformers / terratorch 后备 | ⚠️ 仅 Qwen3 + N 轮注册机制 |

---

## 四、深度差距（机制对齐但实现更浅，多为合理取舍）

- **KV cache**：对齐 BlockPool / Manager / 前缀缓存，但**无 KVCacheCoordinator**
  （vLLM 协调 full + sliding-window + mamba 多组异构 cache），nano 写死单组 full-attention。
- **Attention**：对齐枚举注册表 + 能力选择，但仅 FlashAttn / TorchSDPA 两枚；
  **builder 恒等**（vLLM 各后端 builder 做实质元数据重排）；无 kv_cache_dtype / block_size /
  MLA / sparse 多维能力。
- **采样**：对齐 SamplingMetadata + ops（top-k/p、min-p、penalties、bad_words、seed、logprobs），
  但进程隔离模式下**惩罚类采样不可用**（decode 仅传 last_token）；无 rejection_sampler。
- **InputBatch**：对齐增量行 + 块表增量，但靠**"行序==模型批序"简化**
  （连续批每步调度整个活跃集，无需 gather），vLLM 支持任意子集调度需 gather 重排。
- **async_scheduler**：对齐"占位 token + 采样留 GPU 跨步前向"，但**流水深度固定为 1**，
  仅 UniProc，与 swap / mp 互斥。
- **模型注册**：对齐惰性 importlib，但**无子进程 `_ModelInfo` 探测 + 磁盘 hash 缓存**
  （vLLM 为海量模型类的启动开销服务）。
- **Executor 进程模型**：vLLM rank0 也在子进程且经 ZMQ；nano 用单槽共享内存，靠 execute_model
  同步 + NCCL 集体保时序，**无 FailureCallback 健康监控**（M 轮补 alive_check 兜底）。

---

## 五、定性总结

| | vLLM 0.15.1 (V1) | nano-vllm |
|---|---|---|
| **定位** | 生产级多模态推理服务内核 | 教学 / 单机精简推理引擎 |
| **进程模型** | 前端 / EngineCore / Worker 三级跨进程，ZMQ | 三级拓扑可选（P 轮 MPClient，mp.Queue 替 ZMQ）；默认同进程 InprocClient |
| **驱动** | 异步事件循环，前后端解耦 | 同步驱动（generate 阻塞），async 同进程协程 |
| **并行** | TP / PP / EP / DP + Ray / MP | 仅 TP + MP |
| **代码量** | 数十万行，数百模型 / 多平台 / 多功能域 | ~50 个 py 文件，单模型单平台 |
| **核心路径架构** | — | **已对齐 V1 骨架（A–N）** |

**一句话**：nano-vllm 不是 vLLM 的子集裁剪版，而是**精确复刻了 V1 单批 GPU 推理热路径的模块
架构**（引擎分层、调度器、分页 KV、Executor / Worker、注册表式 Attention / 采样 / 模型），
同时**有意省去进程解耦（ZMQ / 独立 EngineCore）和全部"广度"能力域**（多模态 / LoRA / spec /
结构化 / 量化 / 多平台 / PP-EP-DP / 服务入口）。差距是"工程面积"而非"架构思路"——热路径思路
高度一致，省略项均为对单机单模型教学场景的合理取舍。

---

## 附：对齐工作索引（A–N）

| 轮 | 主题 | 文档 |
|---|---|---|
| A+B | 连续批 + 显式 AttentionMetadata | `docs/arch_alignment/` |
| C+D | 多后端 Attention 抽象 + Worker/RPC | `docs/arch_backend_worker/` |
| E | 引擎层拆分（Processor/Core/OutputProcessor/AsyncLLM） | `docs/arch_engine/` |
| F | 调度器 sched/ 子包 | `docs/arch_sched/` |
| G | KV cache 拆 BlockPool/Manager/Spec | `docs/arch_kvcache/` |
| H | Executor 抽象（UniProc/MultiProc） | `docs/arch_executor/` |
| I | InputBatch 增量行 + 块表增量 | `docs/arch_inputbatch/` |
| J | 采样层 sample/（原 layers/sample/，Q 轮上提） | `docs/arch_sampler/` |
| K | Worker/Executor 进程隔离 | `docs/arch_worker_isolation/` |
| L | Attention 后端注册表 / 能力选择 | `docs/arch_attn_registry/` |
| M | 6 项增强（logprobs/健壮性/采样/async/swap/metrics） | `docs/arch_async/`、`docs/arch_swap/` |
| N | 动态模型注册表 / 惰性加载 | `docs/arch_model_registry/` |
| O | 服务入口（OpenAI 兼容 API server） | `docs/arch_serving/` |
| P | EngineCore 进程化（进程拓扑：Inproc/MP 客户端） | `docs/arch_engine_proc/` |
| Q | 目录结构合理化：attention/、sample/ 从 layers/ 上提为顶层包（对齐 v1/attention、v1/sample） | — |
| R+S | Logits Processor 框架（penalties/bad_words 收编 + logit_bias/min_tokens）+ 引导解码（ChoiceGrammar，UniProc） | `docs/arch_logits_guided/` |
| T | 投机解码：n-gram proposer + 拒绝采样 + 编排 + **GPU verify 循环集成**（_step_spec/verify_spec，KV 自愈，UniProc，贪心等价） | `docs/arch_spec_decode/` |
