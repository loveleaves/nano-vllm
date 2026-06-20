# nano-vllm

一个轻量级、可学习的 LLM 推理引擎 —— vLLM 核心算法的简洁再现,以 Qwen3 为参考模型。
代码量小、注释完整、分阶段实现,适合作为理解现代推理引擎(PagedAttention、连续批处理、
前缀缓存、张量并行、CUDA Graph)的学习项目。

`phase5` 分支在保持"单模型(Qwen3)单节点"定位的前提下,按 vLLM 0.15.1(V1 架构)逐层对齐了
**分层骨架**(引擎组件拆分 + 异步通路、调度器子包、KV cache 三层、Executor 抽象 + 进程隔离、
持久化 InputBatch、结构化采样层、Attention 后端注册表)。逐项对齐说明见
[docs/nano_vs_vllm-架构对比](docs/nano_vs_vllm-架构对比-20260618.md)。

## 特性

- 🗂️ **PagedAttention**:KV cache 分页管理(block_size=256),消除显存碎片;`engine/kv_cache/`
  拆为 BlockPool / KVCacheManager / KVCacheSpec 三层
- ⚡ **统一连续批**:一步内 decode 与 prefill chunk 混排在同一 varlen 批(无 prefill/decode 阶段切换),
  decode 抢占,可插拔调度策略(FCFS / Priority)
- ♻️ **前缀缓存**:链式 xxhash 块哈希 + 引用计数共享,相同前缀的请求跳过重复 prefill
- 🧩 **Chunked Prefill**:长 prompt 分块处理,与 decode 混批,延迟更平稳
- 🚀 **多后端 FlashAttention + Triton**:prefill/decode **统一** `flash_attn_varlen_func`(decode 为 query 长 1 的退化),
  KV 写入用 Triton kernel;`get_attn_backend` 按 head_size/dtype/平台能力选 Flash / SDPA 后端(注册表可覆盖)
- 🧠 **持久化 InputBatch**:跨步常驻行槽位 + 增量块表 + 单次切片 H2D,decode 仅追加新块、结束行回收(condense)
- 🎲 **结构化采样**:真·greedy(temperature=0)、top-k / top-p、presence/frequency/repetition 惩罚、logprobs
- 📈 **CUDA Graph**:decode 阶段按 batch size 录制 graph 复放,消除 kernel 启动开销
- 🔗 **张量并行(TP)+ 进程隔离**:Column/Row/QKV 并行线性层 + 词表并行 Embedding/LMHead;
  Executor 抽象(UniProc 内联 / MultiProc 全 rank 子进程隔离),ShmTransport 广播 + ResultChannel 回传
- 🌊 **异步流式**:`AsyncLLM` 提供 async generator 流式逐步输出(见 `example_async.py`)
- 🔥 **torch.compile**:Fused Add-RMSNorm 编译融合
- ✅ **CPU 可测**:flash-attn / Triton 为可选依赖,不可用时退回 SDPA / Python scatter,
  245 个单元测试无 GPU 即可运行

## 安装

要求 Python ≥ 3.11。

```bash
git clone https://github.com/loveleaves/nano-vllm.git
cd nano-vllm
pip install -e .

# GPU 推理(可选):安装 flash-attn 与 triton
pip install -e ".[gpu]"

# 开发与测试(可选)
pip install -e ".[dev]"
```

## 快速开始

接口与 vLLM 基本兼容,见 `example.py`:

```python
import os
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer

path = os.path.expanduser("~/model/Qwen3-1.7B/")
tokenizer = AutoTokenizer.from_pretrained(path)
llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)

# temperature=0 → 真·greedy(确定性);也可设 top_k / top_p / *_penalty / logprobs
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
prompts = [
    tokenizer.apply_chat_template(
        [{"role": "user", "content": "introduce yourself"}],
        tokenize=False,
        add_generation_prompt=True,
    )
]
outputs = llm.generate(prompts, sampling_params)
print(outputs[0]["text"])
```

```bash
python example.py          # 同步批量生成
python example_async.py    # AsyncLLM 异步流式
```

### 主要配置项

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `max_num_batched_tokens` | 16384 | 单步最多处理的 token 总数 |
| `max_num_seqs` | 512 | 单步最多并发序列数 |
| `max_model_len` | 4096 | 最大序列长度(自动截断到模型上限) |
| `gpu_memory_utilization` | 0.9 | 显存利用率,剩余部分全部分配给 KV cache |
| `tensor_parallel_size` | 1 | 张量并行 GPU 数(1–8) |
| `scheduling_policy` | `"fcfs"` | waiting 队列排队策略:`"fcfs"` 或 `"priority"` |
| `distributed_executor_backend` | `None` | 执行器后端:`None`(按 TP 自动)/ `"uni"`(单进程内联)/ `"mp"`(各 rank 子进程隔离) |
| `enforce_eager` | False | 为 True 时禁用 CUDA Graph(调试用) |

`SamplingParams`:`temperature`(≥0,0 为 greedy)、`max_tokens`、`ignore_eos`、`stop`、
`top_p`、`top_k`、`presence_penalty`、`frequency_penalty`、`repetition_penalty`、`logprobs`。

## 架构概览

```
LLM / LLMEngine(facade) / AsyncLLM     用户接口:同步 generate / 异步流式
   │  Processor(tokenize)  ·  OutputProcessor(增量 detokenize + 停止串 → RequestOutput)
   ▼
EngineCore.step()                       持 Scheduler + Executor,产 EngineCoreOutputs
   ├─ Scheduler (engine/sched/)          统一连续批 → 结构化 SchedulerOutput;KVCacheManager(分页+前缀缓存)
   └─ Executor (engine/executor/)        UniProc 内联 / MultiProc 进程隔离;ShmTransport + ResultChannel
         └─ Worker → ModelRunner         InputBatch(增量行)→ forward(eager / CUDA graph)→ Sampler
               └─ Qwen3ForCausalLM       RMSNorm / Attention(多后端 Flash+Triton/SDPA)/ SwiGLU / TP 线性层
```

注意力元数据 `AttentionMetadata` 经 forward 链**显式透传**(非全局单例)。详细架构与数据流见
[docs/01_architecture.md](docs/01_architecture.md)、[docs/03_data_flow.md](docs/03_data_flow.md);
各 V1 对齐轮次(A–L)的 research/design/testing、工程优化调研报告、模型适配文档的完整索引见
[docs/README.md](docs/README.md)。

## 实现阶段

| 阶段 | 内容 |
|------|------|
| Phase 1 | 基础数据结构:Config、Sequence、BlockManager、Scheduler |
| Phase 2 | 神经网络层:RMSNorm、RoPE、Attention、Linear、Sampler、Qwen3 模型 |
| Phase 3 | 权重加载(safetensors)+ 单进程完整推理 |
| Phase 4 | 工程优化:前缀缓存、Chunked Prefill、TP、CUDA Graph、Flash+Triton、torch.compile |
| Phase 5 | 对齐 vLLM 0.15.1(V1)分层骨架:A/B 统一连续批+显式 AttentionMetadata · C/D 多后端 Attention+Worker/RPC 解耦 · E 引擎拆分+异步 · F 调度器子包 · G KV cache 三层 · H Executor 抽象 · I InputBatch 增量行 · J 结构化采样层 · K Worker/Executor 进程隔离 · L Attention 后端注册表 |

## 分支说明

仓库包含**两条独立的提交历史**:上游原版代码(`main`)与从零开始的自研重写(`my_nano`
及各 phase 分支,根提交为架构设计文档 + 测试设计)。

| 分支 | 作用 |
|------|------|
| `main` | 上游 [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) 原版代码,仅用于跟踪与对照,不在其上开发 |
| `my_nano` | **自研实现的集成主分支**:从架构设计文档起步,Phase 1–4 经 PR #1–#4 依次并入 |
| `phase1` / `phase2` / `phase3` | 各阶段功能分支(已并入 `my_nano`);`phase3` 并入后追加了 prefill 跨序列注意力修复 |
| `phase3_model_adapt` | 自 `phase3` 分出的新模型适配分支:Qwen3.5(dense / MoE、GDN 修复)与 Qwen3.6 dense 支持 |
| `phase4` | Phase 4 工程优化(已并入) |
| `phase5` | **当前开发分支**:对齐 vLLM V1 分层骨架(A–L,见上表) |

## 测试

```bash
pytest -m unit    # 单元测试,纯 CPU,无需 GPU(245 个)
pytest -m gpu     # 集成测试,需要 CUDA GPU 和模型权重
```

> 注:真 GPU 端到端验证经 `example.py` / `example_async.py`(模型 `~/model/Qwen3-1.7B/`);
> `tests/test_qwen3.py` 为 CPU 微模型结构测试。TP>1(多卡 NCCL)在单卡环境未真机验证。

## 目录结构

```
nanovllm/
├── config.py / sampling_params.py   # 配置与采样参数
├── llm.py                           # 用户接口(LLM = LLMEngine)
├── engine/                          # 引擎核心(V1 风格组件拆分)
│   ├── processor / core / detokenizer / output_processor / llm_engine / async_llm
│   ├── sched/        # 调度器子包(interface / output / request_queue / scheduler)
│   ├── kv_cache/     # KV cache 三层(block_pool / kv_cache_manager / interface)
│   ├── executor/     # Executor 抽象(abstract / uniproc / multiproc)
│   ├── worker / rpc / input_batch / block_table / model_runner / sequence
│   └── scheduler.py / block_manager.py   # 向后兼容垫片
├── layers/
│   ├── attention/    # backend 三件套 + registry + selector + flash/sdpa + kv_ops
│   ├── sample/       # 结构化采样(metadata / sampler / ops{topk_topp,penalties,logprobs})
│   └── linear / layernorm / rotary_embedding / activation / embed_head
├── models/qwen3.py                  # Qwen3ForCausalLM
└── utils/                           # context(AttentionMetadata)/ loader
```

## 致谢

- [vLLM](https://github.com/vllm-project/vllm) —— 算法与设计的源头
- [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) —— 本项目参考的上游极简实现
