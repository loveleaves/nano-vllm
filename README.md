# nano-vllm

一个轻量级、可学习的 LLM 推理引擎 —— vLLM 核心算法的简洁再现,以 Qwen3 为参考模型。
代码量小、注释完整、分阶段实现,适合作为理解现代推理引擎(PagedAttention、连续批处理、
前缀缓存、张量并行、CUDA Graph)的学习项目。

## 特性

- 🗂️ **PagedAttention**:KV cache 分页管理(block_size=256),消除显存碎片
- ⚡ **连续批处理**:FCFS 调度 + decode 抢占,最大化 GPU 利用率
- ♻️ **前缀缓存**:链式 xxhash 块哈希 + 引用计数共享,相同前缀的请求跳过重复 prefill
- 🧩 **Chunked Prefill**:长 prompt 分块处理,decode 延迟更平稳
- 🚀 **FlashAttention + Triton**:prefill 用 `flash_attn_varlen_func`、decode 用
  `flash_attn_with_kvcache`,KV 写入用 Triton kernel 向量化 scatter
- 📈 **CUDA Graph**:decode 阶段按 batch size 录制 graph 复放,消除 kernel 启动开销
- 🔗 **张量并行(TP)**:Column/Row/QKV 并行线性层 + 词表并行 Embedding/LMHead,
  多进程 SharedMemory + Event 通信,支持多 GPU
- 🔥 **torch.compile**:Fused Add-RMSNorm 与 Gumbel-max 采样器编译融合
- ✅ **CPU 可测**:flash-attn / Triton 为可选依赖,不可用时退回 SDPA / Python scatter,
  143 个单元测试无 GPU 即可运行

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

sampling_params = SamplingParams(temperature=1, max_tokens=256)
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
python example.py
```

### 主要配置项

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `max_num_batched_tokens` | 16384 | 单步最多处理的 token 总数 |
| `max_num_seqs` | 512 | 单步最多并发序列数 |
| `max_model_len` | 4096 | 最大序列长度(自动截断到模型上限) |
| `gpu_memory_utilization` | 0.9 | 显存利用率,剩余部分全部分配给 KV cache |
| `tensor_parallel_size` | 1 | 张量并行 GPU 数(1–8) |
| `enforce_eager` | False | 为 True 时禁用 CUDA Graph(调试用) |

`SamplingParams`:`temperature`(>0)、`max_tokens`、`ignore_eos`。

## 架构概览

```
LLM / LLMEngine          用户接口:generate(prompts, sampling_params)
   │
Scheduler                调度层:waiting/running 队列 + BlockManager(分页 + 前缀缓存)
   │
ModelRunner (per rank)   执行层:prepare → run_model(eager / CUDA graph)→ Sampler
   │
Qwen3ForCausalLM         模型层:RMSNorm / Attention(Flash+Triton)/ SwiGLU / TP 线性层
```

详细设计见 [docs/architecture.md](docs/architecture.md)(架构与关键设计决策)和
[docs/detailed_design.md](docs/detailed_design.md)(逐模块实现细节与已知限制);
各工程优化专题的 nano-vllm vs vLLM 对比调研报告与完整文档索引见 [docs/README.md](docs/README.md);
新模型架构(Qwen3.5 dense / MoE 混合架构)的适配设计、调研与测试见
[docs/model_adaptation/](docs/model_adaptation/README.md)。

## 实现阶段

| 阶段 | 内容 |
|------|------|
| Phase 1 | 基础数据结构:Config、Sequence、BlockManager、Scheduler |
| Phase 2 | 神经网络层:RMSNorm、RoPE、Attention、Linear、Sampler、Qwen3 模型 |
| Phase 3 | 权重加载(safetensors)+ 单进程完整推理 |
| Phase 4 | 工程优化:前缀缓存、Chunked Prefill、TP、CUDA Graph、Flash+Triton、torch.compile |

## 分支说明

仓库包含**两条独立的提交历史**:上游原版代码(`main`)与从零开始的自研重写(`my_nano`
及各 phase 分支,根提交为架构设计文档 + 测试设计)。

| 分支 | 作用 |
|------|------|
| `main` | 上游 [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) 原版代码,仅用于跟踪与对照,不在其上开发 |
| `my_nano` | **自研实现的集成主分支**:从架构设计文档起步,Phase 1–4 经 PR #1–#4 依次并入,源码阅读笔记(docs/01–07、dispatch、op_adapt 等) |
| `phase1` | Phase 1 功能分支:Config / Sequence / BlockManager / Scheduler(已并入 `my_nano`) |
| `phase2` | Phase 2 功能分支:RMSNorm / RoPE / Attention / Linear / Sampler / Qwen3 模型(已并入) |
| `phase3` | Phase 3 功能分支:权重加载 + 单进程推理(已并入);并入后追加了 prefill 跨序列注意力修复 `25b07bd` |
| `phase3_model_adapt` | 自 `phase3` 分出的新模型适配分支:Qwen3.5(dense / MoE、GDN 修复)与 Qwen3.6 dense 支持 |
| `phase4` | Phase 4 功能分支:前缀缓存 / Chunked Prefill / TP / CUDA Graph / Flash+Triton(已并入);当前开发分支 |

## 测试

```bash
pytest -m unit    # 单元测试,纯 CPU,无需 GPU(143 个)
pytest -m gpu     # 集成测试,需要 CUDA GPU 和模型权重
```

## 目录结构

```
nanovllm/
├── config.py / sampling_params.py   # 配置与采样参数
├── llm.py                           # 用户接口(LLM = LLMEngine)
├── engine/                          # sequence / block_manager / scheduler
│                                    # model_runner / llm_engine
├── layers/                          # attention / linear / layernorm / sampler
│                                    # embed_head / rotary_embedding / activation
├── models/qwen3.py                  # Qwen3ForCausalLM
└── utils/                           # context / loader
```

## 致谢

- [vLLM](https://github.com/vllm-project/vllm) —— 算法与设计的源头
- [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) —— 本项目参考的上游极简实现
