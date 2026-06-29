# CPU 执行后端对齐 — V1 现状调研

> 对照基准：本机 `/home/cb/work/vllm/vllm` @ tag `v0.15.1`（V1）。
> 起点：nano-vllm 的 `ModelRunner` 是**纯 GPU 执行器**——`__init__` 即 `dist.init_process_group("nccl")`
> + `torch.cuda.set_device` + `set_default_device("cuda")`，显存估算靠 `mem_get_info`，
> 输入缓冲全用 pinned 内存 + 异步 H2D。无 GPU 的机器上 import 能过、单测能跑（已有 SDPA / naive
> scatter fallback），但**完整推理路径起不来**。本轮（U）补 V1 的"CPU 执行后端"。

## 背景：什么是 CPU 执行后端

**问题**：推理引擎默认绑死 GPU——NCCL 进程组、CUDA 设备、CUDA graph、pinned 内存、
`cuda.mem_get_info` 显存估算等贯穿执行器。没有 GPU（CI 机、笔记本、纯 CPU 服务器）就完全
跑不起来，连功能验证、教学演示、小模型本地调试都做不了。

**核心思想——复用 GPU 执行器，仅"中和"CUDA 专属操作**：CPU 后端不是另写一套前向/调度，而是
把 GPU 执行器里**真正依赖 CUDA 的点**逐一参数化或旁路，其余（模型结构、注意力、KV 分页、采样、
调度）原样复用。vLLM 的做法极其典型：`CPUModelRunner(GPUModelRunner)` 直接继承，只覆写
少数方法（设备置 `cpu`、关 CUDA graph、把 device 张量替换成 cpu 张量、内存估算改读配置），
并用一个上下文管理器把 `torch.Event` / `torch.cuda.Stream` 替换成空占位，让父类构造不报错。

**需要中和的 CUDA 触点**（nano 对照）：
| 触点 | GPU 行为 | CPU 中和方式 |
|---|---|---|
| 分布式后端 | `init_process_group("nccl")` | TP=1 时进程组本就非必需（层按 world_size=1 工作），直接跳过；vLLM 用 `gloo` |
| 设备 | `cuda.set_device` + `set_default_device("cuda")` | `set_default_device("cpu")`，不 set_device |
| 显存估算 | `mem_get_info` − warmup 峰值 | 无此接口，改由 `cpu_kvcache_gb` 显式预留（对齐 V1 `VLLM_CPU_KVCACHE_SPACE`） |
| CUDA graph | decode 录制 replay | 关闭（强制 eager） |
| pinned 内存 + 异步 H2D | `pin_memory=True` + `.cuda(non_blocking=True)` | `pin_memory=False`，张量直接构造在 cpu（无拷贝） |
| 注意力 kernel | flash_attn varlen | selector 按 `device_type="cpu"` 自动选 SDPA |
| KV 写入 | Triton kernel | `key.is_cuda` 为 False → naive Python scatter |
| dtype | bf16/fp16 | CPU 对 fp16 算子支持不全 → 统一 fp32 |

**作用 / 收益**：让引擎"无 GPU 也能跑"——CI 上做端到端功能回归、笔记本上跑小模型、教学演示
不依赖显卡。代价是 CPU 算力/带宽远低于 GPU，仅适合小模型 / 短序列 / 功能验证，不追求吞吐。

## vLLM CPU 后端结构

```
vllm/platforms/cpu.py                 # CpuPlatform：device_type="cpu"、dist_backend="gloo"、
                                      #   supported_dtypes（按架构 bf16/fp16/fp32）、强制 enforce_eager
vllm/v1/worker/cpu_worker.py          # CPUWorker(Worker)：init_device 不碰 CUDA、OpenMP 线程绑定、
                                      #   determine_available_memory 返回 cpu_kvcache_space_bytes
vllm/v1/worker/cpu_model_runner.py    # CPUModelRunner(GPUModelRunner)：_torch_cuda_wrapper 占位
                                      #   Event/Stream、use_cuda_graph=False、device 张量替 cpu 张量、
                                      #   _sync_device/_init_device_properties 置空
vllm/v1/attention/backends/cpu_attn.py# CPU 注意力后端（torch SDPA / ipex / 自定义 op）
```

关键观察：
- **继承而非重写**：`CPUModelRunner` 仅 ~120 行，绝大部分逻辑继承自 `GPUModelRunner`，
  印证"中和 CUDA 触点"而非"另起炉灶"的思路。
- `_torch_cuda_wrapper()`：构造期把 `torch.Event` / `torch.cuda.Stream` 临时换成空占位类，
  使父类 `__init__` 里对它们的引用不在无 CUDA 环境炸掉。
- `_postprocess_tensors()`：把所有 `CpuGpuBuffer.gpu` 指向其 `.cpu`、device 张量替换成 cpu 张量。
- `determine_available_memory`（在 CPUWorker）：直接返回 `cpu_kvcache_space_bytes`，不做显存探测。

## 与 nano 的差距（本轮范围）

| V1 特性 | 是否对齐 | 说明 |
|---|---|---|
| device 抽象（cuda/cpu 分派） | ✅ | `Config.device` + `ModelRunner.is_cuda`/`self.device` 参数化 |
| CPU 专属执行器（继承复用） | ✅ | `CPUModelRunner(ModelRunner)` 仅覆写内存估算，其余继承 |
| 跳过 NCCL（CPU TP=1） | ✅ | `is_cuda` 为 False 时不建进程组（层按 world_size=1 工作） |
| 关闭 CUDA graph | ✅ | `Config.device=="cpu"` 强制 `enforce_eager=True` |
| pinned 内存 / 异步 H2D 旁路 | ✅ | `pin_memory=self.is_cuda`，`_to_device` 同步构造 |
| 内存估算改读配置 | ✅ | `_available_kvcache_bytes` 覆写为 `cpu_kvcache_gb` |
| SDPA 注意力 + naive KV 写入 | ✅ | 复用既有 fallback（selector / `key.is_cuda` 判定） |
| CPU 统一 fp32 | ✅ | `run_dtype = fp32 if not is_cuda`（权重 `copy_` 自动转换） |
| OpenMP 线程绑定 / NUMA 亲和 | ❌ | 性能调优，超出"能跑"范围 |
| gloo 分布式 CPU TP>1 | ❌ | nano CPU 限定 TP=1（多进程 CPU TP 需 gloo + MultiProc 扩展） |
| ipex / CPU 专用融合 kernel / 量化 | ❌ | nano 走通用 torch SDPA 路径 |
| CPU↔GPU KV offload / swap | ❌ | CPU 后端无 GPU，swap 概念不成立（门控禁用） |
