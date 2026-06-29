# CPU 执行后端 — 设计与实现（轮次 U）

## 设计原则

沿用 V1 思路——**复用 GPU 执行器，仅中和 CUDA 触点**，并把"是否 CUDA"收敛为一个属性
`ModelRunner.is_cuda`，让 GPU 路径**逐字节不变**（默认 `device="cuda"`），CPU 仅是其退化。

## 改动清单

### 1. `config.py`：device 抽象 + 门控
- 新增 `device: str = "cuda"`、`cpu_kvcache_gb: float = 4.0`。
- `__post_init__`：`device in ("cuda","cpu")`；`device=="cpu"` 时：
  - **强制** `enforce_eager=True`（CPU 无 CUDA graph）；
  - 断言 `TP==1`、`backend ∈ {None,"uni"}`、`not multiproc_engine_core`、`num_swap_blocks==0`、
    `cpu_kvcache_gb>0`——这些都依赖 GPU/多进程语义，CPU 后端限定单进程内联。

### 2. `model_runner.py`：按 device 参数化（GPU 路径不变）
- `__init__`：
  - `self.device = torch.device(config.device)`；`self.is_cuda = device.type=="cuda"`；
    `self.pin_memory = self.is_cuda`。
  - **NCCL/set_device 仅在 `is_cuda`**：TP=1 时层按 `world_size=1` 工作（`dist.is_initialized()`
    为 False → all_reduce 跳过），故 CPU 直接不建进程组。
  - `run_dtype = hf_config.dtype if is_cuda else torch.float32`（CPU 统一 fp32）。
  - `set_default_device(self.device)`；InputBatch 用 `device=self.device, pin_memory=self.pin_memory`；
    `capture_cudagraph` 仅 `is_cuda and not enforce_eager`。
- `exit()`：`cuda.synchronize/empty_cache/destroy_process_group` 与 graph 释放均裹 `if self.is_cuda`。
- `warmup_model()`：`cuda.empty_cache/reset_peak_memory_stats` 裹 `if self.is_cuda`（CPU 仅跑通前向）。
- `allocate_kv_cache()`：抽出 `_available_kvcache_bytes()`（GPU：`mem_get_info`−warmup 峰值），
  KVCacheSpec 的 dtype 改取 `torch.get_default_dtype()`（GPU 仍 = hf_config.dtype，CPU = fp32），
  保证块字节估算与实际 kv_cache 张量一致。
- `_to_cuda` → `_to_device`：`torch.tensor(..., pin_memory=self.pin_memory).to(self.device,
  non_blocking=self.pin_memory)`（GPU pinned 异步、CPU 同步构造）。8 处调用点同步改名。
- 持久 generator `torch.Generator(device=self.device)`；`verify_spec` 的 `dev=self.device`
  （投机解码在 CPU 上也能跑）。

### 3. `cpu_model_runner.py`（新）：`CPUModelRunner(ModelRunner)`
- 仅覆写 `_available_kvcache_bytes()` → `int(cpu_kvcache_gb * 1024³)`（对齐 V1
  `CPUWorker.determine_available_memory`）。其余全部继承——印证"中和而非重写"。

### 4. `worker.py`：按 device 选执行器
- `config.device=="cpu"` → `CPUModelRunner`，否则 `ModelRunner`（与 V1 `CPUWorker` 选 runner 同构）。

### 5. `sampler.py`：inference tensor 可写性修复（顺带，CPU 暴露）
- `logits = logits.float()` → `logits.to(torch.float32, copy=True)`。
- **原因**：模型前向在 `@torch.inference_mode()` 下产出 logits（inference tensor），采样在
  inference_mode 外就地改写。GPU 上 logits 是 bf16，`.float()` 必拷贝 → 得普通可写张量；CPU 上
  logits 已是 fp32，`.float()` 原样返回 → 仍是 inference tensor，后续 `div_` 触发
  "Inplace update to inference tensor" 报错。`copy=True` 在两端都保证拷出普通张量，GPU 无额外开销。

### 6. 无需改动的部分（已具备 fallback）
- `attention/selector.py`：按 `device_type` 选后端，CPU 自动选 SDPA。
- `attention/kv_ops.py`：`store_kvcache` 按 `key.is_cuda` 选 Triton / naive scatter。
- `block_table.py` / `input_batch.py`：`CpuGpuBuffer` 已接受 device/pin_memory 参数；`pin_memory=False`
  时 cpu 张量非 pinned、`gpu` 张量也建在 cpu，`copy_` 为 cpu→cpu。
- `layers/linear.py` / `embed_head.py`：TP 操作裹 `tp_size>1 and dist.is_initialized()`，CPU TP=1 天然跳过。

## 数据流（CPU）

```
LLM(model, device="cpu", cpu_kvcache_gb=2.0)
  → Config(device="cpu")          # 强制 eager；门控 TP=1/uni/无 swap
  → EngineCore → UniProcExecutor → Worker → CPUModelRunner(device="cpu")
       __init__: 跳过 NCCL/set_device；default_device=cpu，default_dtype=fp32
                 建模(fp32) → load_model(bf16 权重 copy_→fp32) → warmup(纯前向)
                 → allocate_kv_cache(内存=cpu_kvcache_gb) → 不录 graph
       run():    InputBatch(cpu, 非 pinned) → SDPA 注意力 + naive KV scatter
                 → Sampler(copy=True 可写 logits) → token
```

## 有意不做（与 V1 的差距）

- **CPU TP>1（gloo）**：需多进程 + gloo + MultiProc 扩展，限定 TP=1。
- **OpenMP/NUMA 线程绑定、ipex、CPU 融合 kernel、量化**：性能调优，超出"能跑"目标。
- **CPU↔GPU KV offload**：CPU 后端无 GPU，概念不成立。
- 与 [[arch_engine_proc]]（EngineCore 进程化）正交但当前门控禁用其组合（保持简单）。
