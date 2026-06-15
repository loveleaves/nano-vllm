# GPU 执行模式与调度机制：技术洞察报告

> **关键词**：CUDA Graph · Eager Mode · 算子下发 · 整图下发 · 整体下沉  
> **适用领域**：深度学习推理/训练框架、AI编译器、NPU/GPU运行时系统  

---

## 目录

1. [背景与动机](#1-背景与动机)
2. [核心名词解释](#2-核心名词解释)
   - 2.1 Eager 模式（Eager Mode）
   - 2.2 单算子下发（Per-Op Dispatch）
   - 2.3 CUDA Graph
   - 2.4 整图下发（Whole-Graph Dispatch）
   - 2.5 整体下沉（Full-Graph Offloading）
3. [技术对比分析](#3-技术对比分析)
4. [执行流水线深度剖析](#4-执行流水线深度剖析)
5. [工程实践与性能影响](#5-工程实践与性能影响)
6. [各框架实现概览](#6-各框架实现概览)
7. [选型决策指南](#7-选型决策指南)
8. [趋势与展望](#8-趋势与展望)

---

## 1. 背景与动机

在现代深度学习系统中，**Host（CPU）与 Device（GPU/NPU）之间的协调开销**是制约推理和训练性能的核心瓶颈之一。随着模型规模的急剧增长（从百亿到万亿参数），单次前向推理包含数千至数万个算子（Operator），每个算子的调度开销在总体耗时中的占比不可忽视。

### 典型的性能开销来源

```
[CPU Host]                          [GPU Device]
 │                                       │
 ├─ Python 解释器开销                     │
 ├─ 算子 dispatch 查表                    │
 ├─ 内核启动（Kernel Launch）─────────────►│ GPU Kernel 执行
 ├─ 同步等待（cudaStreamSync）            │
 └─ 内存分配（cudaMalloc / 碎片化）        │
```

为解决上述问题，业界演化出多种执行模式，形成了从"即时执行"到"整体卸载"的完整技术谱系。

---

## 2. 核心名词解释

### 2.1 Eager 模式（Eager Mode）

#### 定义

**Eager 模式**是深度学习框架中最直观的执行模式。每个算子在被调用时**立即执行**，无需构建计算图，执行结果可立即获得。PyTorch 默认工作在 Eager 模式。

#### 工作原理

```
Python 代码:  y = torch.matmul(a, b) + c
                          │
                  ┌───────▼────────┐
                  │  Python 调用栈  │
                  │  dispatch 分发  │
                  └───────┬────────┘
                          │ 立即发起 CUDA Kernel Launch
                          ▼
                  ┌───────────────┐
                  │  GPU 执行 GEMM │  ← 每次调用均同步或异步
                  └───────────────┘
                          │
                          ▼
                  ┌───────────────┐
                  │  GPU 执行 Add  │  ← 再次独立 Launch
                  └───────────────┘
```

#### 优点

- **调试友好**：逐行可查看张量值，报错定位精准
- **动态图支持**：天然支持 `if/while` 等控制流，shape 可动态变化
- **开发效率高**：符合 Python 命令式编程直觉

#### 缺点

- **Launch 开销累积**：每个算子均需独立的 CPU→GPU 调度，小算子场景下 Launch overhead 远超计算时间
- **无全局优化空间**：无法跨算子做算子融合、内存复用等图级优化
- **CPU-GPU 同步气泡**：CPU 必须持续"喂"命令给 GPU，难以充分利用 GPU 异步流水线

#### 典型场景

- 模型研究与实验阶段
- 含复杂控制流的模型（如强化学习、递归网络）
- 需要逐步调试的场景

---

### 2.2 单算子下发（Per-Op Dispatch）

#### 定义

**单算子下发**是 Eager 模式在实现层面的具体描述，指框架**逐个**将算子提交到设备执行队列（CUDA Stream / Command Queue）。它是 Eager Mode 的底层机制，但也可以出现在图模式中（未经融合的图执行）。

#### 详细流程

```
┌─────────────────────────────────────────────────────────┐
│                    单算子下发流程                          │
│                                                         │
│  Host (CPU)                     Device (GPU)            │
│  ──────────                     ─────────────           │
│  [Op1: Conv2d]                                          │
│    │ cuLaunchKernel() ──────────►[Kernel: im2col+GEMM]  │
│    │                             执行中...               │
│  [Op2: BatchNorm]                                       │
│    │ cuLaunchKernel() ──────────►[Kernel: BN forward]   │
│    │                             执行中...               │
│  [Op3: ReLU]                                            │
│    │ cuLaunchKernel() ──────────►[Kernel: elementwise]  │
│    │                             执行中...               │
│                                                         │
│  每个算子：                                               │
│    1. Host 准备参数（指针、shape、stride）                  │
│    2. 调用 cuLaunchKernel                                │
│    3. GPU 调度到 SM 执行                                  │
│    4. Host 继续下发下一个算子                              │
└─────────────────────────────────────────────────────────┘
```

#### 性能瓶颈量化

| 类型 | 典型延迟 |
|------|---------|
| CUDA Kernel Launch 延迟 | 5–20 μs |
| Python→C++ dispatch 开销 | 1–5 μs |
| 显存分配（cudaMalloc）| 50–500 μs |
| 小算子 GPU 执行时间 | 1–10 μs |

> **结论**：对于 elementwise 类小算子，**调度开销 >> 计算开销**，单算子下发是严重的性能反模式。

---

### 2.3 CUDA Graph

#### 定义

**CUDA Graph**（`cudaGraph`）是 NVIDIA 在 CUDA 10.0（2018）引入的机制，允许将一系列 CUDA 操作（Kernel Launch、内存拷贝、Event 等）**录制为一张静态图**，后续可**一次性提交整张图**执行，从而消除重复的 Launch 开销。

#### 核心机制

```
         ┌──────────── 阶段一：录制（Capture）──────────────┐
         │                                                 │
         │  cudaStreamBeginCapture(stream)                 │
         │       │                                         │
         │  [Op1 Launch] → [Op2 Launch] → [Op3 Launch]     │
         │       │              │               │           │
         │  cudaStreamEndCapture(stream, &graph)           │
         │                                                 │
         └─────────────────────────────────────────────────┘
                              │
                              ▼ 编译实例化
         ┌──────────── 阶段二：实例化（Instantiate）─────────┐
         │                                                 │
         │  cudaGraphInstantiate(&graphExec, graph, ...)   │
         │  → 生成可执行图对象（GraphExec）                   │
         │  → GPU 端预分配资源，预解析依赖                     │
         │                                                 │
         └─────────────────────────────────────────────────┘
                              │
                              ▼ 多次重放
         ┌──────────── 阶段三：重放（Replay）────────────────┐
         │                                                 │
         │  for each inference:                            │
         │    cudaGraphLaunch(graphExec, stream)  ← 1次调用 │
         │    → GPU 自主按图拓扑顺序执行所有节点              │
         │    → CPU 无需逐算子参与                           │
         │                                                 │
         └─────────────────────────────────────────────────┘
```

#### CUDA Graph 的图结构

```
CUDA Graph 节点类型：
  ├── KernelNode    → CUDA Kernel Launch
  ├── MemcpyNode    → Host↔Device / Device↔Device 内存拷贝
  ├── MemsetNode    → 内存初始化
  ├── ChildGraphNode→ 嵌套子图
  └── EventNode     → 同步事件

依赖边（Edges）：
  Node A ──depends_on──► Node B
  表示 B 必须在 A 完成后才能执行
```

#### 性能收益分析

```
传统 Eager 模式（100个算子）：
  CPU Launch 时间：100 × 10μs = 1ms（纯 Launch 开销）
  GPU 执行时间：5ms
  总耗时 ≈ 6ms（串行 Launch 导致 GPU 等待 CPU）

CUDA Graph 模式（100个算子）：
  录制：一次性，可均摊
  单次 Replay：~2μs（一次 cudaGraphLaunch）
  GPU 执行时间：5ms（相同计算量）
  总耗时 ≈ 5.002ms（几乎消除 Launch overhead）

性能提升：对 Launch-bound 场景提升可达 20-40%
```

#### 局限性

| 限制 | 说明 |
|------|------|
| **静态图约束** | 录制期间的 Kernel 参数（地址、shape）在 Replay 时必须不变 |
| **动态 shape 不支持** | 输入 shape 变化需重新录制或使用 CUDA Graph Update API |
| **控制流限制** | 不支持 CPU 侧的 if/while（GPU 侧条件节点有限支持） |
| **显存地址固定** | 中间 tensor 地址需预先固定，与动态内存分配冲突 |
| **录制开销** | 首次录制有额外耗时（通常为正式推理的 2–5 倍） |

#### 与 PyTorch 的集成

```python
# PyTorch 2.x CUDA Graph 使用示例
import torch

# 预热（建立内存地址）
s = torch.cuda.Stream()
with torch.cuda.stream(s):
    for _ in range(3):
        y = model(x)

# 录制
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    static_y = model(static_x)

# 推理阶段（高效重放）
for batch in dataloader:
    static_x.copy_(batch)
    g.replay()
    result = static_y.clone()
```

---

### 2.4 整图下发（Whole-Graph Dispatch）

#### 定义

**整图下发**是一种图执行模式：框架首先将整个计算图（或子图）**编译为一个统一的执行计划**，再将该执行计划整体提交给设备执行，而非逐算子下发。它是 AI 编译器（TVM、XLA、TensorRT）和图模式框架（`torch.compile`、TorchScript）的核心执行范式。

#### 与单算子下发的本质区别

```
单算子下发：
  计划生成              执行
  ──────────────────────────────────
  Op1 描述 → Launch → 执行
  Op2 描述 → Launch → 执行
  Op3 描述 → Launch → 执行
  （计划与执行交替进行，CPU 是瓶颈）

整图下发：
  计划生成（编译期）        执行（运行期）
  ────────────────     ─────────────────
  全图分析              一次性提交执行计划
  算子融合优化     →    GPU 自主完成所有算子
  内存规划
  调度排序
  生成执行计划
  （编译与执行分离，CPU 只需提交一次）
```

#### 关键优化能力（只有整图下发才能实现）

**1. 算子融合（Operator Fusion）**

```
融合前：Conv2d → BatchNorm → ReLU（3次 Kernel Launch）
融合后：ConvBnRelu（1次 Kernel Launch）

收益：
  - 减少 Global Memory 读写（中间结果留在寄存器/L1 Cache）
  - 减少 Kernel Launch 次数
  - 提升计算访存比（Arithmetic Intensity）
```

**2. 内存规划（Memory Planning）**

```
整图可见 → 分析所有 tensor 生命周期
         → 死亡 tensor 的显存立即复用
         → Peak Memory 可降低 30–60%
```

**3. 算子重排（Op Reordering）**

```
在保持语义正确的前提下，重新排列算子执行顺序
以最大化硬件利用率（如流水线并行、访存局部性优化）
```

**4. 常量折叠与公共子表达式消除**

```
识别计算图中的冗余计算，在编译期消除
适合推理场景（权重为常量）
```

#### 整图下发的实现层级

```
                ┌─────────────────────────┐
                │      用户 Python 代码    │
                └────────────┬────────────┘
                             │ torch.compile / tf.function
                ┌────────────▼────────────┐
                │      图表示（IR）         │
                │  (FX Graph / HLO / MLIR) │
                └────────────┬────────────┘
                             │ 编译器优化 Pass
                ┌────────────▼────────────┐
                │    优化后执行计划         │
                │  (融合算子 / 内存规划)    │
                └────────────┬────────────┘
                             │ 一次性下发
                ┌────────────▼────────────┐
                │  Device 执行（GPU/NPU）  │
                └─────────────────────────┘
```

---

### 2.5 整体下沉（Full-Graph Offloading）

#### 定义

**整体下沉**是比整图下发更彻底的执行模式：将**整个模型的计算图（包括调度逻辑、控制流、数据准备）全部迁移到设备侧（NPU/专用加速器）**，Host CPU 在模型执行期间几乎不参与，设备自主完成所有工作。

> 注：该概念在华为 MindSpore（昇腾 NPU）、昆仑芯等国产 AI 框架/芯片生态中使用尤为普遍，对应昇腾的 **"图下沉（Graph Sink）"** 特性。

#### 与整图下发的区别

| 维度 | 整图下发 | 整体下沉 |
|------|---------|---------|
| CPU 参与度 | 仍需参与图的调度控制 | CPU 几乎不参与 |
| 控制流位置 | 通常在 CPU 侧 | 控制流下沉到设备 |
| 数据传输 | 每次推理传输输入 | 可预加载数据到设备 |
| 适用场景 | 通用 GPU 推理 | 专用加速器、边端芯片 |
| 代表实现 | TensorRT、XLA | MindSpore Sink Mode、昇腾 |

#### 整体下沉的工作机制

```
┌─────────────────────────────────────────────────────────────┐
│                      整体下沉执行模式                          │
│                                                             │
│  编译期（一次性，离线）：                                       │
│  ┌─────────────────────────────────────────┐               │
│  │ 全图分析 → 编译为设备原生指令流            │               │
│  │ 控制流转换（if/while → 设备侧条件节点）    │               │
│  │ 数据预取策略规划                          │               │
│  │ 设备内存静态分配                          │               │
│  └─────────────────────────────────────────┘               │
│                                                             │
│  运行期：                                                    │
│  ┌──────────┐  发送输入数据（一次）  ┌────────────────────┐  │
│  │ Host CPU │──────────────────────►│    Device（NPU）    │  │
│  │          │                       │                    │  │
│  │  等待...  │                       │ 自主执行全部计算     │  │
│  │          │  接收输出结果（一次）   │ 包含：调度、计算、  │  │
│  │          │◄──────────────────────│ 控制流、多步迭代    │  │
│  └──────────┘                       └────────────────────┘  │
│                                                             │
│  → CPU-Device 通信次数：每次推理仅 2 次（输入+输出）            │
└─────────────────────────────────────────────────────────────┘
```

#### MindSpore 图下沉模式示例

```python
# MindSpore 整体下沉示例
import mindspore as ms
from mindspore import context

# 设置图模式 + 下沉到昇腾 NPU
context.set_context(
    mode=context.GRAPH_MODE,          # 图模式（非 Eager）
    device_target="Ascend",
    enable_graph_kernel=True          # 开启图算融合
)

# 训练时开启 dataset_sink_mode
model.train(
    epoch=10,
    train_dataset=dataset,
    dataset_sink_mode=True   # 数据集也下沉到设备，CPU完全脱离循环
)
```

#### 下沉的递进层次

```
Level 0 — 单算子下发
  CPU 主导每个算子的调度与执行

Level 1 — 子图下沉
  将热点子图（如 Attention 块）编译下沉
  非热点路径仍 Eager 执行

Level 2 — 整图下沉
  整个前向图编译下沉
  CPU 只负责下发执行计划

Level 3 — 全栈下沉（含数据流水）
  数据预处理、计算、后处理全部在设备完成
  CPU 仅作为控制器启动任务
```

---

## 3. 技术对比分析

### 3.1 核心维度对比

| 特性 | Eager / 单算子下发 | CUDA Graph | 整图下发 | 整体下沉 |
|------|-----------------|-----------|---------|---------|
| **调度开销** | 高（每算子）| 极低（一次）| 低（编译期）| 极低 |
| **优化空间** | 无 | 无（结构固定）| 大（编译优化）| 最大 |
| **动态 shape** | ✅ 原生支持 | ❌ 需重录 | ⚠️ 部分支持 | ❌ 通常不支持 |
| **动态控制流** | ✅ 完全支持 | ❌ 不支持 | ⚠️ 受限 | ⚠️ 下沉的控制流 |
| **调试友好性** | ✅ 最佳 | ❌ 黑盒 | ⚠️ 一般 | ❌ 困难 |
| **首次执行开销** | 低 | 中（录制）| 高（编译）| 极高（AOT 编译）|
| **显存效率** | 低 | 中 | 高 | 最高 |
| **CPU 占用** | 高 | 低 | 低 | 极低 |
| **适合推理** | ⚠️ | ✅ | ✅ | ✅✅ |
| **适合训练** | ✅ | ✅（固定 shape）| ✅ | ⚠️ |

### 3.2 性能水位示意

```
推理延迟（相对值，越低越好）：

单算子下发  ████████████████████ 100%
CUDA Graph  ████████████░░░░░░░░  60%（Launch-bound 场景）
整图下发    ██████████░░░░░░░░░░  50%（含算子融合）
整体下沉    ████████░░░░░░░░░░░░  40%（专用硬件最优化）

注：实际提升高度依赖模型结构与硬件特性
```

---

## 4. 执行流水线深度剖析

### 4.1 现代 GPU 执行流水线

```
┌──────────────────────────────────────────────────────────────┐
│                  GPU 软件执行栈                                │
│                                                              │
│  Application Layer                                           │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  PyTorch / TensorFlow / JAX / MindSpore              │   │
│  └──────────────────────────┬─────────────────────────┘    │
│                             │                               │
│  Runtime Layer              ▼                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  CUDA Runtime / ROCm / CANN                          │   │
│  │   ├─ Stream 管理                                     │   │
│  │   ├─ Graph 管理                                      │   │
│  │   └─ 内存管理（Allocator / Cache）                    │   │
│  └──────────────────────────┬─────────────────────────┘    │
│                             │                               │
│  Driver Layer               ▼                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  GPU Driver（KMD + UMD）                             │   │
│  │   ├─ Command Buffer 构建                              │   │
│  │   ├─ 硬件调度（GPC / SM 分配）                        │   │
│  │   └─ PCIe / NVLink 传输                              │   │
│  └──────────────────────────┬─────────────────────────┘    │
│                             │                               │
│  Hardware Layer             ▼                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  GPU 硬件                                            │   │
│  │   ├─ SM（Streaming Multiprocessors）                 │   │
│  │   ├─ Tensor Core / CUDA Core                        │   │
│  │   ├─ L1/L2 Cache                                    │   │
│  │   └─ HBM / GDDR（Global Memory）                    │   │
│  └──────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────┘
```

### 4.2 CUDA Graph 内部节点依赖图示例（ResNet Block）

```
  [Input]
     │
     ▼
[Conv2d_1] ────────────────────────────────┐
     │                                      │
     ▼                                      │
[BatchNorm_1]                               │ (Skip Connection)
     │                                      │
     ▼                                      │
[ReLU_1]                                    │
     │                                      │
     ▼                                      │
[Conv2d_2]                                  │
     │                                      │
     ▼                                      │
[BatchNorm_2]                               │
     │                                      ▼
     └──────────────────────────────► [Add] ──► [ReLU_2] ──► [Output]

CUDA Graph 捕获后：
  以上所有节点成为 Graph 节点
  依赖边由 CUDA 运行时自动推断
  重放时 GPU 自主并发调度无依赖节点
```

---

## 5. 工程实践与性能影响

### 5.1 何时 CUDA Graph 收益最大

```
收益高的场景：
  ✅ 模型 shape 固定（如 batch_size=1 的在线推理）
  ✅ 算子数量多（>100）且有大量小算子
  ✅ 推理延迟敏感（P99 latency 优化）
  ✅ throughput-bound 服务（高 QPS 场景）

收益低或不适用的场景：
  ❌ 动态 shape（如 NLP 序列长度可变）
  ❌ 含 Python 控制流（每 step 图结构不同）
  ❌ 训练阶段（梯度计算图结构每步可能变化）
  ❌ 算子数少且均为大算子（Launch overhead 不是瓶颈）
```

### 5.2 整图下发的编译开销

```
torch.compile 编译耗时（参考量级）：

  小模型（<10M 参数）：   10–30 秒
  中等模型（100M 参数）： 30–120 秒
  大模型（10B 参数）：    数分钟

加速比（编译后 vs Eager）：
  CV 模型（ResNet/ViT）：  1.3–2.5×
  LLM 推理（decode 阶段）：1.2–1.8×
  LLM 训练（前向+反向）：  1.5–3.0×
```

### 5.3 内存管理的影响

```
Eager 模式内存行为：
  Op1 输出 → 分配 A
  Op2 输出 → 分配 B（A 可能已可释放但未释放）
  Op3 输出 → 分配 C
  → Peak Memory 高（allocator 惰性释放）

整图下发内存行为：
  编译器分析全局 liveness
  → A 在 Op2 输入后立即标记可复用
  → Op3 输出直接复用 A 的内存块
  → Peak Memory 可降低 40–60%
```

---

## 6. 各框架实现概览

### 6.1 PyTorch 的执行模式演进

```
PyTorch 1.x（主要）
  └─ Eager Mode（默认）
  └─ TorchScript（静态图，有限使用）

PyTorch 2.x（当前主流）
  └─ Eager Mode（默认，兼容性最佳）
  └─ torch.compile（整图下发，基于 TorchDynamo + Inductor）
       ├─ fullgraph=True：强制整图编译
       ├─ mode="reduce-overhead"：自动使用 CUDA Graph
       └─ mode="max-autotune"：最激进的编译优化
  └─ CUDA Graphs API（手动或自动）
```

### 6.2 TensorFlow / JAX

```
TensorFlow 2.x：
  └─ Eager Mode（默认）
  └─ @tf.function（整图下发，基于 XLA 编译）
       └─ jit_compile=True → XLA 编译优化

JAX：
  └─ 默认 Eager
  └─ jax.jit（整图下发）
  └─ jax.vmap / jax.pmap（向量化 + 并行化）
  XLA 是 JAX 的核心，天然整图编译
```

### 6.3 MindSpore（华为昇腾生态）

```
MindSpore 执行模式：
  └─ PyNative Mode（Eager，类 PyTorch）
  └─ Graph Mode（整图下发 + 整体下沉）
       ├─ 静态图编译（ME/GE 编译器）
       ├─ 图算融合（GraphKernel Fusion）
       ├─ dataset_sink_mode=True → 数据集下沉
       └─ 模型并行图切分（流水线并行）
  昇腾 NPU 的 CANN 框架原生支持整体下沉
```

### 6.4 TensorRT（推理专用）

```
TensorRT：
  └─ 完全静态图编译（整图下发的极致实现）
  └─ 自动 Layer Fusion（Conv+BN+ReLU → 1 Kernel）
  └─ INT8/FP16 量化（编译期完成）
  └─ Kernel Auto-Tuning（选最优 GEMM 实现）
  └─ 显存零拷贝（绑定输入输出指针）
  → 代表推理场景整图下发的最高优化水位
```

---

## 7. 选型决策指南

### 7.1 决策树

```
                    ┌─────────────────────────────┐
                    │        你的主要目标？         │
                    └──────────────┬──────────────┘
                                   │
              ┌────────────────────┼───────────────────┐
              ▼                    ▼                   ▼
         研究 / 调试           生产推理             生产训练
              │                    │                   │
              ▼                    │                   ▼
       Eager Mode             ┌────▼────┐        torch.compile
    （最佳调试体验）           │Shape?   │        + Eager（混合）
                              └────┬────┘
                        ┌──────────┴──────────┐
                        ▼                     ▼
                    静态 Shape           动态 Shape
                        │                     │
              ┌─────────┼──────┐        torch.compile
              ▼         ▼      ▼        （dynamic=True）
           延迟?      内存?   硬件?
              │         │      │
           极低延迟   内存受限  NPU
              │         │      │
         CUDA Graph  整图下发  整体下沉
```

### 7.2 推荐配置

| 场景 | 推荐配置 |
|------|---------|
| 在线推理（fixed shape, low latency）| TensorRT 或 torch.compile(mode="reduce-overhead") + CUDA Graph |
| 在线推理（dynamic shape）| torch.compile(dynamic=True) |
| 离线批量推理（large batch）| TensorRT FP16/INT8 |
| LLM 推理（prefill）| Eager / torch.compile（prefill shape 多变）|
| LLM 推理（decode）| CUDA Graph（decode shape 固定）|
| 模型训练（研究阶段）| Eager Mode |
| 模型训练（生产加速）| torch.compile + gradient checkpointing |
| 昇腾 NPU 生产部署 | MindSpore Graph Mode + 整体下沉 |

---

## 8. 趋势与展望

### 8.1 动态图与静态图的融合

当前主流方向是**"Eager 开发，Compile 部署"**的混合范式：
- 开发时使用 Eager，保持调试便利性
- 部署时通过 `torch.compile`、`tf.function` 自动转换为静态图
- **Dynamo 技术**（PyTorch 2.0 核心）：通过字节码分析实现透明的图捕获，在不改变代码的前提下启用整图优化

### 8.2 编译器基础设施的统一

```
MLIR 生态（Multi-Level IR）：
  HLO → StableHLO → MLIR Linalg → LLVM IR → PTX/XLA
  统一不同框架的 IR 表示，实现跨框架优化复用
```

### 8.3 LLM 场景的特殊挑战

```
LLM 推理的两阶段特性：
  Prefill 阶段：输入 token 数量可变 → 动态 shape → 难以 CUDA Graph
  Decode 阶段：每步生成 1 token → 固定 shape → CUDA Graph 收益极大

主流 LLM 推理框架（vLLM、TGI、TensorRT-LLM）的策略：
  Prefill：整图编译 + 动态 shape 支持（cudnn frontend / Flash Attention）
  Decode：CUDA Graph（针对常见 batch_size 预录制多个 Graph）
  → "CUDAGraph with Paged Attention" 是当前 SOTA 方案
```

### 8.4 硬件专用指令集的挑战

随着 NPU、TPU、专用 AI 芯片的普及，"整体下沉"将成为更主流的范式：
- 软件定义硬件（如 Cerebras WSE、Graphcore IPU）天然面向整图计算
- 编译器需要处理更复杂的内存层级（SRAM / DRAM / HBM）
- **数据流架构（Dataflow Architecture）** 将整体下沉推向极限——数据不回 CPU，在芯片内部流动完成整个模型计算

---

## 附录：关键术语速查

| 术语 | 英文 | 核心含义 |
|------|------|---------|
| Eager 模式 | Eager Mode | 算子即时执行，无图结构 |
| 单算子下发 | Per-Op Dispatch | 逐个提交算子到设备 |
| CUDA Graph | CUDA Graph | 录制→重放，消除重复 Launch |
| 整图下发 | Whole-Graph Dispatch | 编译整图，一次提交执行 |
| 整体下沉 | Full-Graph Offloading | 全部计算迁移到设备，CPU 退出循环 |
| 算子融合 | Operator Fusion | 多算子合并为单 Kernel |
| 内存规划 | Memory Planning | 全局 tensor 生命周期分析与复用 |
| Kernel Launch | Kernel Launch | CPU 向 GPU 提交 Kernel 执行请求 |
| CUDA Stream | CUDA Stream | GPU 异步命令队列 |
| 图下沉 | Graph Sink | MindSpore 的整体下沉实现 |

---

*报告作者：技术洞察团队 | 参考框架版本：PyTorch 2.4 / MindSpore 2.3 / CUDA 12.x*