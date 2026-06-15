# 大模型算子适配手册

## 目录

### Part I：通用算子适配技术

1. [概述与背景](#1-概述与背景)
2. [算子基础知识体系](#2-算子基础知识体系)
3. [算子开发流程](#3-算子开发流程)
4. [算子优化技术](#4-算子优化技术)
5. [算子适配技术](#5-算子适配技术)
6. [主流硬件平台适配指南](#6-主流硬件平台适配指南)
7. [算子调试与性能分析](#7-算子调试与性能分析)
8. [常见问题与解决方案](#8-常见问题与解决方案)
9. [附录：参考工具链与资源](#9-附录参考工具链与资源)

### Part II：vLLM / vLLM-Ascend 算子适配专题

10. [vLLM 框架架构深度解析](#10-vllm-框架架构深度解析)
11. [vLLM 核心算子体系](#11-vllm-核心算子体系)
12. [vLLM GPU 算子开发与优化](#12-vllm-gpu-算子开发与优化)
13. [vLLM Attention 后端体系](#13-vllm-attention-后端体系)
14. [vLLM-Ascend 插件架构与算子适配](#14-vllm-ascend-插件架构与算子适配)
15. [vLLM-Ascend 自定义算子开发实战](#15-vllm-ascend-自定义算子开发实战)
16. [GPU ↔ NPU 算子迁移方法论](#16-gpu--npu-算子迁移方法论)
17. [torch.compile 与 CUDA Graph 适配](#17-torchcompile-与-cuda-graph-适配)
18. [分布式算子与通信适配](#18-分布式算子与通信适配)
19. [量化算子在 vLLM 中的适配](#19-量化算子在-vllm-中的适配)
20. [端到端适配实战案例](#20-端到端适配实战案例)
21. [适配工程最佳实践与 Checklist](#21-适配工程最佳实践与-checklist)

---

## 1. 概述与背景

### 1.1 什么是算子

**算子（Operator / Kernel）** 是深度学习框架中最基本的计算单元，对应神经网络中的一个数学运算，例如矩阵乘法（MatMul）、卷积（Conv）、层归一化（LayerNorm）、注意力机制（Attention）等。

在大模型（LLM）场景下，算子的特征如下：

| 特征 | 描述 |
|------|------|
| **规模超大** | 参数量 7B～数百 B，单算子输入张量可达 GB 级 |
| **访存密集** | Transformer 中大量 ElementWise 算子为内存带宽瓶颈 |
| **计算密集** | MatMul、Attention 等算子为算力瓶颈 |
| **序列依赖** | 自回归推理存在严重的序列依赖，并行度受限 |
| **精度敏感** | 量化、混合精度对模型精度影响显著 |

### 1.2 大模型算子适配的挑战

```
训练框架 (PyTorch/JAX)
        │
        ▼
    模型导出 (ONNX/TorchScript/SafeTensors)
        │
        ▼
  推理框架 (TensorRT/vLLM/TNN/MNN/ONNX Runtime)
        │
        ▼
  硬件后端 (CUDA/ROCm/NPU/MLU/XPU/CPU)
        │
        ▼
    部署产品
```

核心挑战在于：
- 不同训练框架的算子语义差异
- 不同硬件架构的指令集与存储层次结构差异
- 量化感知训练与推理量化的对齐
- 动态 Shape 带来的编译复杂性
- 新硬件上缺少高性能算子库

### 1.3 本手册适用范围

本手册面向以下场景：

- 在自研 AI 芯片（NPU/MLU/XPU）上适配开源大模型
- 为推理框架（vLLM、TensorRT-LLM、MindIE 等）开发/优化算子
- 模型迁移：从 A 硬件平台迁移到 B 硬件平台
- 量化算子开发：INT8/INT4/FP8/FP16 混合精度推理

---

## 2. 算子基础知识体系

### 2.1 大模型核心算子分类

#### 2.1.1 计算密集型算子

| 算子名称 | 数学描述 | 特点 |
|----------|----------|------|
| **MatMul/GEMM** | `C = A × B` | 算力瓶颈，roofline 模型的顶点 |
| **BatchMatMul** | `C[b] = A[b] × B[b]` | 批量矩阵乘，注意力计算核心 |
| **Conv2D** | 滑动窗口卷积 | 视觉模型基础算子（多模态场景） |
| **Flash Attention** | Tiled Softmax + Matmul | 内存访问优化版注意力 |
| **Grouped Query Attention（GQA）** | KV 头分组共享 | 推理加速关键 |

#### 2.1.2 访存密集型算子

| 算子名称 | 典型场景 | 优化要点 |
|----------|----------|----------|
| **LayerNorm / RMSNorm** | Transformer 每层归一化 | 向量化 + Fuse |
| **Softmax** | 注意力权重归一化 | online softmax，防数值溢出 |
| **Embedding Lookup** | token 嵌入 | 稀疏访存，Cache 友好 |
| **RoPE（旋转位置编码）** | 位置信息编码 | 可与 Q/K 矩阵乘融合 |
| **Activation（SiLU/GeLU/ReLU）** | 非线性变换 | 逐元素，带宽敏感 |
| **Elementwise（Add/Mul/...）** | 残差连接等 | Fuse 多个操作减少读写 |

#### 2.1.3 控制/辅助类算子

| 算子名称 | 用途 |
|----------|------|
| **KV Cache 管理** | PagedAttention，显存池化管理 |
| **Beam Search / Sampling** | 解码策略 |
| **LoRA Merge** | 低秩适配推理 |
| **Quantize / Dequantize** | 量化/反量化节点 |
| **Gather / Scatter** | MoE 路由、序列处理 |
| **MoE Router + Dispatch** | 混合专家模型路由 |

### 2.2 Roofline 模型与性能分析基础

```
性能 (FLOP/s)
     |
峰值 |----------------------- 计算瓶颈区
算力 |                      /
     |                    /
     |                  /
     |      访存瓶颈区 /
     |              /
     |____________/___________________
                               算术强度 (FLOP/Byte)
```

**算术强度（Arithmetic Intensity）** = 计算量 / 访存量

- MatMul（N=4096）：~4096 FLOP/Byte → **计算瓶颈**
- LayerNorm、Softmax：< 10 FLOP/Byte → **访存瓶颈**
- 优化原则：计算密集型追求利用率，访存密集型追求融合减少 I/O

### 2.3 张量存储格式

```python
# 常见存储格式
# NCHW（PyTorch 默认，卷积友好）
tensor.shape = [Batch, Channel, Height, Width]

# NHWC（TensorFlow 默认，部分 NPU 友好）
tensor.shape = [Batch, Height, Width, Channel]

# 大模型中常见格式
# [seq_len, batch, hidden_dim]  - 序列优先
# [batch, seq_len, hidden_dim]  - 批次优先（更常见）
# [batch, num_heads, seq_len, head_dim]  - 注意力中间态
```

> **注意**：格式转换（Transpose）本身就是性能热点，适配时需最小化格式转换次数，或将格式转换融合进相邻算子。

---

## 3. 算子开发流程

### 3.1 标准开发流程

```
需求分析
   │
   ▼
接口设计（算子规格定义）
   │
   ▼
参考实现（CPU/Python）
   │
   ▼
单元测试（精度验证）
   │
   ▼
硬件实现（CUDA/HIP/自定义汇编）
   │
   ▼
性能 Profiling
   │
   ▼
迭代优化
   │
   ▼
集成测试（框架集成 + 端到端验证）
   │
   ▼
文档 + 上线
```

### 3.2 算子规格定义

在开发前必须明确以下规格：

```yaml
# 算子规格文档示例：RMSNorm
operator:
  name: rms_norm
  description: "Root Mean Square Layer Normalization"
  
  inputs:
    - name: x
      dtype: [float16, bfloat16, float32]
      shape: "[..., hidden_size]"  # 任意前缀维度
    - name: weight
      dtype: [float16, bfloat16, float32]
      shape: "[hidden_size]"
  
  outputs:
    - name: y
      dtype: "same as x"
      shape: "same as x"
  
  attributes:
    - name: eps
      type: float
      default: 1e-6
      description: "防止除零的小量"
  
  math: |
    y = x / sqrt(mean(x^2) + eps) * weight
  
  constraints:
    - "hidden_size >= 1"
    - "x.dtype == weight.dtype (or weight is fp32 for mixed-precision)"
  
  reference_impl: "transformers/models/llama/modeling_llama.py::LlamaRMSNorm"
```

### 3.3 参考实现（Python）

```python
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    """RMSNorm 参考实现，用于精度验证基准"""
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 计算均方根
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


def test_rms_norm_reference():
    """参考实现精度测试"""
    batch, seq, hidden = 2, 512, 4096
    x = torch.randn(batch, seq, hidden)
    
    norm = RMSNorm(hidden)
    y_ref = norm(x)
    
    # 对比 HuggingFace 实现
    from transformers.models.llama.modeling_llama import LlamaRMSNorm
    norm_hf = LlamaRMSNorm(hidden)
    norm_hf.weight = norm.weight
    y_hf = norm_hf(x)
    
    max_diff = (y_ref - y_hf).abs().max().item()
    print(f"Max diff with HF impl: {max_diff:.2e}")
    assert max_diff < 1e-5, "参考实现与 HF 实现不一致"
```

### 3.4 CUDA 核函数开发

以 RMSNorm 为例，展示 CUDA 核函数开发全流程：

```cuda
// rms_norm_kernel.cu

#include <cuda_fp16.h>
#include <cuda_bf16.h>

// ============================================================
// Step 1: Warp-level reduce
// ============================================================
template <typename T>
__device__ __forceinline__ T warp_reduce_sum(T val) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

// ============================================================
// Step 2: Block-level reduce（使用 shared memory）
// ============================================================
template <typename T, int BLOCK_SIZE>
__device__ T block_reduce_sum(T val, T* shared) {
    int lane = threadIdx.x % 32;
    int wid  = threadIdx.x / 32;
    
    val = warp_reduce_sum(val);
    
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    
    val = (threadIdx.x < BLOCK_SIZE / 32) ? shared[lane] : T(0);
    if (wid == 0) val = warp_reduce_sum(val);
    
    return val;
}

// ============================================================
// Step 3: RMSNorm Kernel（float16 版本）
// ============================================================
template <int BLOCK_SIZE, int VEC_SIZE>
__global__ void rms_norm_kernel_fp16(
    const __half* __restrict__ x,        // [rows, cols]
    const __half* __restrict__ weight,   // [cols]
    __half* __restrict__ out,            // [rows, cols]
    int rows,
    int cols,
    float eps
) {
    extern __shared__ float shared_mem[];
    
    const int row = blockIdx.x;
    if (row >= rows) return;
    
    const __half* x_row = x + row * cols;
    __half* out_row     = out + row * cols;
    
    // ---- 1. 向量化加载 + 计算 sum of squares ----
    float sum_sq = 0.0f;
    
    using Vec = float4;  // 8x fp16 = 128bit load
    int vec_cols = cols / (VEC_SIZE * 2);  // fp16x8 per iteration
    
    for (int i = threadIdx.x; i < vec_cols; i += BLOCK_SIZE) {
        // 128-bit 向量化加载
        const Vec* x_vec = reinterpret_cast<const Vec*>(x_row) + i;
        __half2* h2 = reinterpret_cast<__half2*>(const_cast<Vec*>(x_vec));
        
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            float2 f2 = __half22float2(h2[j]);
            sum_sq += f2.x * f2.x + f2.y * f2.y;
        }
    }
    
    // ---- 2. Block-level reduce ----
    sum_sq = block_reduce_sum<float, BLOCK_SIZE>(sum_sq, shared_mem);
    
    float rms_inv = rsqrtf(sum_sq / cols + eps);
    
    // ---- 3. 向量化写出 ----
    for (int i = threadIdx.x; i < vec_cols; i += BLOCK_SIZE) {
        const Vec* x_vec    = reinterpret_cast<const Vec*>(x_row) + i;
        const Vec* w_vec    = reinterpret_cast<const Vec*>(weight) + i;
        Vec*       out_vec  = reinterpret_cast<Vec*>(out_row) + i;
        
        __half2* xh2  = reinterpret_cast<__half2*>(const_cast<Vec*>(x_vec));
        __half2* wh2  = reinterpret_cast<__half2*>(const_cast<Vec*>(w_vec));
        __half2  oh2[4];
        
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            float2 xf = __half22float2(xh2[j]);
            float2 wf = __half22float2(wh2[j]);
            oh2[j] = __float22half2_rn(
                make_float2(xf.x * rms_inv * wf.x,
                            xf.y * rms_inv * wf.y)
            );
        }
        *out_vec = *reinterpret_cast<Vec*>(oh2);
    }
    
    // 处理尾部（cols 不整除 VEC_SIZE 时）
    int tail_start = vec_cols * VEC_SIZE * 2;
    for (int i = tail_start + threadIdx.x; i < cols; i += BLOCK_SIZE) {
        float xi = __half2float(x_row[i]);
        float wi = __half2float(weight[i]);
        out_row[i] = __float2half(xi * rms_inv * wi);
    }
}

// ============================================================
// Step 4: 启动函数
// ============================================================
void launch_rms_norm_fp16(
    const void* x, const void* weight, void* out,
    int rows, int cols, float eps, cudaStream_t stream
) {
    constexpr int BLOCK_SIZE = 256;
    constexpr int VEC_SIZE   = 8;  // fp16 x8 = 128bit
    
    // shared memory = BLOCK_SIZE/32 floats（用于 block reduce）
    int smem_size = (BLOCK_SIZE / 32) * sizeof(float);
    
    rms_norm_kernel_fp16<BLOCK_SIZE, VEC_SIZE>
        <<<rows, BLOCK_SIZE, smem_size, stream>>>(
            (const __half*)x, (const __half*)weight, (__half*)out,
            rows, cols, eps
        );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in rms_norm: %s\n", cudaGetErrorString(err));
    }
}
```

### 3.5 算子注册（以 PyTorch Extension 为例）

```python
# setup.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='custom_ops',
    ext_modules=[
        CUDAExtension(
            name='custom_ops',
            sources=[
                'csrc/pybind.cpp',
                'csrc/rms_norm.cu',
                'csrc/flash_attention.cu',
            ],
            extra_compile_args={
                'cxx': ['-O3', '-std=c++17'],
                'nvcc': [
                    '-O3',
                    '-arch=sm_80',  # A100
                    '--use_fast_math',
                    '-Xptxas', '-v',  # 查看寄存器使用
                ]
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
```

```cpp
// csrc/pybind.cpp
#include <torch/extension.h>

// 声明
torch::Tensor rms_norm_fp16(
    torch::Tensor x,
    torch::Tensor weight,
    float eps
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rms_norm_fp16", &rms_norm_fp16,
          "RMSNorm FP16 CUDA implementation");
}
```

```python
# 使用
import custom_ops
import torch

x = torch.randn(2, 512, 4096, dtype=torch.float16, device='cuda')
weight = torch.ones(4096, dtype=torch.float16, device='cuda')
out = custom_ops.rms_norm_fp16(x, weight, 1e-6)
```

---

## 4. 算子优化技术

### 4.1 算子融合（Operator Fusion）

算子融合是将多个相邻算子合并为一个 Kernel 执行，核心收益是减少显存读写次数。

#### 4.1.1 典型融合模式

```
# 模式1: LayerNorm + Linear 融合
x → LayerNorm → Linear  =====>  x → FusedLayerNormLinear
节省: LayerNorm 结果无需写回 HBM

# 模式2: Add + RMSNorm 融合（残差 + 归一化）  
(x, residual) → Add → RMSNorm  =====>  (x, residual) → FusedAddRMSNorm
节省: Add 结果无需写回 HBM，同时输出归一化后的值和残差和

# 模式3: QKV 投影融合
q = linear_q(x)
k = linear_k(x)
v = linear_v(x)
======>
q, k, v = fused_qkv_linear(x)  # 单次 GEMM，合并 W_q/W_k/W_v

# 模式4: SiLU Gate MLP 融合（LLaMA-style FFN）
gate = linear_gate(x)
up   = linear_up(x)
out  = silu(gate) * up
======>
gate, up = fused_gate_up_linear(x)
out = fused_silu_mul(gate, up)
```

#### 4.1.2 融合收益估算

```python
def estimate_fusion_benefit(hidden_size, seq_len, batch_size):
    """估算 Add + RMSNorm 融合的带宽节省"""
    dtype_bytes = 2  # FP16
    total_elements = batch_size * seq_len * hidden_size
    
    # 不融合：Add 写一次，RMSNorm 读一次
    unfused_io = 2 * total_elements * dtype_bytes  # bytes
    
    # 融合：只读两次输入，写一次输出
    fused_io = 3 * total_elements * dtype_bytes  # bytes
    
    saving_ratio = 1 - fused_io / (unfused_io + fused_io)
    print(f"融合节省带宽比例: {saving_ratio:.1%}")
    
estimate_fusion_benefit(4096, 2048, 8)
# Output: 融合节省带宽比例: 40.0%
```

### 4.2 内存访问优化

#### 4.2.1 向量化内存访问

```cuda
// ❌ 低效：逐元素加载（4B per transaction）
float val = x[idx];

// ✅ 高效：128bit 向量加载（16B per transaction，4x 吞吐）
float4 val = *reinterpret_cast<const float4*>(x + idx);

// ✅ FP16 对应 __half2（32bit） 或 8x __half（128bit）
__half2 val = *reinterpret_cast<const __half2*>(x + idx);  // 32bit
```

#### 4.2.2 Shared Memory 使用

```cuda
// 避免 Bank Conflict 的 Shared Memory Padding 技巧
// 32 banks, each 4 bytes; 相邻线程访问相邻 bank = 无冲突
__shared__ float smem[BLOCK_SIZE + 1];  // +1 padding 避免 2-way bank conflict
```

#### 4.2.3 Flash Attention 的分块策略

```
传统 Attention 显存复杂度: O(N²)，N=seq_len
Flash Attention 显存复杂度: O(N)

核心思想：将 Q/K/V 分块加载进 SRAM，
在 SRAM 内完成 softmax + matmul，
避免将中间的 N×N 注意力矩阵写回 HBM。

分块参数选择：
  - Block size M_r（Q 分块行数）
  - Block size N_c（K/V 分块列数）
  - 约束: M_r * d + N_c * d ≤ SRAM_size（d=head_dim）
  - A100 SRAM ≈ 192KB per SM
```

### 4.3 量化优化

#### 4.3.1 量化方案对比

| 方案 | 精度损失 | 压缩比 | 硬件支持 | 适用场景 |
|------|----------|--------|----------|----------|
| **FP16** | 无 | 2x vs FP32 | 广泛 | 基线 |
| **BF16** | 极小 | 2x vs FP32 | A100/H100/新 NPU | 训练 + 推理 |
| **FP8 (E4M3/E5M2)** | 小 | 4x vs FP32 | H100/H800 | GEMM 计算 |
| **INT8（W8A8）** | 小～中 | 4x vs FP32 | 广泛 | 推理 GEMM |
| **INT8（W8A16）** | 小 | 权重 4x | 广泛 | 访存受限推理 |
| **INT4（GPTQ/AWQ）** | 中 | 8x vs FP32 | 广泛 | 7B～70B 压缩推理 |
| **INT4（W4A8）** | 中 | 8x 权重 + 推理 INT8 | 部分 | 极致压缩 |

#### 4.3.2 INT8 量化算子开发要点

```python
# SmoothQuant 风格的 W8A8 量化
# 核心公式: Y = X_int8 * W_int8 * (scale_x * scale_w)

class QuantLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # 量化权重 (int8)
        self.weight_int8 = nn.Parameter(
            torch.zeros(out_features, in_features, dtype=torch.int8),
            requires_grad=False
        )
        # Per-channel 权重缩放因子
        self.weight_scale = nn.Parameter(
            torch.ones(out_features, dtype=torch.float32),
            requires_grad=False
        )
        # 激活值缩放因子（动态或静态）
        self.act_scale = None  # 运行时确定
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. 动态量化激活值
        act_scale = x.abs().max(dim=-1, keepdim=True)[0] / 127.0
        x_int8 = (x / act_scale).round().clamp(-128, 127).to(torch.int8)
        
        # 2. INT8 矩阵乘（需要硬件 INT8 指令）
        # 实际生产中调用 CUTLASS int8 GEMM 或 cuBLAS
        out_int32 = torch._int_mm(x_int8, self.weight_int8.t())
        
        # 3. 反量化
        out = out_int32.float() * (act_scale * self.weight_scale.unsqueeze(0))
        return out
```

#### 4.3.3 量化校准流程

```python
def calibrate_quantization(model, calib_dataloader, num_batches=512):
    """
    量化校准：收集激活值分布，确定量化参数
    支持：MinMax / AbsMax / Percentile / Histogram
    """
    model.eval()
    hooks = []
    activation_stats = {}
    
    def make_hook(name):
        def hook(module, input, output):
            if name not in activation_stats:
                activation_stats[name] = []
            # 收集激活值统计
            activation_stats[name].append({
                'min': output.min().item(),
                'max': output.max().item(),
                'abs_max': output.abs().max().item(),
                'p99': torch.quantile(output.abs().float(), 0.99).item(),
            })
        return hook
    
    # 注册 hook
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(make_hook(name)))
    
    with torch.no_grad():
        for i, batch in enumerate(calib_dataloader):
            if i >= num_batches:
                break
            model(**batch)
    
    # 移除 hook
    for h in hooks:
        h.remove()
    
    # 计算最终量化参数
    quant_params = {}
    for name, stats in activation_stats.items():
        quant_params[name] = {
            'scale': max(s['p99'] for s in stats) / 127.0,
            'zero_point': 0,  # 对称量化
        }
    
    return quant_params
```

### 4.4 Tensor Parallelism 下的算子切分

```
Megatron-LM 风格的列/行切分

列切分（Column Parallel）:
  W = [W1 | W2]  (按列切分)
  Y = X @ W = [X@W1 | X@W2]
  → 每个设备持有 W 的一列子块
  → 输入 X 每个设备相同（AllReduce-free for forward）
  → 输出 Y 需要 All-Gather

行切分（Row Parallel）:
  W = [W1; W2]  (按行切分)
  Y = X @ W = X1@W1 + X2@W2  （需要 AllReduce）
  → 每个设备持有 W 的一行子块
  → 输入 X 按列切分分发
  → 输出 Y 需要 AllReduce
```

```python
# Tensor Parallelism 算子实现示例（使用 torch.distributed）
import torch.distributed as dist

class ColumnParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, tp_degree):
        super().__init__()
        assert out_features % tp_degree == 0
        self.local_out = out_features // tp_degree
        self.weight = nn.Parameter(
            torch.randn(self.local_out, in_features) / (in_features ** 0.5)
        )
    
    def forward(self, x):
        # 每个设备计算本地输出
        return F.linear(x, self.weight)
        # 不需要 All-Reduce（配合 RowParallel 使用）

class RowParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, tp_degree):
        super().__init__()
        assert in_features % tp_degree == 0
        self.local_in = in_features // tp_degree
        self.weight = nn.Parameter(
            torch.randn(out_features, self.local_in) / (self.local_in ** 0.5)
        )
    
    def forward(self, x):
        # 每个设备计算本地部分
        local_out = F.linear(x, self.weight)
        # AllReduce 聚合各设备结果
        dist.all_reduce(local_out, op=dist.ReduceOp.SUM)
        return local_out
```

### 4.5 Continuous Batching 与 PagedAttention

```
传统 Static Batching 问题：
  - 所有序列 padding 到同一长度，浪费算力
  - KV Cache 按最大长度预分配，浪费显存

Continuous Batching（vLLM）：
  - 动态调度，序列完成后立即加入新请求
  - 无需等待整个 batch 完成

PagedAttention：
  - KV Cache 分页管理（类 OS 内存分页）
  - 每页固定大小（如 16 tokens × head_dim）
  - 物理不连续但逻辑连续，减少碎片
  - 实现 KV Cache 在请求间的共享（prefix caching）

算子适配要点：
  - Attention 算子需支持非连续 KV Cache 输入
  - 需要 block_table 参数映射逻辑地址到物理地址
  - 实现 paged attention kernel（参考 vLLM 源码）
```

---

## 5. 算子适配技术

### 5.1 算子适配流程总览

```
Step 1: 模型分析
  └─ 统计算子类型和频次
  └─ 识别不支持/低效算子
  └─ 建立优先级列表

Step 2: 算子 Gap 分析
  └─ 框架原生支持 → 直接使用
  └─ 框架支持但需配置 → 参数适配
  └─ 框架不支持 → 需要开发新算子
  └─ 有等价替代 → 算子 Fallback 或等效替换

Step 3: 适配实现
  └─ 新算子开发（见第 3 章）
  └─ 算子注册到目标框架

Step 4: 精度验证
  └─ 单算子精度（vs 参考实现）
  └─ 层级精度（每层输出对比）
  └─ 模型级精度（PPL / 任务评测）

Step 5: 性能调优
  └─ Profiling 瓶颈定位
  └─ 迭代优化

Step 6: 集成测试 & 上线
```

### 5.2 算子 Gap 分析方法

```python
import onnx
from collections import Counter

def analyze_operator_gap(onnx_model_path: str, supported_ops: set):
    """
    分析 ONNX 模型中的算子 Gap
    
    Args:
        onnx_model_path: ONNX 模型路径
        supported_ops: 目标框架已支持的算子集合
    
    Returns:
        gap_report: 未支持算子报告
    """
    model = onnx.load(onnx_model_path)
    
    # 统计所有算子
    all_ops = Counter()
    unsupported_ops = {}
    
    for node in model.graph.node:
        op_type = node.op_type
        all_ops[op_type] += 1
        
        if op_type not in supported_ops:
            if op_type not in unsupported_ops:
                unsupported_ops[op_type] = {
                    'count': 0,
                    'nodes': []
                }
            unsupported_ops[op_type]['count'] += 1
            unsupported_ops[op_type]['nodes'].append(node.name)
    
    # 生成报告
    print("=" * 60)
    print("算子 Gap 分析报告")
    print("=" * 60)
    print(f"总算子数: {sum(all_ops.values())}")
    print(f"算子类型数: {len(all_ops)}")
    print(f"不支持的算子类型数: {len(unsupported_ops)}")
    print()
    
    if unsupported_ops:
        print("不支持的算子（按频次排序）:")
        for op, info in sorted(unsupported_ops.items(),
                               key=lambda x: -x[1]['count']):
            print(f"  {op:30s} 出现 {info['count']:4d} 次")
    else:
        print("✅ 所有算子均已支持")
    
    return unsupported_ops


# 示例：LLaMA-7B 的常见算子集
LLAMA_OPS = {
    'MatMul', 'Add', 'Mul', 'Transpose', 'Reshape', 'Gather',
    'Unsqueeze', 'Softmax', 'Concat', 'Split', 'Sqrt',
    'Div', 'Expand', 'Cast', 'Where', 'Slice', 'Pow',
    'ReduceMean', 'Sub', 'Tanh', 'Erf', 'Sigmoid',
    # LLaMA 特有
    'RMSNorm', 'RotaryEmbedding', 'SiLU',
}
```

### 5.3 从 PyTorch 到自定义推理框架的适配

#### 5.3.1 基于 torch.autograd.Function 的算子接入

```python
class RMSNormFunction(torch.autograd.Function):
    """
    将自定义 CUDA 算子包装为 PyTorch autograd Function
    支持训练（需实现 backward）和推理
    """
    
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, eps: float):
        # 保存 backward 需要的数据
        ctx.save_for_backward(x, weight)
        ctx.eps = eps
        
        # 调用自定义 CUDA Kernel
        output = custom_ops.rms_norm_fp16(x, weight, eps)
        return output
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, weight = ctx.saved_tensors
        eps = ctx.eps
        
        # 实现反向传播（推理可以不实现）
        grad_x, grad_weight = custom_ops.rms_norm_backward(
            grad_output, x, weight, eps
        )
        return grad_x, grad_weight, None  # eps 无梯度


def rms_norm(x, weight, eps=1e-6):
    return RMSNormFunction.apply(x, weight, eps)
```

#### 5.3.2 ONNX 自定义算子注册

```python
# 注册自定义 ONNX 算子域
import onnxruntime as ort
from onnxruntime import SessionOptions

# 方法一：OrtCustomOp（C++ 实现）
# 创建 .so 并通过 register_custom_ops_library 加载

# 方法二：Python 自定义算子（调试用）
class RMSNormOp:
    def __init__(self):
        self.input_types = ['tensor(float16)']
        self.output_types = ['tensor(float16)']
    
    def compute(self, x, weight):
        eps = 1e-6
        variance = (x.astype(float) ** 2).mean(-1, keepdims=True)
        x_norm = x.astype(float) / ((variance + eps) ** 0.5)
        return (x_norm * weight).astype(x.dtype)


# 在 ONNX 模型中使用自定义域
# op: domain="com.mycompany", op_type="RMSNorm"
```

#### 5.3.3 TensorRT 插件开发

```cpp
// TensorRT 自定义插件示例框架
#include "NvInfer.h"
#include "NvInferPlugin.h"

class RMSNormPlugin : public nvinfer1::IPluginV2DynamicExt {
public:
    RMSNormPlugin(float eps) : mEps(eps) {}
    
    // ---- 必须实现的接口 ----
    
    // 描述输出张量的形状
    nvinfer1::DimsExprs getOutputDimensions(
        int outputIndex,
        const nvinfer1::DimsExprs* inputs,
        int nbInputs,
        nvinfer1::IExprBuilder& exprBuilder
    ) noexcept override {
        return inputs[0];  // 输出与输入同形状
    }
    
    // 支持的数据类型
    bool supportsFormatCombination(
        int pos,
        const nvinfer1::PluginTensorDesc* inOut,
        int nbInputs, int nbOutputs
    ) noexcept override {
        return inOut[pos].type == nvinfer1::DataType::kHALF
            && inOut[pos].format == nvinfer1::TensorFormat::kLINEAR;
    }
    
    // 执行 Kernel
    int enqueue(
        const nvinfer1::PluginTensorDesc* inputDesc,
        const nvinfer1::PluginTensorDesc* outputDesc,
        const void* const* inputs,
        void* const* outputs,
        void* workspace,
        cudaStream_t stream
    ) noexcept override {
        // 获取维度信息
        auto dims = inputDesc[0].dims;
        int rows = 1;
        for (int i = 0; i < dims.nbDims - 1; i++) rows *= dims.d[i];
        int cols = dims.d[dims.nbDims - 1];
        
        // 调用 CUDA Kernel
        launch_rms_norm_fp16(
            inputs[0],   // x
            inputs[1],   // weight
            outputs[0],  // output
            rows, cols, mEps, stream
        );
        return 0;
    }
    
    // ... 其他接口（serialize/deserialize/clone 等）
    
private:
    float mEps;
};
```

### 5.4 算子精度验证

#### 5.4.1 精度验证框架

```python
import torch
import numpy as np
from dataclasses import dataclass
from typing import Callable, List, Optional

@dataclass
class PrecisionTestResult:
    max_abs_diff: float
    max_rel_diff: float
    mean_abs_diff: float
    cosine_sim: float
    pass_threshold: bool

def validate_operator_precision(
    ref_fn: Callable,
    target_fn: Callable,
    test_cases: List[dict],
    abs_threshold: float = 1e-3,
    rel_threshold: float = 1e-2,
    cosine_threshold: float = 0.9999,
) -> List[PrecisionTestResult]:
    """
    算子精度验证框架
    
    Args:
        ref_fn: 参考实现（一般是 PyTorch FP32）
        target_fn: 待验证实现（自定义算子）
        test_cases: 测试用例列表，每个 case 是输入 kwargs
    """
    results = []
    
    for i, case in enumerate(test_cases):
        # 运行参考实现
        with torch.no_grad():
            ref_out = ref_fn(**case)
            target_out = target_fn(**case)
        
        # 转为 float32 计算误差
        ref = ref_out.float().cpu()
        tgt = target_out.float().cpu()
        
        abs_diff = (ref - tgt).abs()
        rel_diff = abs_diff / (ref.abs() + 1e-8)
        
        cosine = torch.nn.functional.cosine_similarity(
            ref.flatten().unsqueeze(0),
            tgt.flatten().unsqueeze(0)
        ).item()
        
        result = PrecisionTestResult(
            max_abs_diff=abs_diff.max().item(),
            max_rel_diff=rel_diff.max().item(),
            mean_abs_diff=abs_diff.mean().item(),
            cosine_sim=cosine,
            pass_threshold=(
                abs_diff.max().item() < abs_threshold and
                cosine > cosine_threshold
            )
        )
        results.append(result)
        
        status = "✅ PASS" if result.pass_threshold else "❌ FAIL"
        print(f"Case {i:3d}: {status} | "
              f"MaxAbsDiff={result.max_abs_diff:.2e} | "
              f"CosSim={result.cosine_sim:.6f}")
    
    pass_rate = sum(r.pass_threshold for r in results) / len(results)
    print(f"\n总体通过率: {pass_rate:.1%} ({sum(r.pass_threshold for r in results)}/{len(results)})")
    
    return results


# 使用示例
def test_rms_norm_precision():
    import custom_ops
    
    def ref_rms_norm(x, weight, eps=1e-6):
        """参考实现：FP32"""
        variance = x.float().pow(2).mean(-1, keepdim=True)
        return (x.float() * torch.rsqrt(variance + eps) * weight.float()).half()
    
    def target_rms_norm(x, weight, eps=1e-6):
        """目标实现：自定义 CUDA"""
        return custom_ops.rms_norm_fp16(x, weight, eps)
    
    test_cases = [
        # 标准形状
        {'x': torch.randn(1, 512, 4096, dtype=torch.float16, device='cuda'),
         'weight': torch.ones(4096, dtype=torch.float16, device='cuda')},
        # 长序列
        {'x': torch.randn(1, 8192, 4096, dtype=torch.float16, device='cuda'),
         'weight': torch.ones(4096, dtype=torch.float16, device='cuda')},
        # 大 batch
        {'x': torch.randn(32, 128, 4096, dtype=torch.float16, device='cuda'),
         'weight': torch.ones(4096, dtype=torch.float16, device='cuda')},
        # 边界：hidden_size 不整除 VEC_SIZE
        {'x': torch.randn(2, 128, 3000, dtype=torch.float16, device='cuda'),
         'weight': torch.ones(3000, dtype=torch.float16, device='cuda')},
    ]
    
    validate_operator_precision(ref_rms_norm, target_rms_norm, test_cases)
```

#### 5.4.2 端到端模型精度验证

```python
def eval_model_perplexity(model, tokenizer, dataset, max_samples=1000):
    """
    用困惑度（Perplexity）衡量模型整体精度
    ref PPL 与 adapted PPL 之差应 < 0.5（经验值）
    """
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for i, sample in enumerate(dataset):
            if i >= max_samples:
                break
            
            inputs = tokenizer(sample['text'], return_tensors='pt',
                               truncation=True, max_length=2048).to(model.device)
            
            outputs = model(**inputs, labels=inputs['input_ids'])
            loss = outputs.loss
            
            n_tokens = inputs['input_ids'].numel()
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens
    
    ppl = torch.exp(torch.tensor(total_loss / total_tokens)).item()
    print(f"Perplexity: {ppl:.4f}")
    return ppl


def compare_model_outputs_layer_by_layer(model_ref, model_adapted, inputs):
    """
    逐层输出对比，定位精度劣化来源
    """
    ref_outputs = {}
    adapted_outputs = {}
    
    def make_hook(name, store):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                store[name] = output.detach().float()
        return hook
    
    hooks = []
    for name, module in model_ref.named_modules():
        hooks.append(module.register_forward_hook(make_hook(name, ref_outputs)))
    
    with torch.no_grad():
        model_ref(**inputs)
    for h in hooks:
        h.remove()
    
    hooks = []
    for name, module in model_adapted.named_modules():
        hooks.append(module.register_forward_hook(make_hook(name, adapted_outputs)))
    
    with torch.no_grad():
        model_adapted(**inputs)
    for h in hooks:
        h.remove()
    
    # 对比每层
    print(f"{'Layer':<60} {'MaxAbsDiff':>12} {'CosSim':>10}")
    print("-" * 85)
    
    for name in ref_outputs:
        if name in adapted_outputs:
            ref = ref_outputs[name]
            ada = adapted_outputs[name]
            
            if ref.shape != ada.shape:
                print(f"{name:<60} {'Shape mismatch!':>12}")
                continue
            
            max_diff = (ref - ada).abs().max().item()
            cos_sim = torch.nn.functional.cosine_similarity(
                ref.flatten().unsqueeze(0), ada.flatten().unsqueeze(0)
            ).item()
            
            flag = "⚠️ " if max_diff > 1e-2 else "   "
            print(f"{flag}{name:<58} {max_diff:>12.4e} {cos_sim:>10.6f}")
```

---

## 6. 主流硬件平台适配指南

### 6.1 NVIDIA GPU（CUDA）

#### 关键特性
- **Tensor Core**：执行 FP16/BF16/INT8/FP8 矩阵乘，A100 支持 FP16 TF32；H100 引入 FP8
- **NCCL**：多卡通信库，支持 AllReduce/AllGather/ReduceScatter
- **cuBLAS / cuDNN**：官方高性能算子库
- **CUTLASS**：模板化 GEMM 库，支持自定义 epilogue

#### 适配要点
```bash
# 编译选项
-arch=sm_80    # A100
-arch=sm_86    # A30/RTX3090
-arch=sm_90    # H100

# 推荐工具链
- CUDA 12.x + cuDNN 8.x/9.x
- TensorRT 9.x/10.x
- vLLM（推理框架）
- FlashAttention-2/3
```

#### 性能 Checklist
- [ ] 使用 `nsys`/`ncu` 进行 Profiling
- [ ] GEMM 算子接入 CUTLASS 或 cuBLAS
- [ ] Attention 使用 FlashAttention-2
- [ ] 开启 CUDA Graph 消除 kernel launch 开销
- [ ] 确认未出现 PCIe 带宽瓶颈（多 GPU 场景）

### 6.2 AMD GPU（ROCm/HIP）

#### 关键特性
- HIP：与 CUDA 高度兼容的编程模型
- `hipBLAS` / `rocBLAS`：对应 cuBLAS
- MIOpen：对应 cuDNN
- RCCL：对应 NCCL

#### CUDA → HIP 迁移

```bash
# 自动转换工具
hipify-clang rms_norm_kernel.cu -o rms_norm_kernel.hip.cpp

# 主要替换
cudaMalloc      → hipMalloc
cudaFree        → hipFree
cudaMemcpy      → hipMemcpy
__shfl_xor_sync → __shfl_xor   # 注意：ROCm 不需要 mask 参数（部分版本）
cuda_fp16.h     → hip/hip_fp16.h
```

#### 已知差异与 Workaround

| 差异 | CUDA | ROCm/HIP | 解决方案 |
|------|------|----------|----------|
| Warp size | 32 | 64（MI系列） | 代码参数化，不 hardcode |
| FP8 支持 | H100 原生 | MI300X 支持 | 根据硬件版本条件编译 |
| Shared mem bank | 4B/bank | 4B/bank | 相同 |
| CUB → rocPRIM | 有 | 对应库 | `#include <hipcub/hipcub.hpp>` |

### 6.3 华为昇腾 NPU（CANN/MindSpore）

#### 架构特点
- **AI Core**：矩阵运算单元（类似 Tensor Core）
- **Vector Core**：向量运算单元
- **Cube Unit**：执行 FP16/INT8 矩阵乘
- **UB（Unified Buffer）**：片上高速缓存（约 256KB per Core）

#### 算子开发方式

```python
# 方式一：TBE（Tensor Boost Engine）DSL 方式
from te import tvm
import te.lang.cce as tbe

def rms_norm_tbe(x, weight, eps=1e-6):
    """昇腾 TBE 算子实现"""
    # 计算 x^2 的均值
    x_square = tbe.vmul(x, x)
    x_square_mean = tbe.reduce_mean(x_square, axis=-1, keepdims=True)
    
    # 计算 rsqrt
    x_square_mean_eps = tbe.vadds(x_square_mean, eps)
    rms_inv = tbe.vrsqrt(x_square_mean_eps)
    
    # 归一化
    x_norm = tbe.vmul(x, rms_inv)
    output = tbe.vmul(x_norm, weight)
    return output
```

```python
# 方式二：Ascend C（推荐，类 CUDA C++ 风格）
# Ascend C 代码结构
"""
__aicore__ void rms_norm_kernel(
    GM_ADDR x_gm, GM_ADDR weight_gm, GM_ADDR out_gm,
    GM_ADDR tiling_gm
) {
    // 声明本地内存
    TBuf<TPosition::VECCALC> x_local, weight_local, out_local;
    
    // 从 Global Memory 搬入数据
    DataCopy(x_local, x_gm[offset], tiling.blockLen);
    
    // 向量计算
    Mul(x_sq, x_local, x_local, tiling.blockLen);
    ReduceSum(sum_buf, x_sq, tiling.blockLen);
    // ... 后续计算
    
    // 搬出结果
    DataCopy(out_gm[offset], out_local, tiling.blockLen);
}
"""
```

#### 精度对齐要点

- 昇腾默认使用 FP16，某些归约操作累加精度不同于 CUDA
- `ReduceSum` 结果与 CUDA 有累加顺序差异，需适当放宽阈值
- 推荐使用 `atc` 工具链做模型转换，注意算子融合规则

### 6.4 寒武纪 MLU（BANG C）

#### 特点
- BANG C：类 CUDA 的编程语言
- `cnrtMalloc` / `cnrtMemcpy`：对应 cudaMalloc/cudaMemcpy
- CNNL：对应 cuDNN 的算子库

```c
// BANG C 算子开发示例（伪代码结构）
__mlu_global__ void rms_norm_mlu(
    __mlu_global__ half *x,
    __mlu_global__ half *weight,
    __mlu_global__ half *output,
    int rows, int cols, float eps
) {
    // 声明 NRAM（片上内存，类似 shared memory）
    __nram__ half x_nram[MAX_COLS];
    __nram__ float sq_sum_nram;
    
    // 加载数据到 NRAM
    __memcpy(x_nram, x + taskId * cols, cols * sizeof(half), GDRAM2NRAM);
    
    // 向量计算
    __bang_mul(x_nram, x_nram, x_nram, cols);  // x^2
    __bang_sumpool(sq_sum_nram, x_nram, cols, 1, 1, cols, 1, 1);
    
    // 后续处理...
}
```

### 6.5 Intel GPU（XPU/SYCL）

```cpp
// SYCL 算子开发（基于 DPC++）
#include <sycl/sycl.hpp>

void rms_norm_sycl(
    sycl::queue& q,
    const sycl::half* x, const sycl::half* weight, sycl::half* out,
    int rows, int cols, float eps
) {
    q.submit([&](sycl::handler& h) {
        h.parallel_for(
            sycl::nd_range<1>(rows * 256, 256),
            [=](sycl::nd_item<1> item) {
                int row = item.get_group(0);
                int tid = item.get_local_id(0);
                
                // 使用 group reduce
                auto group = item.get_group();
                
                float sum_sq = 0.0f;
                for (int i = tid; i < cols; i += 256) {
                    float xi = sycl::half2float(x[row * cols + i]);
                    sum_sq += xi * xi;
                }
                
                sum_sq = sycl::reduce_over_group(group, sum_sq, sycl::plus<float>{});
                float rms_inv = sycl::rsqrt(sum_sq / cols + eps);
                
                for (int i = tid; i < cols; i += 256) {
                    float xi = sycl::half2float(x[row * cols + i]);
                    float wi = sycl::half2float(weight[i]);
                    out[row * cols + i] = sycl::float2half(xi * rms_inv * wi);
                }
            }
        );
    });
}
```

---

## 7. 算子调试与性能分析

### 7.1 CUDA 性能分析工具链

#### 7.1.1 Nsight Systems（系统级 Profiling）

```bash
# 采集 Profile
nsys profile \
    --trace=cuda,nvtx,osrt \
    --output=my_model_profile \
    python run_inference.py

# 分析报告
nsys stats my_model_profile.nsys-rep

# 关注指标
# - SM 利用率（目标 > 80%）
# - 显存带宽利用率
# - Kernel launch gap（CUDA Graph 可消除）
# - PCIe 传输时间
```

#### 7.1.2 Nsight Compute（Kernel 级 Profiling）

```bash
# 分析特定 Kernel
ncu \
    --set full \
    --kernel-name rms_norm_kernel_fp16 \
    --launch-count 10 \
    python run_inference.py

# 关键指标解读
# 1. Memory Throughput: 实际带宽 / 峰值带宽
# 2. Compute Throughput: 实际算力 / 峰值算力  
# 3. Occupancy: 活跃 Warp / 最大 Warp
# 4. Warp Efficiency: 无分支发散时应为 100%
# 5. L1/L2 Hit Rate: 缓存命中率
# 6. Bank Conflicts: Shared Memory 冲突
```

#### 7.1.3 Python 层 Profiling

```python
import torch.profiler

# 使用 torch.profiler 获取算子级耗时
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
    profile_memory=True,
    with_flops=True,
) as prof:
    for _ in range(10):
        model(input_ids)

# 打印耗时 Top-K
print(prof.key_averages().table(
    sort_by="cuda_time_total",
    row_limit=20
))

# 导出 Chrome Trace
prof.export_chrome_trace("trace.json")
```

### 7.2 常见性能问题诊断

#### 7.2.1 显存带宽瓶颈

```python
def estimate_bandwidth_bottleneck(hidden_size, seq_len, batch_size):
    """判断某算子是否受显存带宽限制"""
    dtype_bytes = 2  # FP16
    
    # RMSNorm 算术强度估算
    read_bytes = batch_size * seq_len * hidden_size * dtype_bytes  # 读 x
    read_bytes += hidden_size * dtype_bytes                          # 读 weight
    write_bytes = batch_size * seq_len * hidden_size * dtype_bytes  # 写 out
    
    compute_flops = batch_size * seq_len * hidden_size * 4           # 近似

    ai = compute_flops / (read_bytes + write_bytes)
    print(f"算术强度: {ai:.2f} FLOP/Byte")
    
    # A100 ridge point ≈ 312 TFLOPS / 2 TB/s = 156 FLOP/Byte
    if ai < 156:
        print("→ 显存带宽瓶颈（优化方向：减少读写 / 融合算子）")
    else:
        print("→ 计算瓶颈（优化方向：提升算力利用率）")
```

#### 7.2.2 寄存器溢出（Register Spilling）

```bash
# 编译时查看寄存器使用
nvcc -Xptxas -v my_kernel.cu
# 输出示例：
# ptxas info    : Used 128 registers, 32784+0 bytes smem, ...
# 如果 registers > 128（通常），会导致 occupancy 下降

# 限制寄存器数量（牺牲部分性能避免溢出）
__launch_bounds__(256, 4)  // maxThreadsPerBlock=256, minBlocksPerMultiprocessor=4
__global__ void my_kernel(...) {}
```

#### 7.2.3 Kernel Launch 开销

```python
# CUDA Graph 消除重复 kernel launch 开销
# 适用于静态形状、固定计算图的推理场景

# 捕获阶段
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    output = model(input_ids)

# 执行阶段（极低开销）
for _ in range(1000):
    input_ids.copy_(new_input_ids)  # 更新输入
    g.replay()                       # 重放 Graph
```

### 7.3 精度调试技巧

```python
def debug_nan_inf(model, input_ids):
    """检测模型中的 NaN/Inf 问题"""
    
    def nan_hook(module, input, output):
        if isinstance(output, torch.Tensor):
            if torch.isnan(output).any() or torch.isinf(output).any():
                print(f"❌ NaN/Inf detected in: {module.__class__.__name__}")
                print(f"   Input stats: min={input[0].min():.4f}, max={input[0].max():.4f}")
                print(f"   Output has {torch.isnan(output).sum()} NaN, "
                      f"{torch.isinf(output).sum()} Inf")
                raise RuntimeError(f"NaN/Inf in {module.__class__.__name__}")
    
    hooks = []
    for module in model.modules():
        hooks.append(module.register_forward_hook(nan_hook))
    
    try:
        with torch.no_grad():
            model(input_ids)
    finally:
        for h in hooks:
            h.remove()


# 混合精度 NaN 排查
# 常见原因：Softmax 输入太大（超出 FP16 范围）
# 解决：attention score * softmax_scale，确保 scale 正确
# scale = 1 / sqrt(head_dim)，head_dim=128 时 scale ≈ 0.088
```

---

## 8. 常见问题与解决方案

### 8.1 精度问题

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| Softmax NaN | attention score 超出 FP16 范围 | 检查 attention scale，使用 online softmax |
| RMSNorm 偏差大 | 累加顺序不同 | 内部用 FP32 累加，结果转 FP16 |
| 量化精度劣化严重 | 激活值分布不均 | 使用 SmoothQuant 平滑激活值分布 |
| 长序列精度下降 | RoPE 角度累计误差 | 使用 FP32 计算 RoPE，结果截断为 FP16 |
| 多卡结果不一致 | AllReduce 浮点累加非交换 | 固定 reduce 顺序，或使用确定性算法 |

### 8.2 性能问题

| 问题 | 诊断方法 | 解决方案 |
|------|----------|----------|
| GEMM 效率低 | ncu 查看 Tensor Core 利用率 | 确保矩阵维度对齐（A100: 64B 对齐） |
| Attention 慢 | 内存占用大，带宽瓶颈 | 使用 FlashAttention-2 |
| 推理延迟高 | nsys 发现大量小 Kernel | 启用 CUDA Graph；算子融合 |
| 显存 OOM | 峰值显存超出限制 | PagedAttention；减少 KV Cache |
| 吞吐低 | GPU 利用率低 | 增大 batch size；Continuous Batching |

### 8.3 适配问题

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| 算子不支持动态 Shape | 静态编译假设固定维度 | 实现 shape inference 逻辑；bucket 化 |
| 数据类型不匹配 | 框架默认 dtype 差异 | 显式 cast 或在算子内部处理 |
| 算子语义差异 | 不同框架对 axis/dim 的理解 | 仔细阅读文档，编写测试用例覆盖边界 |
| 内存对齐问题 | NPU 要求特定对齐 | padding 到对齐边界，或使用对齐分配器 |

### 8.4 调试 Checklist

```
新算子上线前必查：
□ 单元测试：覆盖标准形状、边界形状、极值输入
□ 精度测试：vs FP32 参考实现，cosine sim > 0.9999
□ 端到端测试：模型 PPL 与 baseline 差异 < 0.5
□ 性能测试：与 baseline 相比，目标场景有提升
□ 显存测试：无显存泄漏（运行 100 次，显存稳定）
□ 多线程安全：多 stream 并发无竞态
□ 输入异常处理：空 tensor、极大/极小值、NaN 输入
□ 文档完整：接口说明、性能数据、已知限制
```

---

## 9. 附录：参考工具链与资源

### 9.1 核心工具链

| 工具 | 用途 | 备注 |
|------|------|------|
| **NVCC** | CUDA 编译器 | NVIDIA 官方 |
| **Nsight Systems** | 系统级 Profiling | `nsys` 命令 |
| **Nsight Compute** | Kernel 级 Profiling | `ncu` 命令 |
| **CUTLASS** | 高性能 GEMM 模板 | NVIDIA 开源 |
| **FlashAttention** | 注意力优化实现 | Tri Dao 等开源 |
| **vLLM** | 推理框架 | PagedAttention |
| **TensorRT-LLM** | NVIDIA 推理框架 | 生产级 |
| **torch.profiler** | PyTorch 内置 Profiler | 算子级耗时 |
| **ONNX** | 模型交换格式 | 适配桥梁 |
| **Netron** | 可视化 ONNX 模型 | 调试利器 |

### 9.2 关键参考文献

- **FlashAttention**: "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" (Dao et al., 2022)
- **FlashAttention-2**: "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning" (Dao, 2023)
- **PagedAttention**: "Efficient Memory Management for Large Language Model Serving with PagedAttention" (Kwon et al., 2023)
- **SmoothQuant**: "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models" (Xiao et al., 2022)
- **AWQ**: "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration" (Lin et al., 2023)
- **GPTQ**: "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers" (Frantar et al., 2022)
- **Megatron-LM**: "Efficient Large-Scale Language Model Training on GPU Clusters" (Narayanan et al., 2021)

### 9.3 开源仓库索引

```
算子实现参考：
├── NVIDIA/cutlass          - GEMM 模板库
├── Dao-AILab/flash-attention - FlashAttention 实现
├── vllm-project/vllm       - PagedAttention + 推理框架
├── NVIDIA/TensorRT-LLM     - TensorRT 推理优化
├── huggingface/transformers - 模型参考实现
├── microsoft/DeepSpeed     - 分布式训练/推理
├── pytorch/ao              - 量化工具箱
└── OpenAI/triton           - GPU 编程 DSL（Python 风格）

量化工具：
├── IST-DASLab/gptq         - GPTQ 量化
├── mit-han-lab/llm-awq     - AWQ 量化
└── mit-han-lab/smoothquant  - SmoothQuant
```

### 9.4 快速参考：大模型关键算子性能指标（A100 80GB 基准）

| 算子 | 典型配置 | 理论峰值带宽/算力占比 | 优化目标 |
|------|----------|----------------------|----------|
| GEMM (FP16) | M=N=K=4096 | Tensor Core > 70% | > 300 TFLOPS |
| FlashAttention | BS=8, seq=2048, h=32 | MBW > 70% | < 5ms |
| RMSNorm | BS=8, seq=2048, h=4096 | MBW > 60% | < 0.5ms |
| Embedding | vocab=32000, emb=4096 | MBW > 50% | < 1ms |
| KV Cache Copy | 4层, BS=32, seq=2048 | MBW > 80% | < 2ms |

---

---

# Part II：vLLM / vLLM-Ascend 算子适配专题

## 目录（Part II）

10. [vLLM 框架架构深度解析](#10-vllm-框架架构深度解析)
11. [vLLM 核心算子体系](#11-vllm-核心算子体系)
12. [vLLM GPU 算子开发与优化](#12-vllm-gpu-算子开发与优化)
13. [vLLM Attention 后端体系](#13-vllm-attention-后端体系)
14. [vLLM-Ascend 插件架构与算子适配](#14-vllm-ascend-插件架构与算子适配)
15. [vLLM-Ascend 自定义算子开发实战](#15-vllm-ascend-自定义算子开发实战)
16. [GPU ↔ NPU 算子迁移方法论](#16-gpu--npu-算子迁移方法论)
17. [torch.compile 与 CUDA Graph 适配](#17-torchcompile-与-cuda-graph-适配)
18. [分布式算子与通信适配](#18-分布式算子与通信适配)
19. [量化算子在 vLLM 中的适配](#19-量化算子在-vllm-中的适配)
20. [端到端适配实战案例](#20-端到端适配实战案例)
21. [适配工程最佳实践与 Checklist](#21-适配工程最佳实践与-checklist)

---

## 10. vLLM 框架架构深度解析

### 10.1 整体架构全景

vLLM 是当前最主流的开源大模型推理服务框架，2024 年 7 月由 UC Berkeley 贡献至 Linux Foundation，成为 PyTorch Foundation 托管项目。2026 年 1 月，核心团队成立商业公司 Inferact 并完成 1.5 亿美元种子融资。

其核心创新在于 **PagedAttention**（分页 KV 缓存管理）与 **Continuous Batching**（迭代级动态批处理）。V1 引擎自 v0.6.0（2025 年）起成为默认引擎，对调度器、KV Cache 管理器、Worker 模型进行了全面重构。

```
                        ┌─────────────────────────────┐
                        │      API Server              │
                        │  (OpenAI-Compatible / gRPC)  │
                        │  [独立进程，ZeroMQ IPC]       │
                        └────────────┬────────────────┘
                                     │
                        ┌────────────▼────────────────┐
                        │      Engine Core Process     │
                        │  ┌──────────┬─────────────┐  │
                        │  │ Scheduler│ KV Cache Mgr│  │
                        │  │ (请求调度)│ (Block 分配) │  │
                        │  └──────────┴─────────────┘  │
                        │  每个 DP rank 一个 Engine Core │
                        └────────────┬────────────────┘
                                     │ 异步调度（Step N+1 调度 ∥ Step N 执行）
                       ┌─────────────┼─────────────┐
              ┌────────▼──────┐             ┌──────▼────────┐
              │  Worker 0     │             │  Worker N     │
              │ (GPU/NPU)     │   ...       │ (GPU/NPU)     │
              │ ┌───────────┐ │             │ ┌───────────┐ │
              │ │ModelRunner│ │             │ │ModelRunner│ │
              │ │ (模型加载  │ │             │ │ (模型加载  │ │
              │ │  前向执行  │ │             │ │  前向执行  │ │
              │ │  采样输出) │ │             │ │  采样输出) │ │
              │ └───────────┘ │             │ └───────────┘ │
              └───────────────┘             └───────────────┘
                       │                             │
                  NCCL / HCCL AllReduce / AllGather
```

### 10.2 V1 引擎核心设计

#### 10.2.1 多进程架构

V1 引擎采用多进程设计——即使单 GPU 也有 2 个进程（1 Engine Core + 1 Worker）。当 TP=n、PP=m 时共 n×m+1 个进程。

```
时序示意（调度-执行异步重叠流水线）:

Engine Core:  ┌Sched_0┐ ┌Sched_1┐ ┌Sched_2┐ ┌Sched_3┐ ...
              └───┬───┘ └───┬───┘ └───┬───┘ └───┬───┘
                  │         │         │         │
Worker GPU:       │  ┌Exec_0┐  ┌Exec_1┐  ┌Exec_2┐
                  │  └──────┘  └──────┘  └──────┘
                  └──overlap──┘──overlap──┘

关键优化:
  • Pinned host memory + DMA 零拷贝传输
  • CPU 侧预处理（tokenize / 多模态处理 / detokenize）异步独立进程
  • Worker 维护请求状态——In-flight 请求仅传增量（request ID + 新 block ID）
```

#### 10.2.2 统一 Prefill-Decode 调度

V0 严格分离 prefill 与 decode 阶段；V1 统一处理所有 token：

```python
class V1Scheduler:
    """V1 调度器核心逻辑（伪代码）"""
    
    def schedule(self) -> SchedulerOutput:
        budget = self.compute_budget()  # 可用 KV Cache blocks
        
        # 1. 优先调度正在运行的请求（decode）
        for req in self.running_requests:
            if budget.can_allocate(req.next_tokens):
                budget.allocate(req)
        
        # 2. 调度等待中的新请求（prefill），支持 chunk 切分
        for req in self.waiting_requests:
            chunk_size = min(req.remaining_prefill, budget.remaining)
            if chunk_size > 0:
                budget.allocate_prefill(req, chunk_size)
        
        # 3. 预算不足时，可 preempt 低优先级请求（swap_out）
        return SchedulerOutput(
            scheduled_requests=budget.scheduled,
            blocks_to_swap_in=budget.swap_in,
            blocks_to_swap_out=budget.swap_out,
        )
```

#### 10.2.3 VllmConfig 全局配置

```python
@dataclass
class VllmConfig:
    """引擎级全局状态，所有组件共享"""
    model_config: ModelConfig           # 模型结构
    cache_config: CacheConfig           # KV Cache 参数
    parallel_config: ParallelConfig     # TP/PP/DP 策略
    scheduler_config: SchedulerConfig   # 调度策略
    device_config: DeviceConfig         # 硬件后端（cuda/npu/...）
    compilation_config: CompilationConfig  # torch.compile / Graph 配置
    # ...
```

### 10.3 代码关键路径（算子开发视角）

```
vllm/
├── model_executor/
│   ├── models/               # 各模型实现（算子组装层）
│   │   ├── llama.py / qwen2.py / deepseek_v3.py ...
│   ├── layers/               # 可复用的算子抽象层 ★
│   │   ├── linear.py                # 线性层（含 TP 切分）
│   │   ├── layernorm.py             # LayerNorm / RMSNorm
│   │   ├── rotary_embedding.py      # RoPE / ALiBi
│   │   ├── activation.py            # SiLU / GeLU 等
│   │   ├── sampler.py               # 采样层
│   │   └── quantization/            # 量化算子（awq/gptq/fp8/...）
│   └── custom_op.py          # CustomOp 基类 ★
│
├── attention/
│   ├── backends/             # Attention 后端实现 ★
│   │   ├── abstract.py              # AttentionBackend / AttentionImpl 接口
│   │   ├── flash_attn.py            # FlashAttention-2/3
│   │   ├── flashinfer.py / triton.py / pallas.py ...
│   └── layer.py
│
├── v1/                       # V1 引擎
│   ├── engine/core.py               # EngineCore（调度循环）
│   ├── worker/gpu_worker.py         # GPU Worker
│   └── attention/backends/          # V1 注意力后端
│
├── _custom_ops.py            # CUDA 自定义 ops Python 绑定 ★
├── distributed/              # 分布式通信（NCCL 封装）
└── platforms/                # 硬件平台抽象（cuda/rocm/...）★
```

> **算子适配核心入口**：自定义算子在 `_custom_ops.py` 注册；Attention 后端在 `attention/backends/` 实现；层算子在 `model_executor/layers/` 组装；硬件平台通过 `platforms/` 或外部插件接入。

---

## 11. vLLM 核心算子体系

### 11.1 自定义算子注册机制

vLLM 提供两种注册方式：

```python
# ========================================
# 方式一：直接绑定 C++/CUDA 扩展
# ========================================
def paged_attention_v1(out, query, key_cache, value_cache, ...):
    torch.ops._C_cache_ops.paged_attention_v1(
        out, query, key_cache, value_cache, ...
    )

# ========================================
# 方式二：CustomOp 基类（推荐，支持多后端分发）
# ========================================
from vllm.model_executor.custom_op import CustomOp

class MyRMSNorm(CustomOp):
    def forward_native(self, x, residual=None):
        """PyTorch 原生 fallback"""
        variance = x.float().pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight
    
    def forward_cuda(self, x, residual=None):
        """CUDA 优化路径"""
        from vllm import _custom_ops as ops
        if residual is not None:
            ops.fused_add_rms_norm(x, residual, self.weight.data, self.eps)
            return x, residual
        out = torch.empty_like(x)
        ops.rms_norm(out, x, self.weight.data, self.eps)
        return out, residual
    
    # vLLM-Ascend 可在 Patch 系统中增加:
    # def forward_npu(self, x, residual=None): ...
```

> **重要动向（2026 年）**：vLLM 社区已提出 [RFC] vLLM IR——基于 torch 的函数式中间表示。它将算子语义与具体实现解耦，在 `torch.fx` 图上表达为自定义高级算子节点，便于编译 Pass 识别和融合，同时支持 OOT（树外）后端注册。这将逐步取代现有的 `CustomOp` 调度机制。

### 11.2 完整算子清单（v0.18+）

#### Attention 相关

| 算子 | 功能 | 适配优先级 |
|------|------|-----------|
| `paged_attention_v1` | 单分区 PagedAttention（短-中序列） | ★★★★★ |
| `paged_attention_v2` | 多分区 PagedAttention（长序列并行） | ★★★★★ |
| `reshape_and_cache` | KV Cache 写入（vLLM 交错布局） | ★★★★★ |
| `reshape_and_cache_flash` | FlashAttention 兼容 KV 写入 | ★★★★☆ |
| `copy_blocks` | KV Cache 块拷贝（Beam Search） | ★★★★☆ |
| `swap_blocks` | KV Cache 块换入换出（CPU ↔ GPU） | ★★★☆☆ |

#### 归一化

| 算子 | 融合模式 |
|------|---------|
| `rms_norm` | 独立 RMSNorm |
| `fused_add_rms_norm` | 残差 + RMSNorm 融合 |
| `layernorm` / `fused_add_layernorm` | LayerNorm 及融合版 |

#### 激活与 MLP

| 算子 | 功能 |
|------|------|
| `silu_and_mul` | SiLU(gate) × up — LLaMA FFN |
| `gelu_and_mul` / `gelu_tanh_and_mul` | GeLU 变体 |
| `gelu_fast` / `gelu_new` | 近似 GeLU |

#### 位置编码

| 算子 | 功能 |
|------|------|
| `rotary_embedding` | 标准 RoPE |
| `batched_rotary_embedding` | 批量 RoPE |

#### 量化

| 算子 | 功能 |
|------|------|
| `scaled_fp8_quant` / `static_scaled_fp8_quant` | FP8 量化 |
| `scaled_int8_quant` | INT8 量化 |
| `cutlass_scaled_mm` | CUTLASS 量化 GEMM |
| `gptq_gemm` / `awq_gemm` / `marlin_gemm` | 低比特 GEMM |

#### 采样 / MoE

| 算子 | 功能 |
|------|------|
| `top_k_top_p_sampling` | Top-K + Top-P 联合采样 |
| `moe_align_block_size` | MoE 路由对齐 |

### 11.3 算子执行顺序（LLaMA-like 模型单步 Decode）

```
Embedding Lookup
  │
  ▼
┌─── Transformer Layer ×N ────────────────────────┐
│                                                   │
│  RMSNorm → QKV Linear → RoPE → reshape_and_cache│
│                                      │            │
│                              ┌───────▼────────┐  │
│                              │ PagedAttention  │  │
│                              │ (核心瓶颈算子)   │  │
│                              └───────┬────────┘  │
│                                      │            │
│  O Linear → Add Residual                         │
│       │                                           │
│  fused_add_rms_norm → Gate Linear + Up Linear    │
│                              │                    │
│                         silu_and_mul              │
│                              │                    │
│                         Down Linear → Add Residual│
└───────────────────────────────────────────────────┘
  │
  ▼
RMSNorm → LM Head Linear → Sampling
```

> **适配优先级**：PagedAttention / Attention、GEMM（Linear）、RMSNorm、RoPE、silu_and_mul 合计 >95% 推理计算量。

---

## 12. vLLM GPU 算子开发与优化

### 12.1 PagedAttention Kernel 深度剖析

#### 12.1.1 KV Cache 分页内存布局

```
传统连续分配（大量内部碎片）:
  seq_0: ████████████████░░░░░░░░░░  ← 预留 max_len，浪费
  seq_1: ████████░░░░░░░░░░░░░░░░░░  ← 浪费

PagedAttention 按需分页（碎片约 4% 以下）:
  Block Pool: [B0][B1][B2][B3][B4][B5][B6][B7] ...

  seq_0 Block Table: [B0, B3, B7]      逻辑连续，物理不连续
  seq_1 Block Table: [B1, B5]          按需动态扩展

  每个 Block 存固定 token 数（典型 block_size=16）:
    Key Block:   [num_heads, head_dim/x, block_size, x]   ← 交错布局
    Value Block: [num_heads, head_dim, block_size]
    
  交错布局说明（Key 独有）:
    x = VEC_SIZE（如 FP16 时 x=8 → 128-bit 向量加载）
    读取时一次加载 [block_size, x] 连续块
    → 同时获取多 token 在 x 维度的数据，最大化访存效率
```

#### 12.1.2 V1 Kernel 核心流程

```cuda
// 线程组织: Grid[num_heads, num_seqs], Block[WARP_SIZE × NUM_WARPS]

template <typename T, int BLOCK_SIZE, int HEAD_DIM>
__global__ void paged_attention_v1_kernel(
    T* out, const T* q, const T* k_cache, const T* v_cache,
    const int* block_tables, const int* seq_lens, float scale)
{
    const int seq_idx  = blockIdx.y;
    const int head_idx = blockIdx.x;
    const int seq_len  = seq_lens[seq_idx];
    const int num_blocks = CEIL_DIV(seq_len, BLOCK_SIZE);
    
    // 1. 加载 Query 到寄存器
    float q_vec[VEC_SIZE];
    load_query(q, seq_idx, head_idx, q_vec);
    
    // 2. 遍历所有 KV blocks → 点积 Q·K × scale
    float max_logit = -FLT_MAX;
    __shared__ float logits[MAX_SEQ_LEN];
    
    for (int bi = 0; bi < num_blocks; bi++) {
        int phys_block = block_tables[seq_idx * max_blocks + bi];
        // 通过 block table 间接寻址 → 加载 K 到 shared memory
        // 计算 dot(Q, K) × scale → logits[]
        // 同步更新 max_logit
    }
    
    // 3. Online Softmax（数值稳定：先 reduce max，再 exp-sum）
    max_logit = block_reduce_max(max_logit);
    float exp_sum = 0.0f;
    for (int i = threadIdx.x; i < seq_len; i += blockDim.x) {
        logits[i] = expf(logits[i] - max_logit);
        exp_sum += logits[i];
    }
    exp_sum = block_reduce_sum(exp_sum);
    
    // 4. 加权求和 V → 输出
    // acc = sum( softmax_weight[i] × V[i] )
    store_output(out, seq_idx, head_idx, acc);
}
```

#### 12.1.3 V2 多分区并行

```
V1 瓶颈：序列极长时（>8K），单 thread block 遍历全部 KV blocks → GPU 利用率下降

V2 方案：将 KV blocks 分为 P 个 partition，各 partition 独立并行
  Grid: [num_heads, num_seqs, num_partitions]
  
  Phase 1: 各 partition 独立计算局部 softmax(Q·K^T) × V
  Phase 2: 跨 partition 合并（在线 softmax 跨分区合并）
    max_global     = max(max_1, max_2, ...)
    rescale_i      = exp(max_i − max_global)
    exp_sum_global = Σ(exp_sum_i × rescale_i)
    output         = Σ(output_i × rescale_i × exp_sum_i) / exp_sum_global
```

### 12.2 KV Cache 管理算子（reshape_and_cache）

```python
# 核心逻辑：将新 token 的 K/V 写入分页 KV Cache

# Key 和 Value 的 cache 布局不同——这是适配新硬件时的关键设计决策：
# Key:   [num_blocks, num_heads, head_dim/x, block_size, x]  ← 优化 Q·K 向量化
# Value: [num_blocks, num_heads, head_dim, block_size]        ← 优化 softmax·V

# slot_mapping[token_i] → 物理 slot
# slot = block_idx × block_size + block_offset

# 适配新硬件时须决策：
# 1. VEC_SIZE (x) 取多大？→ 取决于硬件最优访存宽度
# 2. 是否保留 K 交错布局？→ 无硬件向量加载指令则不必
# 3. K/V 可否统一布局？→ 统一简化代码但可能损失访存效率
```

### 12.3 Fused 算子模式（silu_and_mul 示例）

```cuda
// out = SiLU(gate) × up
// gate 和 up 在输入中连续: input = [gate | up], 各 d 维

template <typename T>
__global__ void silu_and_mul_kernel(T* out, const T* input, int d) {
    const int tok = blockIdx.x;
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        float gate = (float)input[tok * 2 * d + i];
        float up   = (float)input[tok * 2 * d + d + i];
        float silu = gate / (1.0f + expf(-gate));  // SiLU = x·σ(x)
        out[tok * d + i] = (T)(silu * up);
    }
}
// 融合收益: 消除 gate 和 up 的中间结果写回 HBM，带宽节省约 50%
```

---

## 13. vLLM Attention 后端体系

### 13.1 三层抽象接口

```python
# ========== Layer 1: AttentionBackend ==========
class AttentionBackend(ABC):
    @staticmethod
    def get_name() -> str: ...            # "FLASH_ATTN" / "TRITON" / "ASCEND"
    @staticmethod
    def get_impl_cls() -> Type["AttentionImpl"]: ...
    @staticmethod
    def get_metadata_cls() -> Type["AttentionMetadata"]: ...
    @staticmethod
    def get_kv_cache_shape(...) -> Tuple[int, ...]: ...   # KV Cache 形状

# ========== Layer 2: AttentionImpl ==========
class AttentionImpl(ABC):
    @abstractmethod
    def forward(self, query, key, value, kv_cache, attn_metadata, output=None):
        # 核心：执行 Attention 计算
        ...

# ========== Layer 3: AttentionMetadata ==========
class AttentionMetadata:
    num_prefills: int             # prefill 请求数
    num_decode_tokens: int        # decode token 数
    block_tables: torch.Tensor    # [num_seqs, max_blocks] → 物理 block 映射
    slot_mapping: torch.Tensor    # [num_tokens] → 物理 slot
    seq_lens: List[int]           # 各序列长度
    # + prefill/decode 特有字段（cu_seqlens 等）
```

### 13.2 各后端对比（截至 2026-06）

| 后端 | 语言 | Prefill | Decode | CUDA Graph | 平台 | 备注 |
|------|------|---------|--------|------------|------|------|
| **FlashAttention-2** | CUDA/Cutlass | ✅ varlen | ✅ PA | Piecewise | NVIDIA SM80+ | 成熟稳定 |
| **FlashAttention-3** | CUDA/CuTe | ✅ | ✅ | Full+Piecewise | SM90 (H100) | 性能最优 |
| **FlashAttention-4** | CUDA/SM100 | ✅ | ✅ | Full | SM100 (B200) | Blackwell 专用 |
| **Triton** | Triton DSL | ✅ | ✅ | Full+Piecewise | NVIDIA+AMD | 跨平台 |
| **FlashInfer** | CUDA | ✅ | ✅ | Decode Only | NVIDIA | 灵活 API |
| **FlashMLA** | CUDA | ✅ | ✅ MLA | Decode Only | NVIDIA | DeepSeek 专用 |
| **Ascend** | CANN/AscendC | ✅ NFA | ✅ FIA/DFlash | ACL Graph | 昇腾 NPU | 见第 14 章 |

### 13.3 FlashAttention 后端集成核心逻辑

```python
class FlashAttentionImpl(AttentionImpl):
    def forward(self, query, key, value, kv_cache, attn_metadata, output=None):
        # reshape: [tokens, hidden] → [tokens, heads, head_dim]
        query = query.view(-1, self.num_heads, self.head_size)
        
        # 1. 写入 KV Cache
        if kv_cache is not None:
            ops.reshape_and_cache_flash(
                key, value, kv_cache[0], kv_cache[1],
                attn_metadata.slot_mapping, ...
            )
        
        # 2. Prefill → flash_attn_varlen_func (FlashAttention-2/3)
        if attn_metadata.num_prefills > 0:
            prefill_output = flash_attn_varlen_func(
                q=..., k=..., v=...,
                cu_seqlens_q=attn_metadata.query_start_loc,
                cu_seqlens_k=attn_metadata.seq_start_loc,
                max_seqlen_q=..., max_seqlen_k=...,
                softmax_scale=self.scale, causal=True,
                block_table=attn_metadata.block_tables,
            )
        
        # 3. Decode → PagedAttention V1/V2
        if attn_metadata.num_decode_tokens > 0:
            ops.paged_attention_v1(  # 或 v2（长序列）
                decode_output, decode_query,
                kv_cache[0], kv_cache[1],
                self.num_kv_heads, self.scale,
                attn_metadata.block_tables, ...
            )
        
        return output.view(-1, self.num_heads * self.head_size)
```

### 13.4 开发新 Attention 后端标准步骤

```python
# Step 1: 实现 AttentionBackend
class MyHWBackend(AttentionBackend):
    @staticmethod
    def get_name(): return "MY_HW"
    @staticmethod
    def get_impl_cls(): return MyHWImpl
    @staticmethod
    def get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size):
        return (2, num_blocks, block_size, num_kv_heads, head_size)

# Step 2: 实现 AttentionImpl（核心是 forward 方法）
class MyHWImpl(AttentionImpl):
    def forward(self, query, key, value, kv_cache, attn_metadata, output=None):
        # 调用硬件特有 attention kernel
        ...

# Step 3: 在 Platform 中注册
class MyPlatform(Platform):
    def get_attn_backend_cls(self, ...):
        return MyHWBackend
```


---

## 14. vLLM-Ascend 插件架构与算子适配

### 14.1 插件架构总览

vLLM-Ascend 是 vLLM 官方推荐的昇腾 NPU 适配插件，基于 Hardware Pluggable RFC 实现解耦集成，由社区维护（vllm-project/vllm-ascend）。从 2024 年 12 月启动至今，已发布至 v0.18.0（2026-05），支持 A2（910B）、A3（910C）、A5（950）三代芯片。

```
vLLM Core（上游主仓）                    vLLM-Ascend Plugin（独立仓库）
┌────────────────────┐                  ┌───────────────────────────┐
│ Platform Interface │◄── entrypoint ──│ NPUPlatform               │
│ WorkerBase         │◄────────────────│ NPUWorker / NPUModelRunner│
│ AttentionBackend   │◄────────────────│ AscendAttn / DFlash / DSA │
│ CommunicatorBase   │◄────────────────│ HCCLCommunicator          │
└────────────────────┘                  │                           │
                                        │ Custom Ops (AscendC)      │
                                        │ Triton Kernels (NPU)      │
                                        │ Patch System (两级)        │
                                        │ ACL Graph (piecewise/full)│
                                        └───────────────────────────┘
```

#### 14.1.1 插件注册机制

```python
# setup.py — Python setuptools entrypoint
setup(
    name='vllm-ascend',
    entry_points={
        'vllm.platform_plugins': [
            'ascend = vllm_ascend:register'
        ]
    }
)

# __init__.py — vLLM 启动时自动发现并加载
def register():
    from vllm_ascend.platform import NPUPlatform
    return NPUPlatform
```

#### 14.1.2 NPUPlatform 核心实现

```python
class NPUPlatform(Platform):
    device_name: str = "npu"
    device_type: str = "npu"
    
    @classmethod
    def check_and_update_config(cls, vllm_config):
        """启动时校验并设置 NPU 参数"""
        vllm_config.device_config.device = torch.device("npu")
        # 加载 AscendConfig（attention 后端、量化、图模式等）
        cls._load_ascend_config(vllm_config)
        # 根据硬件代次（A2/A3/A5）计算 ACL Graph 最大 bucket
        update_aclgraph_sizes(vllm_config.compilation_config)
    
    @classmethod
    def get_attn_backend_cls(cls, selected_backend, head_size, ...):
        """根据模型架构返回对应注意力后端"""
        # MLA 模型 → DSA Backend
        # GQA/MHA → AscendAttention (FIA / DFlash)
        return AscendAttentionBackend
    
    @classmethod
    def get_device_communicator_cls(cls):
        return HCCLCommunicator  # NCCL → HCCL 替换
```

### 14.2 两级 Patch 系统

```python
class PatchManager:
    """
    Patch 系统：修改 vLLM 上游行为，避免 fork
    
    Level 1 — Platform Patch（全局，启动时应用）:
      替换模型层中的算子实现
    
    Level 2 — Worker Patch（进程级，Worker 初始化时应用）:
      修改执行流程
    """
    
    @staticmethod
    def apply_platform_patches():
        # Patch RMSNorm → AscendRMSNorm
        import vllm.model_executor.layers.layernorm as ln
        ln.RMSNorm = AscendRMSNorm
        
        # Patch RoPE → AscendRotaryEmbedding
        import vllm.model_executor.layers.rotary_embedding as rope
        rope.RotaryEmbedding = AscendRotaryEmbedding
        
        # Patch 激活函数 → Triton/CANN 实现
        import vllm.model_executor.layers.activation as act
        act.SiluAndMul.forward_native = ascend_silu_and_mul
    
    @staticmethod
    def apply_worker_patches():
        # Patch Worker / ModelRunner 为 NPU 版本
        import vllm.v1.worker.gpu_worker as gw
        gw.Worker = NPUWorker
```

### 14.3 vLLM-Ascend 算子适配清单（v0.18 → v0.20）

| 分类 | GPU 原算子 | NPU 对应实现 | 方式 | 版本 |
|------|-----------|-------------|------|------|
| **Attention (Prefill)** | FlashAttention-2 | `npu_fusion_attention` | CANN API | v0.7+ |
| **Attention (Prefill)** | FlashAttention-2 | `_npu_flash_attention_unpad` | CANN API | v0.19 (A2/A3 升级) |
| **Attention (Decode)** | PagedAttention V1/V2 | FIA (`npu_fused_infer_attention_score`) | CANN API | v0.7+ |
| **Attention (Decode)** | Flash Decoding | DFlash Attention Backend | AscendC | v0.19+ |
| **Attention (MLA)** | FlashMLA | DSA Backend (DeepSeek MLA) | AscendC | v0.11+ |
| **Attention (FA3)** | FlashAttention-3 | FA3 Backend (训推一致) | CANN | v0.20+ |
| **RMSNorm** | `fused_add_rms_norm` | `npu_rms_norm` / Triton | CANN/Triton | v0.7+ |
| **RoPE** | `rotary_embedding` | `npu_apply_rotary_pos_emb` | CANN API | v0.7+ |
| **Activation** | `silu_and_mul` | Triton kernel / `npu_swiglu` | Triton/CANN | v0.7+ |
| **GEMM** | cuBLAS / CUTLASS | aclnn MatMul / `npu_linear` | CANN | v0.7+ |
| **Quantization** | W8A8 (SmoothQuant) | W8A8C8 (含 INT8 KV Cache) | AscendC | v0.13+ |
| **Quantization** | FP8 (H100) | MXFP4 FlatQuant (A5) | AscendC | v0.18+ |
| **Communication** | NCCL | HCCL | HCCL SDK | v0.7+ |
| **Graph** | CUDA Graph | ACL Graph (piecewise + full) | CANN | v0.11+ |
| **MoE** | NCCL AllToAll | FlashComm V1/V2 | AscendC | v0.9+ |

---

## 15. vLLM-Ascend 自定义算子开发实战

### 15.1 AscendC 算子开发（以 RMSNorm 为例）

```cpp
// AscendC 编程模型: GM(全局) → UB(本地) → 计算 → UB → GM
#include "kernel_operator.h"
using namespace AscendC;

class RMSNormKernel {
public:
    __aicore__ inline void Init(
        GM_ADDR x_gm, GM_ADDR weight_gm, GM_ADDR out_gm,
        int32_t hidden_size, float eps)
    {
        this->hidden_size = hidden_size;
        this->eps = eps;
        xGm.SetGlobalBuffer((__gm__ half*)x_gm);
        weightGm.SetGlobalBuffer((__gm__ half*)weight_gm);
        outGm.SetGlobalBuffer((__gm__ half*)out_gm);
        
        // 初始化 UB 队列和缓冲区
        pipe.InitBuffer(inQueueX, 1, hidden_size * sizeof(half));
        pipe.InitBuffer(inQueueW, 1, hidden_size * sizeof(half));
        pipe.InitBuffer(outQueue, 1, hidden_size * sizeof(half));
        pipe.InitBuffer(tmpBuf,   1, hidden_size * sizeof(float));
    }
    
    __aicore__ inline void Process() {
        int32_t row = GetBlockIdx();   // 每个 AI Core 处理一行
        CopyIn(row);                    // Stage 1: GM → UB
        Compute();                      // Stage 2: UB 内计算
        CopyOut(row);                   // Stage 3: UB → GM
    }

private:
    __aicore__ inline void CopyIn(int32_t row) {
        LocalTensor<half> xLocal = inQueueX.AllocTensor<half>();
        DataCopy(xLocal, xGm[row * hidden_size], hidden_size);
        inQueueX.EnQue(xLocal);
        // weight 类似
    }
    
    __aicore__ inline void Compute() {
        LocalTensor<half> xLocal = inQueueX.DeQue<half>();
        LocalTensor<float> tmp = tmpBuf.Get<float>();
        LocalTensor<half> outLocal = outQueue.AllocTensor<half>();
        
        // 1. Cast FP16 → FP32
        Cast(tmp, xLocal, RoundMode::CAST_NONE, hidden_size);
        // 2. x^2
        Mul(tmp, tmp, tmp, hidden_size);
        // 3. mean(x^2)
        float sum_sq = 0.0f;
        ReduceSum(tmp, tmp, hidden_size, sum_sq);
        float rms_inv = 1.0f / sqrtf(sum_sq / hidden_size + eps);
        // 4. x * rms_inv * weight → Cast 回 FP16
        // ... (省略详细步骤)
        
        outQueue.EnQue(outLocal);
        inQueueX.FreeTensor(xLocal);
    }
    
    __aicore__ inline void CopyOut(int32_t row) {
        LocalTensor<half> outLocal = outQueue.DeQue<half>();
        DataCopy(outGm[row * hidden_size], outLocal, hidden_size);
        outQueue.FreeTensor(outLocal);
    }
    
    GlobalTensor<half> xGm, weightGm, outGm;
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> inQueueX, inQueueW;
    TQue<QuePosition::VECOUT, 1> outQueue;
    TBuf<TBufType::VECCALC> tmpBuf;
    int32_t hidden_size; float eps;
};
```

### 15.2 Triton on NPU（跨平台算子）

vLLM-Ascend v0.13+ 支持 Triton NPU 后端，可编写 GPU/NPU 共用算子：

```python
import triton
import triton.language as tl

@triton.jit
def silu_and_mul_kernel(
    output_ptr, input_ptr,
    d: tl.constexpr, BLOCK_SIZE: tl.constexpr,
):
    """SiLU(gate) × up — GPU 和 NPU 共用"""
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < d
    
    gate = tl.load(input_ptr + row * 2 * d + offsets, mask=mask)
    up   = tl.load(input_ptr + row * 2 * d + d + offsets, mask=mask)
    
    gate_f32 = gate.to(tl.float32)
    silu_out = gate_f32 * tl.sigmoid(gate_f32)
    result = silu_out * up.to(tl.float32)
    
    tl.store(output_ptr + row * d + offsets, result.to(gate.dtype), mask=mask)
```

### 15.3 CANN API 算子封装

```python
import torch_npu

class AscendOps:
    """CANN 高性能算子直接调用"""
    
    @staticmethod
    def npu_fusion_attention(query, key, value, head_num, scale, ...):
        """NPU FlashAttention（Prefill）"""
        return torch_npu.npu_fusion_attention(
            query, key, value, head_num=head_num, scale=scale, ...
        )[0]
    
    @staticmethod
    def npu_fused_infer_attention_score(query, k_cache, v_cache,
                                         block_table, context_lens, ...):
        """FIA: NPU PagedAttention（Decode）"""
        return torch_npu.npu_fused_infer_attention_score(
            query, k_cache, v_cache,
            block_table=block_table, context_lens=context_lens, ...
        )
    
    @staticmethod
    def npu_rms_norm(x, weight, eps):
        return torch_npu.npu_rms_norm(x, weight, epsilon=eps)[0]
    
    @staticmethod
    def npu_apply_rotary_pos_emb(query, key, cos, sin):
        return torch_npu.npu_apply_rotary_pos_emb(query, key, cos, sin)
```

### 15.4 Ascend Attention Backend（多后端选择）

```python
class AscendAttentionImpl(AttentionImpl):
    """昇腾注意力实现——根据配置选择内部后端"""
    
    def forward(self, query, key, value, kv_cache, attn_metadata, output=None):
        
        # 1. 写入 KV Cache（NPU 版 reshape_and_cache）
        if kv_cache is not None:
            self._write_kv_cache(key, value, kv_cache, attn_metadata.slot_mapping)
        
        # 2. Prefill → npu_fusion_attention / _npu_flash_attention_unpad
        if attn_metadata.num_prefills > 0:
            prefill_output = AscendOps.npu_fusion_attention(
                query[:attn_metadata.num_prefill_tokens], ...,
                head_num=self.num_heads, scale=self.scale,
                actual_seq_lengths=attn_metadata.prefill_seq_lens,
            )
        
        # 3. Decode → FIA / DFlash（配置选择）
        if attn_metadata.num_decode_tokens > 0:
            if self.use_dflash:
                decode_output = self._dflash_decode(...)  # DFlash 后端
            else:
                decode_output = AscendOps.npu_fused_infer_attention_score(
                    query=..., key_cache=..., value_cache=...,
                    block_table=attn_metadata.block_tables,
                    context_lens=attn_metadata.seq_lens_tensor, ...
                )
        
        return output
```

---

## 16. GPU ↔ NPU 算子迁移方法论

### 16.1 迁移流程

```
Gap Analysis（列出所有 CUDA 算子 ↔ NPU 可用算子）
    ├── 直接映射（CANN API 已覆盖）→ 封装调用
    ├── 需要适配（语义/布局差异）→ 转换层开发
    └── 需要新开发（无对应）→ AscendC / Triton
          ↓
精度验证（单算子 → 单层 → 模型 PPL）
          ↓
性能调优（Profile → 瓶颈分析 → 优化）
          ↓
集成测试 & 上线
```

### 16.2 CUDA → NPU 核心映射

| CUDA 算子/API | NPU 方案 | 关键差异 | 策略 |
|---------------|---------|---------|------|
| `flash_attn_varlen_func` | `npu_fusion_attention` | 参数命名/layout | 封装转换 |
| `paged_attention_v1/v2` | FIA / DFlash | KV Cache 布局 | 适配 block 格式 |
| cuBLAS GEMM | aclnn MatMul | 精度模式 | 直接映射 |
| `__shfl_xor_sync` | AscendC ReduceSum | 无 warp 概念 | 使用 AscendC 原语 |
| NCCL AllReduce | HCCL AllReduce | API 相同 | 直接替换 |
| CUDA Graph | ACL Graph | 捕获/重放/流限制 | 适配 |
| `torch.compile` (Inductor) | Torchair → ACL Graph | 编译后端不同 | NPU compile 路径 |
| Shared Memory (48KB) | UB (Unified Buffer, ~256KB) | 大小/使用方式 | 重新设计分块 |
| Tensor Core (mma) | Cube Unit | 指令集不同 | 交给 CANN 调度 |

### 16.3 KV Cache 布局适配

```python
# GPU vLLM KV Cache（交错布局）:
#   Key:   [num_blocks, num_heads, head_dim/x, block_size, x]
#   Value: [num_blocks, num_heads, head_dim, block_size]

# NPU KV Cache（统一 BSNH）:
#   Key:   [num_blocks, block_size, num_heads, head_dim]
#   Value: [num_blocks, block_size, num_heads, head_dim]

# ★ 最佳实践: 不在推理路径上做 permute/contiguous
# 应在 reshape_and_cache 算子中直接按目标布局写入，一步到位
```

### 16.4 数值精度差异

```python
# 场景1: GEMM 累加精度
# GPU Tensor Core: FP16×FP16 → FP32 累加
# NPU Cube Unit: 部分版本 FP16 累加
# 解决: torch_npu.npu.set_option({"ACL_OP_PRECISION_MODE": "must_keep_origin_dtype"})

# 场景2: 推荐精度阈值（NPU vs GPU）
# Cosine Similarity > 0.999
# Max Abs Diff < 5e-3（比 GPU 对 GPU 略宽）

# 场景3: Softmax 数值稳定
# 确保 attention scale = 1/√head_dim 显式传入
# 某些 NPU API 默认 scale=1.0
```

---

## 17. torch.compile 与 CUDA Graph 适配

### 17.1 vLLM V1 编译架构

```
Forward Pass（每 Transformer Layer）:

  ┌──────────┐   ┌──────────┐   ┌──────────┐
  │ Pre-Attn │   │Attention │   │Post-Attn │
  │ (Linear, │   │ (FA/PA)  │   │ (Linear, │    × N layers
  │  RMSNorm)│   │          │   │  SiLU)   │
  └────┬─────┘   └────┬─────┘   └────┬─────┘
  ← Piece 0 →  ← Eager  →  ← Piece 1 →
  (CUDA Graph)   (Attn)     (CUDA Graph)

Piecewise 模式（默认）:
  • Attention 以外部分捕获为 CUDA Graph
  • Attention eager 执行
  • 兼容所有 attention backends

Full 模式:
  • 整个 forward 含 Attention 全部捕获
  • 要求 backend 支持（FA3 / Triton）
  • 消除全部 kernel launch 开销 → 性能最优
```

### 17.2 自定义 Compiler Pass

```python
# vLLM 在 torch.fx 图上实现自定义融合 Pass

# 示例: SiLU + FP8 Quant 融合
def silu_quant_pattern(x, scale):
    silu = torch.ops.vllm.silu_and_mul(x)
    return torch.ops.vllm.scaled_fp8_quant(silu, scale)

def silu_quant_replacement(x, scale):
    return torch.ops.vllm.silu_quant_fused(x, scale)

register_replacement(silu_quant_pattern, silu_quant_replacement)

# 通信-计算重叠 Pass:
#   Before: matmul → all_reduce → silu
#   After:  matmul → reduce_scatter ─→ silu（与通信重叠）→ all_gather
```

### 17.3 ACL Graph 适配（昇腾版 CUDA Graph）

```python
# ACL Graph 关键差异:
# 1. API: torch.npu.graph() / torch.npu.ACLGraph()
# 2. 流限制: 每个 graph 至少一个独立 stream，上限约 2048
#    → 最多捕获约 1800 个 graph（预留 248 stream buffer）
# 3. A2 (AIV 模式) 与 A3 (FFTS+ 模式) bucket 计算逻辑不同
# 4. Full graph 需用 graph_task_update_begin/end + ExternalEvent 更新 attn 参数

# vLLM-Ascend 的 Graph 策略:
#   Attention 可 graph → Full mode（最优）
#   Attention 不可 graph → Piecewise mode
#   都不行 → Separate prefill/decode + graph per phase
```

---

## 18. 分布式算子与通信适配

### 18.1 vLLM 分布式架构

```
TP (Tensor Parallelism): 同层模型横切，AllReduce
PP (Pipeline Parallelism): 层间纵切，Send/Recv
DP (Data Parallelism): 请求分配到不同 Engine Core
EP (Expert Parallelism): MoE 专家分发，AllToAll

典型部署: TP=8（单机 8 卡）+ PP=1
大规模:   TP=8 + PP=2（跨机）+ DP=2
```

### 18.2 NCCL → HCCL 通信适配

```python
class HCCLCommunicator(CommunicatorBase):
    """torch.distributed API 相同，底层替换为 HCCL"""
    
    def all_reduce(self, tensor, op=ReduceOp.SUM):
        torch.distributed.all_reduce(tensor, op=op, group=self.group)
    
    # 关键差异:
    # 拓扑: NVLink/NVSwitch vs HCCS
    # 带宽: A100 NVLink 600GB/s vs 910B HCCS ~400GB/s
    #        A5 (950) Lingqu 互联 784GB/s（追平 NVLink）
    # AllReduce 算法: Ring vs HD (Halving-Doubling)
```

### 18.3 Expert Parallelism（MoE 适配）

```python
# DeepSeek V3/V4 MoE 在昇腾上的适配要点:

# 1. Router: Linear + TopK → 直接映射
# 2. Expert Dispatch: AllToAll → HCCL AllToAll
# 3. Expert Compute: 分组 GEMM → aclnn GroupedMatMul
# 4. Expert Gather: AllToAll 逆

# 昇腾特有优化:
#   FlashComm V1/V2: 通信-计算重叠（o_shared linear + comm domain fix）
#   EPLB: Expert Load Balancing（动态均衡）
#   IndexCache: 缓存 DSA topk_indices 减少重复计算
```

---

## 19. 量化算子在 vLLM 中的适配

### 19.1 量化方案对比

| 平台 | 方案 | 权重 | 激活值 | KV Cache | 支持模型 |
|------|------|------|--------|----------|---------|
| **GPU (H100)** | FP8 W8A8 | E4M3 | E4M3 | FP8 | 通用 |
| **GPU (通用)** | SmoothQuant | INT8 | INT8 | FP16 | 通用 |
| **GPU (通用)** | GPTQ/AWQ | INT4 | FP16 | FP16 | 通用 |
| **GPU (通用)** | Marlin | INT4 | FP16 | FP16 | W4A16 高效 |
| **NPU (A2/A3)** | W8A8 | INT8 | INT8 | FP16 | LLaMA/Qwen/DS |
| **NPU (A2/A3)** | W8A8C8 | INT8 | INT8 | INT8 | DeepSeek V3 |
| **NPU (A5)** | MXFP4 FlatQuant | MX-FP4 | FP16 | FP16 | 实验性 |
| **NPU (A5)** | w8a8_mxfp8 | INT8 | MX-FP8 | — | Qwen VL |

### 19.2 GPU FP8 量化算子

```python
# H100+ 原生 FP8: E4M3 × E4M3 → FP32 累加
def fp8_gemm(a_fp8, b_fp8, a_scale, b_scale, out_dtype=torch.float16):
    return torch.ops._C.cutlass_scaled_mm(
        a_fp8, b_fp8, a_scale, b_scale, out_dtype, None
    )
```

### 19.3 昇腾 W8A8 量化路径

```python
class AscendW8A8QuantLinear(nn.Module):
    def forward(self, x):
        x_scale = x.abs().max() / 127.0
        x_int8 = (x / x_scale).round().clamp(-128, 127).to(torch.int8)
        return torch_npu.npu_quant_matmul(
            x_int8, self.weight_int8.t(),
            scale=x_scale * self.weight_scale,
            output_dtype=torch.float16,
        )
```

---

## 20. 端到端适配实战案例

### 20.1 案例一：LLaMA-3 70B on 8×昇腾 910B

```bash
# 环境: CANN 8.5+, torch_npu 2.5+, vLLM 0.18+, vllm-ascend 0.18+
pip install vllm-ascend

# 一条命令启动（全部算子已自动适配）
vllm serve meta-llama/Meta-Llama-3-70B-Instruct \
    --tensor-parallel-size 8 \
    --max-model-len 4096 \
    --block-size 16 \
    --gpu-memory-utilization 0.9
```

Gap 分析：GQA / RoPE / RMSNorm / SiLU / PagedAttention 全部已覆盖，**无 gap**。

### 20.2 案例二：DeepSeek-V3/V4 MoE 适配

```
DeepSeek-V4（2026-04 发布，1.6T 参数，49B 活跃，1M 上下文）:

特殊架构:
  • MLA (Multi-Latent Attention) + Compress-4/128-Attention
  • Manifold-Constrained Hyper-Connections (mHC)
  • MoE 256 experts, top-8
  
vLLM-Ascend 适配方案:
  ├── DSA Attention Backend (MLA 专用)
  ├── DSA-CP (压缩注意力, 可独立开关)
  ├── EPLB (Expert Load Balancing)
  ├── FlashComm V2 (通信优化)
  ├── W8A8C8 量化 (三路 INT8)
  ├── PCP & DCP (Prefill-Decode 分离部署)
  ├── MTP (Multi-Token Prediction, MTP>1 支持)
  ├── KV Pool Connector 适配
  └── ACL Graph full mode 支持
  
昇腾 950 (A5) 表现:
  • DeepSeek V4-Pro: ~20ms TPOT
  • DeepSeek V4-Flash: ~10ms TPOT
```

### 20.3 案例三：Qwen3-VL 多模态适配

```
特殊需求:
  1. 视觉编码器 ViT → aclnn BatchMatMulV2（2.7× 卷积加速）
  2. M-RoPE (3D 位置) → qkv_rmsnorm_mrope 融合算子
  3. Flash Comm V1 支持 VL + MLA
  4. EAGLE 推测解码 + 推测解码后端独立选择

量化: 支持 w8a8_mxfp8
```

### 20.4 案例四：GLM-5 / Bailing MoE 适配

```
vLLM-Ascend v0.18+ 新增:
  • GLM-5: W8A8C8 量化 + PD 分离
  • GLM4.7-Flash: W8A8 量化
  • Bailing MoE: linear 适配 + ModelSlim 量化

注意事项:
  • GLM5 在 PD 分离 + TP16 DP2 下 GPQA 精度可能不达标（已知 issue）
  • DeepSeek V3.2 PD 分离存在概率性空输出/乱码（v0.19.1 修复中）
```

---

## 21. 适配工程最佳实践与 Checklist

### 21.1 五大原则

```
1. 优先使用已有高性能算子库
   CANN API > AscendC > Triton > Python Fallback

2. 最小化布局转换
   在 reshape_and_cache 中直接写目标格式，禁止关键路径 permute

3. 分层验证精度
   单算子 → 单层 → Transformer Block → 全模型 PPL → 任务评测

4. 数据驱动优化
   先 Profile（msprof / nsys / ncu）后优化，定位真正瓶颈

5. 保持上游同步
   Patch 系统适配，不 fork；跟随 vLLM 版本 day-1 发布
```

### 21.2 新模型适配五阶段 Checklist

```
【第一阶段：准备】
□ 阅读模型论文 & HuggingFace 源码
□ 列出全部算子，对比已有覆盖
□ 确认 TP/EP 策略 & 精度方案

【第二阶段：算子适配】
□ Attention (MHA / GQA / MLA / Compress-Attention)
□ 位置编码 (RoPE / ALiBi / M-RoPE / YaRN)
□ 归一化 (RMSNorm / LayerNorm / GroupNorm)
□ 激活函数 (SiLU / GeLU / Swish)
□ MoE 路由 + Expert Dispatch（如有）
□ 特殊结构 (MLA / Mamba SSM / mHC / Compress-Attn)

【第三阶段：精度验证】
□ 单算子对比 (vs GPU FP32 参考)
□ 逐层输出 Cosine > 0.999
□ 模型 PPL 差异 < 0.5
□ 任务评测 (MMLU / HumanEval / GPQA)
□ 长序列稳定性 (4K / 8K / 32K / 128K / 1M)

【第四阶段：性能验证】
□ Prefill 吞吐 (tokens/s)
□ Decode 延迟 (ms/token, TPOT)
□ TP 扩展效率
□ Continuous Batching 吞吐
□ ACL Graph / CUDA Graph 加速效果
□ 与 GPU 同配置性能对比

【第五阶段：集成 & 上线】
□ vLLM serve 启动正常
□ OpenAI API 兼容 & 流式输出
□ 多并发 + OOM 边界测试
□ 24h+ 稳定性
□ 异常输入处理 (空/超长/特殊 token)
□ 性能数据 + 已知限制 + 部署配置文档
□ 监控告警 (显存使用率 / 吞吐 / 队列深度)
```

### 21.3 性能调优路径

```
性能不达标
    │
    ▼
Profile 分析（nsys / msprof / ncu）
    │
    ├── 计算瓶颈 → GEMM 调优 / 增大 Tile / 量化加速
    ├── 访存瓶颈 → 算子融合 / 向量化 / KV Cache 布局优化
    └── 通信瓶颈 → FlashComm / 通信重叠 / TP 策略
    │
    ▼
ACL Graph / CUDA Graph（消除 launch 开销）
    │
    ▼
再次 Profile → 验证效果 → 迭代
```

### 21.4 FAQ

**Q1: vLLM-Ascend 支持哪些芯片？**
A: A2（昇腾 910B）、A3（910C）、A5（950）。310P 设备需 CANN 9.0。

**Q2: 性能不及 GPU 如何排查？**
A: `msprof` 抓算子耗时，对比 GPU ncu。常见瓶颈在 Attention decode 和 AllReduce。可尝试 DFlash 后端、FlashComm V2、ACL Graph full mode。

**Q3: 精度问题定位方法？**
A: 逐层输出对比找首个劣化层。常见原因：RoPE 累加用 FP32、Softmax scale 显式传入、量化参数格式不匹配。

**Q4: ACL Graph 捕获失败？**
A: 检查动态 shape 操作、不支持 graph 的算子。注意流上限约 2048（graph 上限 ~1800）。可用 piecewise 排除 attention。A2/A3 bucket 计算逻辑不同。

**Q5: 如何跟随 vLLM 新版本？**
A: vLLM-Ascend 目标 day-1 发布跟随 vLLM 版本。每个 vLLM 版本对应 `releases/vX.Y.Z` 开发分支。

---

> **本手册 Part I + Part II 共同构成完整的大模型算子适配技术体系。**
>
> **Part I**（第 1-9 章）：算子基础、开发流程、CUDA/ROCm/昇腾/寒武纪/Intel 多平台适配通用技术
> **Part II**（第 10-21 章）：vLLM 框架深度解析、vLLM-Ascend 插件适配、GPU/NPU 迁移实战
>
> 参考资源：
> - vLLM 官方文档：https://docs.vllm.ai
> - vLLM-Ascend 文档：https://docs.vllm.ai/projects/ascend
> - vLLM-Ascend 源码：https://github.com/vllm-project/vllm-ascend
> - vLLM V1 Blog：https://blog.vllm.ai/2025/01/27/v1-alpha-release.html
> - Hardware Plugin Blog：https://blog.vllm.ai/2025/05/12/hardware-plugin.html
> - Triton Backend Blog：https://vllm.ai/blog/2026-03-04-vllm-triton-backend-deep-dive
> - torch.compile Blog：https://blog.vllm.ai/2025/08/20/torch-compile.html