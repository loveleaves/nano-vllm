# Automatic Prefix Caching（APC）技术调研报告

> **报告时间**：2026 年 6 月  
> **参考来源**：vLLM 官方文档（docs.vllm.ai）、学术论文、开源社区实践

---

## 核心内容概要

1. **背景与动机** — LLM 推理瓶颈分析，重复前缀的普遍性（System Prompt、RAG、多轮对话等）

2. **核心原理** — Block 哈希计算机制，与 PagedAttention 的关系

3. **vLLM 的 APC 实现** — 数据结构（哈希表+LRU双向链表）、Block 分配/释放/淘汰三大核心操作、V1 架构改进（批次不变性、默认开启等）

4. **技术路线全景（四大方向）**
   - 本地 GPU/CPU 缓存
   - 分布式 KV 缓存（LMCache、Mooncake、HF3FS 等）
   - 跨节点 KV 迁移（Disaggregated Prefill + NixlConnector）
   - 持久化磁盘缓存（IndexCache）

5. **典型应用场景** — 固定 System Prompt、多轮对话、RAG、代码补全、批量推理

6. **关键限制** — Block 对齐、采样参数兼容性、显存权衡、多 LoRA、分布式复杂性

7. **业界实现对比** — vLLM/TensorRT-LLM/SGLang/LMDeploy/Ollama/OpenAI/Anthropic 横向对比表

8. **前沿进展** — 语义缓存、推测预取、KV 量化+APC、MLA 与 APC 协同

9. **配置与使用指南** — 开箱即用的代码示例和最佳实践

10. **总结与选型建议** — 按场景推荐方案的决策矩阵

---

## 1. 背景与动机

### 1.1 LLM 推理的计算瓶颈

大型语言模型（LLM）推理分为两个阶段：

- **Prefill（预填充）**：对输入 Prompt 中所有 token 并行计算 KV（Key-Value）Cache，计算量大、延迟高。
- **Decode（解码）**：逐 token 自回归生成，受内存带宽限制。

Transformer 的 Attention 机制要求每个新 token 都能"看到"所有历史 token 的 K、V 向量，因此这些中间结果必须被保存在 KV Cache 中。对于一个 70B 参数的模型、8k 上下文、FP16 精度，单个序列的 KV Cache 可达数 GB。

### 1.2 重复前缀的普遍性

在实际部署中，大量请求共享相同的前缀内容：

- **系统提示（System Prompt）**：几乎所有请求都携带相同的指令/角色设定。
- **RAG 场景**：检索到的文档片段被反复注入到 Prompt 中。
- **多轮对话**：每轮新请求都包含完整的历史对话。
- **Few-shot 示例**：相同的示例在不同问题中重复使用。
- **代码补全**：相同的代码文件前缀被反复送入模型。

如果每次请求都重新计算这些相同前缀的 KV Cache，将造成巨大的计算浪费。**Automatic Prefix Caching（APC）** 正是为解决这一问题而生。

---

## 2. 核心原理

### 2.1 基本思想

APC 的核心思想是：**将 KV Cache 按照固定大小的 Block 组织，并对每个 Block 计算一个唯一的哈希标识（Block Hash）。当新请求到来时，如果其 Prompt 前缀对应的 Block 已在缓存中，则直接复用，跳过 Prefill 计算。**

```
请求 A: [System Prompt] [User Query A]
                ↓
         计算完整 KV Cache，缓存 Block

请求 B: [System Prompt] [User Query B]
                ↓
         System Prompt 部分 → 缓存命中，直接复用！
         User Query B 部分 → 只需计算新增 token
```

### 2.2 Block 哈希计算

Block Hash 通常由以下要素决定：

```
block_hash = hash(
    parent_block_hash,   // 父 Block 的哈希（保证前缀连续性）
    token_ids            // 当前 Block 内所有 token 的 ID 序列
)
```

这种链式哈希设计确保：两个 Block 内容相同，但父 Block 不同，则哈希不同，从而精确区分前缀上下文。

### 2.3 与 PagedAttention 的关系

APC 建立在 vLLM 的 **PagedAttention** 机制之上。PagedAttention 将 GPU 显存中的 KV Cache 划分为固定大小的"物理块（Physical Block）"，通过"页表"映射到逻辑序列位置。APC 在此基础上增加了以 Block Hash 为键的缓存查找层。

---

## 3. vLLM 的 APC 实现

### 3.1 数据结构设计

vLLM 的 APC 核心数据结构是一个**哈希表 + 双向链表**组合，实现带有 LRU（Least Recently Used）淘汰策略的缓存。

```
evictor（已完成序列的 block 池）:
  ┌─────────────────────────────────┐
  │  hash_to_block: Dict[int, Block]│  ← 哈希表，O(1) 查找
  │  free_table:    OrderedDict     │  ← LRU 顺序，O(1) 淘汰
  └─────────────────────────────────┘

每个 Block 包含:
  - block_hash: int          # 唯一标识
  - ref_count: int           # 引用计数（>0 表示正在被使用）
  - content_hash: int        # 内容哈希
  - computed: bool           # 是否已完成 KV 计算
```

Block 的生命周期状态：

```
[Free] → [Allocated] → [Computed/Cached] → [Evicted]
           ↑                  ↓
           └──── 命中复用 ────┘
```

### 3.2 核心操作

**Block 分配（Allocation）**

1. 调度器将新请求的 Prompt token 序列切分为固定大小的 Block（默认 16 token/block）。
2. 对每个 Block 计算哈希值。
3. 在 `hash_to_block` 中查找：
   - **命中（Cache Hit）**：增加引用计数，直接映射到已有物理块，跳过计算。
   - **未命中（Cache Miss）**：从空闲池分配新物理块，或触发 LRU 淘汰。

**释放（Free）**

序列生成完毕后，关联的 Block 引用计数减 1。当 `ref_count == 0` 时，Block 移入 `evictor` 的 `free_table`（LRU 尾部），等待可能的命中复用或淘汰。

**LRU 淘汰（Eviction）**

当空闲物理块不足时，从 `free_table` 头部（最久未使用）取出 Block，清除其哈希映射，将物理块归还空闲池。

### 3.3 vLLM V1 的改进

vLLM V1 架构（2024 年末引入）对 APC 进行了大幅重构：

- **HashBasedEvictor**：改用纯哈希表实现，去除了 V0 中的链式 Block 依赖，降低了元数据管理开销。
- **KVCacheManager 统一管理**：将 APC 逻辑整合到新的缓存管理器中，支持更灵活的多级存储（GPU → CPU → 磁盘）。
- **批次不变性（Batch Invariance）**：保证在相同输入下，无论批次大小，输出数值完全一致，这在 V0 中 APC 开启时并不总能保证。
- **默认开启**：V1 中 APC 默认启用（`enable_prefix_caching=True`）。

### 3.4 示例流程

```
Block Size = 4 tokens

请求序列: [A, B, C, D, E, F, G, H]

Block 0: [A, B, C, D]  hash_0 = hash(0, [A,B,C,D])
Block 1: [E, F, G, H]  hash_1 = hash(hash_0, [E,F,G,H])

新请求: [A, B, C, D, X, Y, Z, W]
  Block 0: hash = hash(0, [A,B,C,D]) → 命中！复用物理块
  Block 1: hash = hash(hash_0, [X,Y,Z,W]) → 未命中，分配新块
  → 节省 50% Prefill 计算
```

---

## 4. 技术路线全景

APC 技术路线可按缓存存储层次分为四大类：

### 4.1 本地内存缓存（GPU/CPU）

这是最基础的路线，也是 vLLM 默认实现的形式。

**GPU 显存缓存**

- Block 直接保留在 GPU 显存中，命中时零拷贝复用。
- 受显存容量限制，通常只能缓存数百至数千个 Block。
- LRU 淘汰是主流策略，部分实现引入频率感知（LFU）或混合策略。

**CPU 内存卸载（KV Offloading）**

- 当 GPU 显存不足时，将 LRU Block 换出到 CPU RAM。
- 命中时通过 PCIe 将数据传回 GPU，引入传输延迟（通常 10-50ms）。
- vLLM 提供 `--kv-transfer-config` 配置 CPU 卸载连接器（`SimpleCPUOffloadConnector`）。

```
存储层次:  GPU VRAM → CPU DRAM → NVMe SSD
延迟量级:  μs        ms         10s ms
容量量级:  数十 GB   数百 GB    数 TB
```

### 4.2 分布式 KV 缓存

随着部署规模增大，单节点缓存命中率受限，需要跨节点共享。

**集中式 KV 存储（Centralized KV Store）**

部署独立的 KV Cache 服务器，多个推理实例共享：

- **LMCache**：vLLM 官方支持的外部 KV Cache 库，支持 Redis / 本地存储后端，通过 `LMCacheConnector` 集成。
- **Mooncake（月饼，字节跳动）**：以 DRAM 为主体的分布式 KV 缓存系统，使用 RDMA 高速互联，支持 prefill/decode 实例间的 KV 共享。vLLM 提供 `MooncakeConnector` 和 `MooncakeStoreConnector`。
- **MoRIIO**：面向多节点场景的 KV 路由与缓存服务，vLLM 提供 `MoRIIOConnector`。

**分布式内存文件系统**

- **HF3FS（火山引擎三火文件系统）**：字节跳动开源的高性能并行文件系统，vLLM 集成了 `HF3FSConnector`，支持将 KV Cache 存储到共享分布式文件系统中。

### 4.3 跨节点 KV 迁移（Disaggregated Prefill）

这是近年来最热门的技术路线之一，将 **Prefill** 和 **Decode** 解耦到不同节点。

**动机**：Prefill 是计算密集型（Compute-bound），Decode 是内存带宽密集型（Memory-bound），混合部署导致两者互相干扰。

**工作流程**：

```
┌─────────────────┐    KV Transfer     ┌─────────────────┐
│  Prefill 节点   │ ──────────────────→│  Decode 节点    │
│ (高算力 GPU)    │                    │ (多副本 GPU)    │
│ 计算 KV Cache   │                    │ 接收并复用 KV   │
└─────────────────┘                    └─────────────────┘
```

**vLLM 的实现**：

- 通过 `KVTransferConfig` 和 KV Connector 框架支持。
- **NixlConnector**：基于 NVIDIA NIXL（NCCL Interconnect eXtension Library）实现高带宽 KV 传输，支持 Push/Pull 两种模式。
- **FlexKV**：更灵活的 KV 传输框架，支持自定义传输协议。

**APC 与 Disaggregated Prefill 的协同**：当 Decode 节点接收来自 Prefill 节点的 KV 后，同样可以将其注册到本地 APC 缓存，后续相同前缀的请求无需再次触发跨节点 Prefill。

### 4.4 持久化 / 磁盘缓存

将 KV Cache 持久化到 SSD/NVMe，实现跨服务重启的缓存复用：

- **IndexCache（vLLM 实验特性）**：将 Block 的哈希索引和 KV 数据序列化到磁盘，重启后可直接加载。
- **适用场景**：System Prompt 极长（如百万 token）且内容固定的场景（法律文档、代码库、书籍全文）。
- **局限**：磁盘 I/O 延迟远高于 GPU 计算，需要异步预加载配合使用。

---

## 5. 典型应用场景

### 5.1 固定 System Prompt

```
所有请求:
┌──────────────────────────────┬─────────────────┐
│ System Prompt (1000 tokens)  │ User Query (变化)│
└──────────────────────────────┴─────────────────┘
        ↑ 每次命中，节省大量计算
```

典型收益：系统提示占比越高，加速效果越显著。1000 token 系统提示 + 100 token 用户查询的场景，APC 可将 Prefill 时间降低 90%。

### 5.2 多轮对话

```
Turn 1: [Sys][User1][Asst1]
Turn 2: [Sys][User1][Asst1][User2][Asst2]
Turn 3: [Sys][User1][Asst1][User2][Asst2][User3]
            ↑─────────────────────────────↑
            历史部分命中，只需计算新增 turn
```

### 5.3 RAG（检索增强生成）

```
┌────────────┬─────────────────────────┬──────────────┐
│ Sys Prompt │ Retrieved Docs (固定集) │ User Question│
└────────────┴─────────────────────────┴──────────────┘
              ↑ 文档内容相同时可命中缓存
```

当检索结果高度重叠时（如来自同一文档库），APC 效果显著。

### 5.4 代码补全 / Agent 场景

- 代码补全：相同文件前缀反复被送入模型，APC 命中率极高。
- Agent 工具调用：System Prompt + 工具描述部分固定，APC 节省每步骤的重复计算。

### 5.5 批量推理（Batch Inference）

对同一数据集进行批量分析时，若每条数据的 Prompt 前缀相同（如"请分析以下内容：…"），APC 可大幅提升整体吞吐量。

---

## 6. 关键限制与挑战

### 6.1 Token 粒度对齐要求

APC 要求前缀必须在 Block 边界对齐。若 Prompt 的共享部分恰好不是 `block_size` 的整数倍，则最后一个不完整 Block 无法缓存。

**解决方案**：vLLM V1 引入了对最后一个不完整 Block 的部分缓存支持（Partial Block Caching）。

### 6.2 采样参数影响

含温度（temperature > 0）的随机采样请求，其 KV Cache 理论上与确定性输出相同，可以复用。但部分实现（尤其 V0）在非贪婪采样时禁用 APC，以避免与束搜索（beam search）的兼容性问题。

**vLLM 现状**：V1 中对大多数采样参数（包括 temperature > 0）都支持 APC，但 `prompt_logprobs` 模式下存在限制。

### 6.3 显存压力与命中率权衡

保留更多历史 Block 以提升命中率，会减少可用于新序列的显存，可能降低并发数（并发度）。需根据实际流量模式调整 `gpu_memory_utilization` 和 `max_model_len`。

### 6.4 多 LoRA 适配器

不同 LoRA 适配器产生的 KV Cache 不可互用（因为 LoRA 改变了 Attention 权重）。Block 哈希需要包含 LoRA ID 以区分，这降低了多适配器场景下的缓存利用率。

### 6.5 分布式并行的复杂性

在张量并行（Tensor Parallelism）和流水线并行（Pipeline Parallelism）下，KV Cache 跨 GPU 分片，跨节点 APC 需要协调多个设备的 Block 分配，实现难度显著增加。

### 6.6 冷启动问题

服务重启后缓存清空，需要预热（warm-up）阶段才能建立有效缓存。持久化缓存（IndexCache）可缓解此问题，但带来额外的磁盘管理开销。

---

## 7. 业界实现对比

| 框架 / 系统       | APC 默认状态 | 存储层次           | 分布式支持      | 淘汰策略         | 特色                              |
|-------------------|:------------:|--------------------|-----------------|-----------------|-----------------------------------|
| **vLLM V1**       | ✅ 默认开启  | GPU + CPU Offload  | LMCache / Mooncake | LRU          | 生态最完整，Connector 插件化      |
| **vLLM V0**       | ❌ 需手动开启| GPU                | 有限支持        | LRU             | 成熟稳定                          |
| **TensorRT-LLM**  | ✅ 默认开启  | GPU                | 部分支持        | LRU             | NVIDIA 官方，性能极致             |
| **SGLang**        | ✅ 默认开启  | GPU + CPU          | RadixAttention  | LRU + 频率感知  | RadixAttention 树形前缀复用       |
| **LMDeploy**       | ✅ 支持      | GPU                | 有限            | LRU             | 轻量高效                          |
| **Ollama**        | ✅ 支持      | CPU/GPU            | ❌              | 简单 LRU        | 面向边缘，资源占用低              |
| **OpenAI API**    | ✅（内部）   | 云端分布式         | ✅              | 未公开          | 自动前缀缓存，对用户透明          |
| **Anthropic API** | ✅ 显式      | 云端分布式         | ✅              | 未公开          | Prompt Caching 需显式标注缓存断点 |

**SGLang 的 RadixAttention** 值得特别关注：它将所有请求的前缀组织为一棵 **Radix Tree（前缀树）**，每个节点对应一段 token 序列的 KV Cache，不同请求共享公共前缀节点，自然地实现了细粒度的前缀复用，无需 Block 对齐。

---

## 8. 前沿进展

### 8.1 跨请求批次的 APC（Cross-Batch Prefix Sharing）

传统 APC 在序列级别复用，最新研究探索在同一批次内多个序列之间实时共享 KV，进一步减少冗余计算。

### 8.2 语义级别缓存

基于 token 序列的精确哈希匹配要求输入完全一致。语义缓存通过将 Prompt 嵌入到向量空间，对相似（但不完全相同）的输入也尝试复用 KV Cache。这是研究热点，尚未进入主流部署。

### 8.3 推测性预取（Speculative Prefetching）

在 Decode 阶段，利用当前生成内容预测下一轮对话的可能前缀，提前从磁盘 / 远端加载对应 KV Block，实现零等待命中。

### 8.4 KV Cache 压缩与 APC 联合优化

将 KV Cache 量化（INT8/FP8）或稀疏化后缓存，在相同显存下存储更多 Block，提升命中率。vLLM 的 `QuantizedKVCache` 特性已支持 FP8 量化 KV，与 APC 兼容。

### 8.5 长上下文优化

对于百万 token 级别的超长文档（如 Gemini 1.5、Qwen2.5 的扩展上下文），APC 的价值更为突出。字节的 Mooncake、Google 的 Infini-Attention 均在此方向有所探索。

### 8.6 MLA（Multi-head Latent Attention）与 APC

DeepSeek 系列模型采用 MLA 压缩 KV Cache，使每 token 的 KV 存储量大幅降低（降至标准 MHA 的 5-13%）。MLA 与 APC 的结合使得在相同显存下可缓存更多前缀，进一步提升命中率。

---

## 9. 配置与使用指南

### 9.1 vLLM 开启 APC

**离线推理**：

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3-8b-instruct",
    enable_prefix_caching=True,   # V0 需显式开启；V1 默认开启
    gpu_memory_utilization=0.9,
    max_model_len=32768,
)
```

**在线服务**：

```bash
vllm serve meta-llama/Llama-3-8b-instruct \
    --enable-prefix-caching \
    --gpu-memory-utilization 0.9
```

### 9.2 查看缓存命中率

vLLM 提供 Prometheus 指标：

```
vllm:gpu_prefix_cache_hit_rate   # GPU 缓存命中率
vllm:cpu_prefix_cache_hit_rate   # CPU 缓存命中率（开启 offload 时）
```

### 9.3 CPU KV Offloading 配置

```python
llm = LLM(
    model="...",
    enable_prefix_caching=True,
    kv_transfer_config={
        "kv_connector": "SimpleCPUOffloadConnector",
    }
)
```

### 9.4 LMCache 分布式缓存配置

```python
llm = LLM(
    model="...",
    kv_transfer_config={
        "kv_connector": "LMCacheConnector",
        "kv_connector_extra_config": {
            "lmcache_config_file": "/path/to/lmcache.yaml"
        }
    }
)
```

### 9.5 最佳实践

- **System Prompt 置于 Prompt 头部**：确保共享内容在所有请求的 Block 对齐位置一致。
- **避免在 System Prompt 中插入时间戳或随机数**：会破坏哈希匹配。
- **多轮对话保持连续性**：传入完整历史，而非截断后的部分历史。
- **监控 Cache Hit Rate**：低于 30% 时考虑调整 Prompt 结构或增加缓存容量。
- **长文档预加载**：对于固定的知识库文档，可在服务启动时预先计算并缓存 KV。

---

## 10. 总结与展望

### 10.1 技术总结

Automatic Prefix Caching 是当前 LLM 推理优化中最具实用价值的技术之一，其核心价值在于：

- **零侵入性**：对用户完全透明，无需修改 Prompt 格式（除特殊实现外）。
- **通用性**：适用于几乎所有 Transformer 架构的生产场景。
- **收益显著**：在高前缀重复率场景下，可将 TTFT（Time To First Token）降低 50%-95%，整体吞吐量提升 2-10 倍。

### 10.2 技术演进方向

```
基础 APC              分布式 APC              语义 APC
(单节点 GPU)    →    (跨节点共享)     →    (近似匹配)
     ↓                    ↓                    ↓
 已成熟                进行中              研究阶段

持久化缓存            压缩 KV 缓存          主动预取
(跨重启复用)    →    (量化 + APC)     →    (推测性加载)
     ↓                    ↓                    ↓
 实验阶段              已部分实现           研究阶段
```

### 10.3 选型建议

| 场景                         | 推荐方案                                 |
|------------------------------|------------------------------------------|
| 单机部署，System Prompt 固定 | vLLM V1（默认 APC）                     |
| 多机部署，高并发             | vLLM + LMCache / Mooncake               |
| 超长文档（>100k tokens）     | Disaggregated Prefill + APC + 持久化    |
| 边缘部署，资源受限           | Ollama / LMDeploy + 本地 APC            |
| 极致性能，NVIDIA 平台        | TensorRT-LLM + 原生 APC                 |

随着 LLM 上下文窗口持续扩大（从 4k 到 1M token）、多轮对话和 Agent 场景日益普及，APC 及其衍生技术将成为高效 LLM 推理基础设施中不可或缺的核心组件。

---

## 参考资料

1. vLLM 官方文档 - Automatic Prefix Caching: https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/
2. vLLM 设计文档 - Prefix Caching: https://docs.vllm.ai/en/latest/design/prefix_caching/
3. vLLM 设计文档 - Paged Attention: https://docs.vllm.ai/en/latest/design/paged_attention/
4. vLLM 设计文档 - Hybrid KV Cache Manager: https://docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager/
5. SGLang RadixAttention: https://lmsys.org/blog/2024-01-17-sglang/
6. Efficient Memory Management for Large Language Model Serving with PagedAttention (Kwon et al., SOSP 2023)
7. Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving (ByteDance, 2024)
8. LMCache: https://github.com/LMCache/LMCache
9. DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model (MLA 设计)
10. Anthropic Prompt Caching: https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching