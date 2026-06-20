# CUDA Graph 设计与实现对比调研报告：nano-vllm vs 开源 vLLM

**调研日期**：2026-06-14
**调研主体**：当前工程 nano-vllm（phase4 分支）与开源 vLLM 的 CUDA Graph 设计与实现
**调研意图**（技术向）：拆解当前工程 CUDA Graph 的捕获/replay 机制、静态张量管理、batch size 分桶与 attention 协同，与生产级 vLLM 逐层对比，识别简化取舍、正确性边界与可演进方向。
**代码基准**：
- nano-vllm：`nanovllm/engine/model_runner.py`（`capture_cudagraph` / `run_model`）、`nanovllm/utils/context.py`、`nanovllm/layers/attention.py`（commit `f8d495d`）
- vLLM：`vllm/compilation/cuda_graph.py`、`vllm/config/compilation.py`（本地 checkout `/home/cb/work/vllm/vllm`）

---

## 一、执行摘要

1. **核心机制同源**：两者都遵循 CUDA Graph 的标准范式——「预分配静态输入/输出张量 → warmup 一次 → `torch.cuda.graph` 捕获 → replay 时只改静态张量数据」。nano-vllm 的 `capture_cudagraph` 是这套范式的最小直接实现，vLLM 的 `CUDAGraphWrapper` 是同一范式的工程化封装。

2. **最大差异是「捕获粒度模式」**：nano-vllm 只有一种模式——**整图捕获、仅 decode**（等价 vLLM 的 `FULL_DECODE_ONLY`）。vLLM 有 5 种模式：`NONE / PIECEWISE / FULL / FULL_DECODE_ONLY / FULL_AND_PIECEWISE`（v1 默认后者），其中 **PIECEWISE**（在 attention 处切开、只捕获 attention 之间的算子片段）让 **prefill/变长批次也能用 graph**——这是 nano-vllm「prefill 一律 eager」无法做到的。

3. **dispatch key 粒度不同**：nano-vllm 用 **batch size** 单维分桶（`graph_bs = [1,2,4,8,16,32,...,512]`），replay 时 `next(x for x in graph_bs if x >= bs)` 向上 padding 到最近桶。vLLM 用 **`BatchDescriptor`（num_tokens + 是否 uniform）** 作 key，能区分纯 decode 与 mixed prefill-decode 批次，并各自走 FULL/PIECEWISE。

4. **静态缓冲管理：手工 vs 解耦**：nano-vllm 把所有静态张量塞进一个 `graph_vars` dict，在 `run_model` 里手动 `gv["x"][:bs] = ...` 拷入再 replay。vLLM 的 `CUDAGraphWrapper` **刻意不管理持久缓冲**（注释明言），把「拷数据进静态 buffer」的职责留给 runner，wrapper 只负责按 key 捕获/replay，保持与 compile 逻辑正交。

5. **与 torch.compile 的关系**：nano-vllm 纯手写、零 compile 依赖。vLLM 的 CUDA Graph 与 **Inductor 编译深度耦合**——PIECEWISE 模式下，是 compile 后端把 fx graph 在 attention 处切分并对每段套 `CUDAGraphWrapper`，还叠加 RMSNorm+quant、SiluMul+quant、allreduce 融合等 pass。两者是「手写脚本」与「编译器基础设施」的代差。

**核心结论**：nano-vllm 的 CUDA Graph 是一份**正确、精炼、抓住本质的 decode 整图实现**——静态张量池、从大到小捕获共享 memory pool、batch size 向上 padding、attention 元数据全放 GPU 张量以支持整图捕获，这些关键点都做对了，单卡/多卡 decode 都能消除 Python 调度开销。它与 vLLM 的差距不在「能否用 graph」，而在 **prefill 能否用 graph（PIECEWISE）、多场景 dispatch（BatchDescriptor）、与编译器/融合的协同**。对纯 decode 加速足够，但 prefill-heavy 或需要算子融合的场景与 vLLM 有量级差距。

---

## 二、背景与概述

### 2.1 CUDA Graph 为什么能加速 LLM 推理

LLM decode 阶段每步只处理 1 个 token/seq，单步 GPU kernel 极多但每个都很小，**CPU 端的 Python 调度 + kernel launch 开销**往往超过 GPU 实际计算时间（launch-bound）。CUDA Graph 把一整串 kernel 录制成一个可重放的图，replay 时**一次提交、零 Python 开销、零 per-kernel launch 开销**，对 decode（小 batch、固定结构）收益最大。

代价是 **graph 要求形状静态**：捕获时的张量地址、shape 在 replay 时不能变，只能改张量里的数据。这与 LLM 的动态特性（变长 prompt、变长 batch）天然冲突，是所有实现的核心矛盾。

### 2.2 两个工程定位

| 维度 | nano-vllm | vLLM |
|---|---|---|
| 捕获模式 | 仅整图、仅 decode | NONE/PIECEWISE/FULL/混合 5 种 |
| prefill 用 graph | ❌（一律 eager） | ✅（PIECEWISE） |
| dispatch key | batch size | BatchDescriptor(tokens, uniform) |
| 静态缓冲 | 手工 graph_vars dict | runner 管理，wrapper 解耦 |
| torch.compile | ❌ | ✅ 深度集成 + 融合 pass |
| 算子融合 | ❌ | ✅ RMSNorm/SiluMul/AllReduce 融合 |
| 代码量 | ~70 行 | 跨 compilation/ 多文件数千行 |

---

## 三、nano-vllm 的设计与实现

### 3.1 捕获流程（`model_runner.py:269-310`）

```python
@torch.inference_mode()
def capture_cudagraph(self):
    max_bs = min(config.max_num_seqs, 512)
    max_num_blocks = (config.max_model_len + block_size - 1) // block_size

    # ① 预分配静态张量（按 max_bs）
    input_ids   = torch.zeros(max_bs, dtype=int64)
    positions   = torch.zeros(max_bs, dtype=int64)
    slot_mapping = torch.zeros(max_bs, dtype=int32)
    context_lens = torch.zeros(max_bs, dtype=int32)
    block_tables = torch.zeros(max_bs, max_num_blocks, dtype=int32)
    outputs     = torch.zeros(max_bs, hidden_size)

    # ② 分桶
    self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))

    # ③ 从大到小捕获（第一个建 pool，其余共享）
    for bs in reversed(self.graph_bs):
        graph = torch.cuda.CUDAGraph()
        set_context(False, slot_mapping[:bs], context_lens[:bs], block_tables[:bs])
        outputs[:bs] = self.model(input_ids[:bs], positions[:bs])      # warmup（必须）
        with torch.cuda.graph(graph, self.graph_pool):
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])  # 捕获
        if self.graph_pool is None:
            self.graph_pool = graph.pool()                            # 首图建池
        self.graphs[bs] = graph
        torch.cuda.synchronize()
        reset_context()

    self.graph_vars = dict(input_ids=..., positions=..., slot_mapping=...,
                           context_lens=..., block_tables=..., outputs=...)
```

四个关键设计点：
- **warmup 必须**：捕获前先跑一次同样的 forward，让 cuDNN/cublas 选好算法、分配好 workspace，否则捕获会把算法选择也录进去或报错。
- **从大到小捕获**：先捕获 max_bs，建立 memory pool；后续小 bs 共享同一 `graph_pool`，避免每个 graph 各占一份显存（碎片）。
- **静态张量按 max_bs 预分配，捕获用 `[:bs]` 切片**：所有 bs 共享同一块底层显存的前缀，replay 时地址不变。
- **`block_tables[max_bs, max_num_blocks]`**：这是最大的静态显存占用——max_model_len=4096、block_size=256 时每行 16 个 int32，但若 block_size 小则会显著膨胀。

### 3.2 Replay 流程（`model_runner.py:242-257`）

```python
def run_model(self, input_ids, positions, is_prefill):
    if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
        return self.model.compute_logits(self.model(input_ids, positions))  # eager 路径

    bs = input_ids.size(0)
    context = get_context()
    graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]   # 向上取最近桶
    gv = self.graph_vars
    gv["input_ids"][:bs] = input_ids            # ① 拷入新数据
    gv["positions"][:bs] = positions
    gv["slot_mapping"].fill_(-1); gv["slot_mapping"][:bs] = context.slot_mapping
    gv["context_lens"].zero_();   gv["context_lens"][:bs] = context.context_lens
    gv["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
    graph.replay()                              # ② 重放
    return self.model.compute_logits(gv["outputs"][:bs])  # ③ 从静态 outputs 读结果
```

- **向上 padding**：bs=5 用 bs=8 的 graph，多算 3 行废数据但形状匹配。
- **三段式**：拷入 → replay → 读出。`fill_(-1)/zero_()` 先清残留（padding 行的 slot=-1 不写 KV，context_len=0 不参与 attention），保证废行不污染。
- **三条 eager 逃逸**：prefill、`enforce_eager`、bs>512 都绕过 graph。

### 3.3 整图捕获为何对 decode 可行（关键洞察）

CUDA Graph 要求形状静态，但 decode 的 KV 长度、block table 内容每步都在变。nano-vllm 能**把 attention（`flash_attn_with_kvcache`）也捕进图**的原因是：

> **所有「动态」信息都编码在 GPU 张量的内容里，而非 shape 里。**

`flash_attn_with_kvcache(q, k_cache, v_cache, cache_seqlens=context_lens, block_table=block_tables)` 里，变长 KV 由 `cache_seqlens`（GPU int32 张量）在 **kernel 运行时**读取，`block_table` 同理。捕获时固定的只有 `bs` 和 `max_num_blocks` 两个 shape 维度。replay 时改这些 GPU 张量的**数据**即可表达不同的序列长度/块映射——shape 不变，所以图有效。

这正是 decode 能整图、prefill 不能的根因：**prefill 的 total_tokens、cu_seqlens 长度随 batch 变化是 shape 级变化**，无法用固定 shape 表达，故 nano-vllm 让 prefill 走 eager。

### 3.4 Context 与 graph 的配合（`context.py`）

nano-vllm 用全局 `_CONTEXT` 隐式传递 attention 元数据。捕获时 `set_context(..., block_tables[:bs])` 让 attention 记录读取静态张量地址；捕获结束 `reset_context()`。replay 时 attention 读到的仍是捕获时 baked-in 的静态地址（graph 特性），`run_model` 只需把新数据写进这些静态张量——`prepare_decode` 设置的临时 Context 仅用于「提供拷贝源」，不影响已捕获的图。设计自洽。

### 3.5 与 TP 的协同

每个 rank 独立 `capture_cudagraph`，graph 内部含 RowParallel 的 `dist.all_reduce`——NCCL all-reduce 被直接录进图。replay 时各 rank 同步重放，集合通信随图执行。能 work，但依赖 NCCL 算子可被 graph 捕获（见张量并行报告的讨论）。

---

## 四、vLLM 的设计与实现

### 4.1 五种 CUDAGraphMode（`compilation.py:53-94`）

```python
class CUDAGraphMode(enum.Enum):
    NONE = 0                          # 不用 graph
    PIECEWISE = 1                     # 分段捕获（attention 处切开）
    FULL = 2                          # 整图捕获（含 attention）
    FULL_DECODE_ONLY = (FULL, NONE)        # decode 整图，mixed 走 eager
    FULL_AND_PIECEWISE = (FULL, PIECEWISE) # decode 整图，mixed 用分段（v1 默认）
```

- **`decode_mode()` / `mixed_mode()`**：元组型模式对「纯 decode 批」和「含 prefill 的 mixed 批」分别返回子模式。这是 nano-vllm 没有的「按批次类型分派」能力。
- **nano-vllm ≈ `FULL_DECODE_ONLY`**：decode 整图、其余 eager。

### 4.2 PIECEWISE：让 prefill 也能用 graph（核心创新）

整图捕获无法处理 attention 的动态形状。vLLM 的 PIECEWISE 思路：

> 用 torch.compile 把模型 fx graph **在每个 attention 算子处切断**，attention 本身留在 graph 外（eager，处理变长），attention **之间**的稠密算子段（QKV proj、MLP、norm 等，形状只随 token 数变）各自捕获成一个小 CUDA graph。

这样变长 prefill 也能让 80%+ 的算子受益于 graph，只有 attention 走 eager。代价是一次 forward 要 replay 多段图（每层约一段）。`FULL_AND_PIECEWISE`（默认）则更聪明：**decode 批用 FULL（attention 形状也固定，整图最快），prefill/mixed 批用 PIECEWISE**——兼得两者之长。

### 4.3 CUDAGraphWrapper：dispatch 与捕获（`cuda_graph.py:145-320`）

```python
class CUDAGraphWrapper:
    def __call__(self, *args, **kwargs):
        fc = get_forward_context()
        mode = fc.cudagraph_runtime_mode
        key  = fc.batch_descriptor                       # dispatch key
        if mode == NONE or mode != self.runtime_mode:
            return self.runnable(*args, **kwargs)         # 直接跑（不匹配/profile/warmup）
        if key not in self.concrete_cudagraph_entries:
            self.concrete_cudagraph_entries[key] = CUDAGraphEntry(key)
        entry = self.concrete_cudagraph_entries[key]
        if entry.cudagraph is None:                       # 首次见此 key → 捕获
            entry.input_addresses = [x.data_ptr() for x in args if tensor]
            with torch.cuda.graph(cudagraph, pool=self.graph_pool, stream=...):
                output = self.runnable(*args, **kwargs)
            ...
        else:                                             # 已有 → replay
            entry.cudagraph.replay()
```

设计要点（与 nano-vllm 对比）：
- **dispatch key 是 `BatchDescriptor`**（num_tokens + uniformity），比 nano-vllm 的单一 bs 更细，能区分 decode/mixed。
- **运行时模式由 forward_context 下发**，wrapper「盲信」并据此分派——支持嵌套多个不同 mode 的 wrapper（PIECEWISE 各段 + 外层 FULL）。
- **wrapper 不管理持久 buffer**（`cuda_graph.py:161-167` 明言）：不像 nano-vllm 在 wrapper 内拷数据，vLLM 把「写静态 buffer」留给 runner，wrapper 只捕获/replay，保持与 compile 正交。
- **全局 graph pool**（`get_global_graph_pool()`）：所有 wrapper 实例共享一个 memory pool（nano-vllm 是单 runner 内共享）。
- **debug 模式校验输入地址**：`input_addresses` 记录捕获时地址，replay 时检查一致性（`VLLM_LOGGING_LEVEL=DEBUG`），防静默错误——nano-vllm 无此校验。
- **gc 优化**：PIECEWISE 一次 forward 捕获几十段图，捕获期间 patch 掉 `gc.collect` 避免逐段 GC 拖慢。

### 4.4 capture sizes 与 padding（`compilation.py:640+`）

`cudagraph_capture_sizes` 指定要捕获的批大小列表，默认从 `max_num_seqs` 生成（类似 nano-vllm 的 `graph_bs`，但可由用户覆盖、可设 `max_capture_size`）。runner 把实际 batch **pad 到最近的 capture size**——与 nano-vllm `next(x >= bs)` 同思路，但 vLLM 还记录 `num_paddings` 指标用于观测 padding 浪费。

### 4.5 与编译/融合的协同

PIECEWISE 依赖 Inductor 编译，并叠加 PassConfig 里的融合 pass（`compilation.py:120-152`）：RMSNorm+quant、SiluMul+quant、Attention+quant、QK-norm+RoPE、allreduce+RMS 融合、Sequence Parallel、Async TP 等。**CUDA Graph 在 vLLM 里是「编译栈的最后一环」**，而非独立功能；nano-vllm 的 graph 则是脱离编译的纯手写脚本。

---

## 五、横向对比与关键判断

### 5.1 核心机制对照表

| 维度 | nano-vllm | vLLM | 判断 |
|---|---|---|---|
| 捕获范式 | 静态张量+warmup+capture+replay | 同 | ✅ 同源 |
| 捕获模式 | 整图 decode-only | 5 种（含 PIECEWISE） | **vLLM 远胜** |
| prefill 用 graph | ❌ eager | ✅ PIECEWISE | **vLLM 独有** |
| dispatch key | batch size | BatchDescriptor | vLLM 更细 |
| decode/mixed 分派 | ❌ | decode_mode/mixed_mode | vLLM 独有 |
| 静态 buffer | wrapper 内手工 dict | runner 管理、wrapper 解耦 | vLLM 更解耦 |
| padding 分桶 | next(x>=bs) | pad 到 capture size | ✅ 一致 |
| memory pool | 单 runner 共享 | 全局共享 | 思路一致 |
| 从大到小捕获 | ✅ | ✅ | ✅ 一致 |
| 输入地址校验 | ❌ | ✅ debug 模式 | vLLM 更安全 |
| attention 入图 | decode 整图捕获 | FULL 捕获/PIECEWISE 切开 | 思路一致，vLLM 更灵活 |
| torch.compile | ❌ | ✅ 深度集成 | **代差** |
| 算子融合 | ❌ | ✅ 多 pass | **代差** |
| 可观测 | ❌ | CUDAGraphStat 指标表 | vLLM 独有 |

### 5.2 关键判断

**判断一：nano-vllm 的整图 decode 实现是「教科书级正确」。** 静态张量池、warmup 预热、从大到小捕获共享 pool、向上 padding、`fill_(-1)/zero_()` 清残留防污染、以及「把变长信息全编码进 GPU 张量内容而非 shape」从而让 attention 也能入图——这些 CUDA Graph 最容易踩坑的点全部处理正确。作为理解 LLM CUDA Graph 的参考实现，质量很高。

**判断二：最本质的能力缺口是 prefill 无法用 graph。** nano-vllm「prefill 一律 eager」在 prefill-heavy 负载（长 prompt、低生成长度、RAG/总结类）下，CPU launch 开销无法消除。vLLM 的 PIECEWISE 正是为此而生——在 attention 处切开，让变长 prefill 的稠密算子段也享受 graph。这不是优化而是能力维度的差异，是 nano-vllm 最值得演进的方向（但工程量大，依赖 fx graph 切分）。

**判断三：dispatch key 的粒度差异反映设计哲学。** nano-vllm 用 bs 单维，隐含「只有 decode 用 graph，decode 的唯一变量就是 bs」的简化假设——在它的世界里成立。vLLM 的 `BatchDescriptor` 要同时表达「纯 decode vs mixed、token 数」，因为它要支持 prefill+decode 混合批的多模式分派。key 的复杂度直接对应支持的场景数。

**判断四：`CUDAGraphWrapper` 不管 buffer 是个值得学习的解耦。** nano-vllm 把「拷数据进静态张量」写死在 `run_model` 里，graph 逻辑与数据准备耦合。vLLM 让 wrapper 只负责「按 key 捕获/replay」，buffer 管理交给 runner——这使得同一套 wrapper 能包裹 PIECEWISE 各段、FULL 整图、不同模型，复用性强。对 nano-vllm 这种规模虽非必需，但揭示了「捕获机制」与「数据流」分离的工程价值。

**判断五：CUDA Graph 在 vLLM 是编译栈的一环，在 nano-vllm 是独立脚本。** 这是两者最根本的代差——vLLM 的 graph 与 Inductor 融合 pass 协同（融合后的 kernel 更少、graph 更短），而 nano-vllm 捕获的是未融合的原始 kernel 序列。即使同样整图捕获，vLLM 图内 kernel 数也更少、replay 更快。

### 5.3 一个值得注意的潜在风险（nano-vllm）

`block_tables[max_bs, max_num_blocks]` 静态预分配：当 block_size 调小（如前缀缓存报告建议的 16）时，`max_num_blocks = max_model_len / block_size` 会放大 16 倍，`max_bs=512` 下这块静态显存从 ~16MB 级膨胀到上百 MB。**CUDA Graph 的静态 buffer 与 block_size 优化存在张力**，调小 block_size 时需同步关注 graph 显存占用。vLLM 因 capture size 可配置、buffer 由 runner 按需管理，缓解了这一点。

---

## 六、改进建议（按性价比排序）

1. **【中】增加 padding 浪费观测**。仿 vLLM `num_paddings`，在 `run_model` 统计 `bucket_bs - actual_bs` 的累计浪费，便于评估 `graph_bs` 分桶是否合理（当前 16 步长在中等 bs 可能浪费明显）。成本极低。

2. **【中】捕获时记录并在 debug 下校验输入地址**。仿 vLLM 的 `input_addresses` 一致性检查，防止未来重构时静默的「replay 读了错地址」类 bug。低成本、高保险。

3. **【中/按需】graph_bs 分桶可配置**。当前 `[1,2,4,8]+range(16,max,16)` 硬编码；暴露为 config 让用户按实际 bs 分布调整，减少 padding 浪费与捕获显存。

4. **【大/战略】引入 PIECEWISE 让 prefill 用 graph**。这是最大收益但最大工程量——需要在 attention 处切分模型 forward、attention 走 eager、其余段捕获。可不依赖完整 torch.compile，手工在 `Qwen3Model.forward` 层按 decoder layer 边界切分捕获，作为轻量版 PIECEWISE 探索。

5. **【大/战略】关注 block_size 优化与 graph 静态显存的张力**。若采纳前缀缓存报告中「调小 block_size」的建议，需同步评估 `block_tables` 静态 buffer 膨胀，或改为按实际 max 块数动态确定 `max_num_blocks`。

---

## 七、调研局限性与待补充方向

- **未做性能实测**：nano-vllm graph vs eager 的 decode 吞吐提升、padding 浪费占比、不同 bs 分布下的命中率均为代码层推断，未跑 benchmark。建议测「graph on/off」decode tok/s 对比及 launch 开销占比。
- **vLLM 侧深度有限**：PIECEWISE 的 fx graph 切分细节（attention 算子如何被识别为 split point）、`FULL_AND_PIECEWISE` 的运行时 mode 选择逻辑、`CUDAGraphEntry` 的 buffer 与 runner 的具体交互只做了概览，未逐行核对 backend/piecewise_backend.py。
- **未验证 nano-vllm graph 在 TP 下的正确性边界**：NCCL all-reduce 入图后各 rank replay 的同步性、graph_pool 与 NCCL buffer 的显存交互，未专项测试。
- **CUDA Graph 与 chunked prefill 的交互**未深入：chunked prefill 的中间 chunk 是 prefill 形态走 eager，与 decode graph 的切换边界是否有额外开销，留待后续。

---

## 参考来源

- nano-vllm 源码：`nanovllm/engine/model_runner.py`（`capture_cudagraph`/`run_model`/`run`）、`nanovllm/utils/context.py`、`nanovllm/layers/attention.py`（commit `f8d495d`，phase4 分支）
- vLLM 源码（本地 checkout `/home/cb/work/vllm/vllm`）：`vllm/compilation/cuda_graph.py`、`vllm/config/compilation.py`、`vllm/compilation/{backends,piecewise_backend}.py`、`vllm/forward_context.py`
- CUDA Graph 背景：PyTorch `torch.cuda.graph` / CUDA Graphs 官方文档
- 相关文档：`docs/02_core_concepts.md`、`docs/tensor_parallelism-调研报告-20260614.md`、`docs/prefix_caching-调研报告-20260614.md`、`docs/chunked_prefill-调研报告-20260614.md`
