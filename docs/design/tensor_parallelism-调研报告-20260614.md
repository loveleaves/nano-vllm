# Tensor Parallelism 设计与实现对比调研报告：nano-vllm vs 开源 vLLM

**调研日期**：2026-06-14
**调研主体**：当前工程 nano-vllm（phase4 分支）与开源 vLLM 的 Tensor Parallelism（张量并行）设计与实现
**调研意图**（技术向）：拆解当前工程 TP 的进程模型、权重切分、通信原语与并行层设计，与生产级 vLLM 逐层对比，识别简化取舍、正确性边界与可演进方向，服务于工程优化决策。
**代码基准**：
- nano-vllm：`nanovllm/layers/{linear,embed_head}.py`、`nanovllm/engine/{model_runner,llm_engine}.py`、`nanovllm/models/qwen3.py`（commit `f8d495d`）
- vLLM：`vllm/distributed/parallel_state.py`、`vllm/model_executor/layers/{linear,vocab_parallel_embedding}.py`（本地 checkout `/home/cb/work/vllm/vllm`）

---

## 一、执行摘要

1. **算法范式完全同源**：两者都是 Megatron-LM 式张量并行——Attention/MLP 的第一层用 **ColumnParallel**（按输出维切分，无通信），第二层用 **RowParallel**（按输入维切分，forward 末尾 **all-reduce** 求和）；词表用 **VocabParallel**（按 vocab 切分 + mask + all-reduce）。nano-vllm 的 `linear.py` 几乎是 vLLM `linear.py` 的「最小可运行骨架」，连 QKV/gate_up 的合并切分（`shard_id`）设计都一致。

2. **每个 Transformer 层只 2 次 all-reduce**：attention 的 `o_proj`（RowParallel）和 MLP 的 `down_proj`（RowParallel）各一次 all-reduce，QKV/gate_up（ColumnParallel）无通信。这是 Megatron 的经典「每层 2 次通信」结构，两个工程一致。nano-vllm 还在 VocabEmbedding 入口 all-reduce、LMHead 出口用 `gather` 收集 logits。

3. **进程模型差异最大**：nano-vllm 用 **`multiprocessing.spawn` + `SharedMemory(2^20) + Event`** 自制 RPC——rank 0 把 `(method_name, args)` pickle 进共享内存、`event.set()` 唤醒 rank>0，rank>0 阻塞在 `loop()` 里 `read_shm → call`。vLLM 用分层的 **`GroupCoordinator`** 抽象（TP/PP/DP 各一个 group）+ **多种 executor（mp/Ray）** + **ShmMessageQueue broadcaster**，并区分 `device_group`（NCCL）与 `cpu_group`（Gloo）。

4. **通信后端：裸 NCCL vs 多级优选**：nano-vllm 直接调 `dist.all_reduce`（PyTorch 默认 NCCL）。vLLM 在 `device_communicator` 层做**自定义 all-reduce（CUDA IPC，小张量低延迟）→ pynccl → NCCL** 的自动优选，并通过 `torch.ops.vllm.all_reduce` 自定义算子支持 `torch.compile`/CUDA graph 捕获。

5. **功能边界**：nano-vllm 只有纯 TP（单节点、`tcp://localhost:2333` 硬编码端口）；vLLM 还有 **Pipeline Parallel、Data Parallel、Expert Parallel、Sequence Parallel、Async TP**，且 ColumnParallel 支持可选 `gather_output`、RowParallel 支持 `input_is_parallel/reduce_results` 开关与量化（`quant_method.apply`）。

**核心结论**：nano-vllm 的 TP 是一份**正确、完整、可单机多卡跑通的 Megatron 式精简实现**，把「切分 + 2 次 all-reduce + 词表并行」的核心骨架都做对了。它与 vLLM 的差距不在算法，而在**工程基础设施**：进程/通信抽象、通信后端优选、与 CUDA graph/compile 的集成、以及 PP/DP/SP/量化等正交扩展。对单机 ≤8 卡的稠密模型推理足够，但缺乏跨节点扩展与通信优化能力。

---

## 二、背景与概述

### 2.1 张量并行原理（Megatron-LM）

张量并行把单个算子的权重矩阵切到多张 GPU 上并行计算。对 Transformer，标准做法是让 MLP / Attention 的**一对相邻线性层**配合，使得整块只需一次通信：

- **MLP**：`Y = down_proj(act(gate_up_proj(X)))`
  - `gate_up_proj`：**列并行**（按输出维切 A = [A₁, A₂]），各 GPU 算 `XAᵢ`，无需通信。
  - `down_proj`：**行并行**（按输入维切 B = [B₁; B₂]），各 GPU 算 `(XAᵢ)Bᵢ` 后 **all-reduce 求和**。
- **Attention**：`qkv_proj` 列并行（按 head 切分），`o_proj` 行并行（all-reduce）。
- **Embedding/LMHead**：按词表维切分，embedding 后 all-reduce，logits gather。

每个 Transformer 层因此只需 **2 次 all-reduce**（前向）。

### 2.2 两个工程定位

| 维度 | nano-vllm | vLLM |
|---|---|---|
| 并行类型 | 仅 Tensor Parallel | TP + PP + DP + EP + SP + Async TP |
| 进程启动 | `mp.spawn` 子进程 | mp / Ray executor，worker per rank |
| 跨节点 | ❌（localhost 硬编码） | ✅ |
| 通信抽象 | 直接 `torch.distributed` | `GroupCoordinator` 分层抽象 |
| 通信后端 | 裸 NCCL | custom-allreduce / pynccl / NCCL 优选 |
| 量化下的 TP | ❌ | ✅（`quant_method.apply`） |
| 代码量 | `linear.py` 177 行 | `linear.py` 1500+ 行 |

---

## 三、nano-vllm 的设计与实现

### 3.1 进程模型与自制 RPC（`llm_engine.py:30-41`、`model_runner.py:38-116`）

```
LLMEngine.__init__:
  ctx = mp.get_context("spawn")
  for i in 1..tp_size-1:
      event = ctx.Event()
      ctx.Process(target=ModelRunner, args=(config, i, event)).start()   # 子进程
  self.model_runner = ModelRunner(config, 0, events)                     # 主进程 rank 0
```

- **rank 0（主进程）**：跑调度器 + 采样，是唯一对外接口。
- **rank 1..N（子进程）**：`ModelRunner.__init__` 末尾进入 `loop()` 永久阻塞，等 rank 0 指令。

**自制 RPC（共享内存 + Event）**：
```python
# rank 0 下发指令
def call(method, *args):
    if world_size > 1 and rank == 0:
        write_shm(method, *args)         # pickle 写入 SharedMemory，event.set() 唤醒
    return getattr(self, method)(*args)  # rank 0 自己也执行

# rank>0 收指令
def loop():
    while True:
        method, args = read_shm()        # event.wait() → 读 4 字节长度 + pickle body
        call(method, *args)
        if method == "exit": break
```

每个推理 step，`llm_engine.step` 调 `model_runner.call("run", seqs, is_prefill)`，rank 0 把 seqs 广播给所有 rank，各 rank 并行执行 `run()`，最终只有 rank 0 做 sampling（`run()` 里 `if self.rank == 0`）。

### 3.2 NCCL 初始化（`model_runner.py:47-49`）

```python
dist.init_process_group("nccl", "tcp://localhost:2333",
                        world_size=tp_size, rank=rank)
torch.cuda.set_device(rank)
```

固定 `localhost:2333`、NCCL 后端、rank == GPU device id。**单节点假设硬编码**。

### 3.3 列并行 `ColumnParallelLinear`（`linear.py:70-90`）

```python
def __init__(self, input_size, output_size, bias):
    super().__init__(input_size, divide(output_size, tp_size), bias, tp_dim=0)
def weight_loader(self, param, loaded_weight, *args):
    shard_size = param.data.size(0)                    # 本地切片大小
    start = self.tp_rank * shard_size
    param.data.copy_(loaded_weight.narrow(0, start, shard_size))   # 加载时切片
def forward(self, x):
    return F.linear(x, self.weight, self.bias)         # 无通信
```

- **加载即切分**：`weight_loader` 从完整权重 narrow 出本 rank 的行块，权重在显存中只存本地分片。
- **forward 无通信**：各 rank 算各自的输出列块。

### 3.4 合并列并行：QKV / gate_up（`linear.py:93-146`）

为减少 kernel 启动，Q/K/V 合并成单个 `qkv_proj`、gate/up 合并成 `gate_up_proj`。`weight_loader` 接 `shard_id` 把各子矩阵写进合并参数的正确偏移：

- **`MergedColumnParallelLinear`**：`shard_id` 是 int，`shard_offset = sum(output_sizes[:shard_id]) // tp_size`。
- **`QKVParallelLinear`**：`shard_id` 是 `"q"/"k"/"v"`，按 GQA 的 `num_heads/num_kv_heads`（已 `divide(.., tp_size)`）计算偏移，正确处理 KV head 少于 Q head 的情形。

### 3.5 行并行 `RowParallelLinear`（`linear.py:149-176`）

```python
def __init__(self, input_size, output_size, bias):
    super().__init__(divide(input_size, tp_size), output_size, bias, tp_dim=1)  # 按输入维切
def forward(self, x):
    y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)  # bias 只 rank0 加
    if self.tp_size > 1 and dist.is_initialized():
        dist.all_reduce(y)        # ⭐ 求和得完整输出
    return y
```

- **bias 只在 rank 0 加**：bias 是全局值不切分，否则 all-reduce 会重复累加 N 次。
- **forward 末尾 all-reduce**：这是每层 2 次通信的来源之一。

### 3.6 词表并行（`embed_head.py`）

**`VocabParallelEmbedding`**（入口）：
```python
mask = (x >= vocab_start) & (x < vocab_end)     # 本 rank 负责的 token 范围
x_local = mask * (x - vocab_start)
y = F.embedding(x_local, self.weight)
y = mask.unsqueeze(1) * y                         # 范围外的 token 置 0
dist.all_reduce(y)                                # ⭐ 汇总（每个 token 只有一个 rank 非零）
```

**`ParallelLMHead`**（出口）：
```python
logits = F.linear(x, self.weight)                # 各 rank 算本地 vocab 分片的 logits
all_logits = [empty for _ in tp_size] if rank==0 else None
dist.gather(logits, all_logits, 0)               # ⭐ gather 到 rank 0（非 all-gather）
logits = torch.cat(all_logits, -1) if rank==0 else None
```

注意 LMHead 用 **`gather`（仅 rank 0 收）** 而非 all-gather——因为只有 rank 0 做 sampling，省一半通信量。这是个合理的小优化。

### 3.7 Attention 的 head 切分（`qwen3.py:47-63`）

```python
_, tp_size = _get_tp_info()
self.num_heads = divide(num_heads, tp_size)        # 每 rank 的 head 数
self.num_kv_heads = divide(num_kv_heads, tp_size)
self.attn = Attention(self.num_heads, ..., self.num_kv_heads)
```

KV cache 也据此按本地 head 数分配（`model_runner.py:139`：`num_kv_heads = hf_config.num_key_value_heads // world_size`），每张卡只存自己 head 的 KV。

### 3.8 与 CUDA Graph 的关系

decode 阶段每 rank 各自 replay 自己的 CUDA graph（`run_model:242-257`），graph 内部包含 RowParallel 的 `dist.all_reduce`。由于 nano-vllm 用裸 NCCL all-reduce，**all-reduce 被直接捕获进 graph**——能 work 但依赖 NCCL 算子可被 graph 捕获这一隐含前提（vLLM 为此专门做了自定义算子封装，见下）。

---

## 四、vLLM 的设计与实现

### 4.1 分层并行抽象：`GroupCoordinator`（`parallel_state.py:290`）

vLLM 不直接用 `torch.distributed`，而是为每种并行维度建一个 `GroupCoordinator`：

```
全局进程网格被切成多个正交 group：
  get_tp_group()  — 张量并行
  get_pp_group()  — 流水线并行
  get_dp_group()  — 数据并行
  get_ep_group()  — 专家并行
```

每个 `GroupCoordinator` 持有：
- **`device_group`**（NCCL）：GPU 张量通信。
- **`cpu_group`**（Gloo）：CPU 侧协调（如元数据广播）。
- **`device_communicator`**：实际通信实现（含自定义 all-reduce）。
- **`mq_broadcaster`**：共享内存消息队列（广播控制信息，类似 nano-vllm 的 SharedMemory，但更通用）。
- **`rank_in_group` vs `rank`**：区分组内 rank 与全局 rank，支持多节点多组拓扑（`parallel_state.py:304-312` 的多节点 rank 映射表）。

nano-vllm 没有这层抽象——它的「group」就是全局唯一的 NCCL world，rank == GPU id。

### 4.2 通信原语的自定义算子封装（`parallel_state.py:533-560`）

```python
def all_reduce(self, input_):
    if self.world_size == 1: return input_
    if self.use_custom_op_call:
        return torch.ops.vllm.all_reduce(input_, group_name=self.unique_name)  # 自定义算子
    return self._all_reduce_out_place(input_)   # → device_communicator.all_reduce
```

关键设计：把 all-reduce 包成 **`torch.ops.vllm.all_reduce` 自定义算子**（传 `group_name` 字符串而非 `self` 对象），原因注释写得很清楚：
1. **Dynamo/torch.compile 不能传任意对象**，只能传字符串 group_name 再查表。
2. **PyTorch 自定义算子不支持原地修改/同算子返回新张量**，所以 all-reduce 强制 out-of-place。

这让 all-reduce 能被 `torch.compile` 和 CUDA graph 安全捕获。nano-vllm 直接调 `dist.all_reduce`（原地），在 eager 下没问题，但不具备 compile 集成能力。

### 4.3 `device_communicator` 的后端优选

`_all_reduce_out_place → device_communicator.all_reduce`（`CudaCommunicator`）内部按张量大小/拓扑优选：
- **Custom all-reduce**（CUDA IPC P2P）：小张量低延迟，绕过 NCCL。
- **pynccl**：Python 直接调 NCCL，避开 PyTorch ProcessGroup 在 graph 捕获下的限制。
- **NCCL**（PyTorch 默认）：兜底。

这是 nano-vllm 完全没有的一层——后者所有 all-reduce 都走 PyTorch 默认 NCCL，小张量（decode 时每 token 的 hidden_state all-reduce）延迟无法优化。

### 4.4 ColumnParallelLinear / RowParallelLinear（`linear.py`）

结构与 nano-vllm 同源，但有更多开关与生产特性：

**RowParallelLinear.forward（`linear.py:1537-1563`）**：
```python
if self.input_is_parallel:
    input_parallel = input_                       # 上游已是分片（attn→o_proj）
else:
    input_parallel = split_tensor_along_last_dim(input_)[tp_rank]  # 否则现场切
bias_ = None if (tp_rank > 0 or skip_bias_add) else self.bias      # 同 nano-vllm：仅 rank0 加 bias
output_parallel = self.quant_method.apply(self, input_parallel, bias_)  # ⭐ 量化感知
if self.reduce_results and self.tp_size > 1:
    output = tensor_model_parallel_all_reduce(output_parallel)    # all-reduce
```

比 nano-vllm 多三点：
1. **`input_is_parallel`**：上游若已输出分片（如 attention 输出），跳过现场切分，省一次 split。
2. **`reduce_results`**：可关 all-reduce（用于 sequence parallel 等把 all-reduce 替换成 reduce-scatter 的场景）。
3. **`quant_method.apply`**：matmul 走量化方法分派，TP 与 INT8/FP8/GPTQ/AWQ 等正交组合。

**ColumnParallelLinear** 还支持可选 `gather_output`（all-gather 把分片输出拼回完整，用于不接 RowParallel 的孤立列并行层）。nano-vllm 的 ColumnParallel 永远不 gather（总是接 RowParallel）。

### 4.5 VocabParallelEmbedding（`vocab_parallel_embedding.py:192`）

核心 mask + all-reduce 与 nano-vllm 一致，但多了生产细节：
- **`VocabParallelEmbeddingShardIndices`**：精确管理「真实 vocab」与「padding」的分片边界——vLLM 把 vocab **pad 到 tp_size 的整数倍**（甚至 pad 到 64 对齐以利 kernel），nano-vllm 则直接 `assert num_embeddings % tp_size == 0`（要求模型 vocab 天然可整除，否则报错）。
- 支持 LoRA 额外词表、量化 embedding 等。

### 4.6 正交扩展（nano-vllm 全缺）

- **Pipeline Parallel**：层间切分 + send/recv，跨节点扩展。
- **Data Parallel / Expert Parallel**：MoE 模型的专家分布。
- **Sequence Parallel**：把 LayerNorm/Dropout 的激活也按序列维切分，all-reduce → reduce-scatter + all-gather，省激活显存。
- **Async TP**：通信与计算 overlap。
- **多节点**：通过 Ray 或 torchrun 跨机，`GroupCoordinator` 的 rank 映射表原生支持。

---

## 五、横向对比与关键判断

### 5.1 核心机制对照表

| 维度 | nano-vllm | vLLM | 判断 |
|---|---|---|---|
| TP 算法 | Megatron 列/行并行 | 同 | ✅ 同源 |
| 每层 all-reduce 次数 | 2（o_proj+down_proj） | 2 | ✅ 一致 |
| ColumnParallel 通信 | 无 | 无（可选 gather_output） | vLLM 更灵活 |
| RowParallel 通信 | 末尾 all-reduce | all-reduce（可关） | 同思路 |
| bias 处理 | 仅 rank0 加 | 仅 rank0 加 | ✅ 一致 |
| QKV/gate_up 合并切分 | shard_id 偏移 | 同 | ✅ 一致 |
| 词表并行 | mask+all_reduce | 同 + padding 对齐 | vLLM 更鲁棒 |
| LMHead logits | `gather` 到 rank0 | gather/all-gather | nano 省通信 |
| 进程模型 | mp.spawn + SharedMem RPC | GroupCoordinator + executor | **vLLM 远胜** |
| 通信抽象 | 全局单 NCCL world | TP/PP/DP/EP 分组 | **vLLM 远胜** |
| 通信后端 | 裸 NCCL | custom-AR/pynccl/NCCL 优选 | **vLLM 远胜** |
| compile/graph 集成 | 直接捕获 dist.all_reduce | 自定义算子封装 | vLLM 更稳健 |
| 量化 × TP | ❌ | ✅ quant_method | vLLM 独有 |
| PP/DP/SP/EP | ❌ | ✅ | vLLM 独有 |
| 跨节点 | ❌ localhost 硬编码 | ✅ | vLLM 独有 |

### 5.2 关键判断

**判断一：算法层面 nano-vllm 是「教科书级正确实现」。** 列/行并行的配对、每层 2 次 all-reduce、bias 仅 rank 0 加、QKV GQA 切分、词表 mask+all-reduce、LMHead 用 gather 省通信——这些最容易写错的点全部正确，且 KV cache 按本地 head 分配、与 CUDA graph 协同。作为理解 Megatron TP 的参考实现质量很高。

**判断二：差距集中在「进程与通信基础设施」。** nano-vllm 的自制 SharedMemory RPC 思路其实和 vLLM 的 `mq_broadcaster` 一脉相承（都是共享内存广播控制信息），但 vLLM 把它做成了通用的 `GroupCoordinator` 抽象，能同时支撑 TP/PP/DP 多组、多节点、CPU/GPU 双 group。nano-vllm 的「全局单 NCCL world + rank==GPU id + localhost:2333」假设把它锁死在**单机**。

**判断三：通信后端是真实性能差距。** decode 阶段每个 token 都要在 o_proj/down_proj 做 2 次小张量 all-reduce（hidden_size 维），这种**高频小消息**正是 vLLM custom all-reduce（CUDA IPC）的优化目标。nano-vllm 走裸 NCCL，小张量 all-reduce 的固定延迟无法摊薄，TP 下 decode 吞吐会明显受限。这是最值得借鉴的优化方向。

**判断四：`torch.ops.vllm.all_reduce` 自定义算子封装是个易被忽视但重要的设计。** nano-vllm 把 `dist.all_reduce` 直接捕获进 CUDA graph 目前能 work，但这依赖具体 NCCL/PyTorch 版本对 graph 捕获原地集合通信的支持；vLLM 用 out-of-place 自定义算子规避了这个脆弱点，也才能接 `torch.compile`。若 nano-vllm 未来引入 compile，这里需要重构。

**判断五：量化 × TP 的缺失是能力边界。** nano-vllm 的 `F.linear` 直接吃 fp16/bf16 权重，没有 `quant_method` 分派层，意味着**无法在 TP 下跑量化模型**。生产部署里 TP + 量化是常态组合，这是功能性缺口而非优化项。

---

## 六、改进建议（按性价比排序）

1. **【高性价比】引入小张量 all-reduce 优化**。decode 的高频小 all-reduce 是 TP 吞吐瓶颈，可移植 vLLM 的 custom all-reduce（CUDA IPC）或至少用 pynccl，对 TP≥2 的 decode 吞吐提升直接。

2. **【中】解除单机硬编码**。把 `tcp://localhost:2333` 改为可配置 `init_method`、rank/local_rank 解耦（当前 `set_device(rank)` 假设 rank==local GPU id），为多节点留口子。改动小但解锁扩展性。

3. **【中】vocab padding 对齐**。当前 `assert num_embeddings % tp_size == 0` 会让 vocab 不可整除的模型直接报错；按 vLLM 思路 pad 到 tp_size（或 64）倍数可兼容更多模型。

4. **【中/按需】all-reduce 自定义算子封装**。若计划引入 `torch.compile`，需把 `dist.all_reduce` 改成 out-of-place 自定义算子；否则当前 eager + 直接 graph 捕获可暂不动。

5. **【大/战略】量化 × TP**。若要支持量化模型部署，需在 LinearBase 引入 `quant_method.apply` 分派层，工作量大但是生产刚需。

6. **【大/战略】Pipeline Parallel**。突破单机显存上限、跑超大模型的前提，但需要 `GroupCoordinator` 式分组抽象先落地，是较远期目标。

---

## 七、调研局限性与待补充方向

- **未做性能实测**：裸 NCCL vs custom all-reduce 的小张量延迟差距、SharedMemory RPC 的广播开销均为代码层推断，未在 nano-vllm 上跑 TP benchmark 量化。建议测 TP=2/4 下 decode 吞吐随 tp_size 的 scaling 曲线，定位 all-reduce 占比。
- **vLLM 侧深度有限**：`device_communicator` 的 custom all-reduce 具体实现（CUDA IPC buffer 管理、何时切换 NCCL）、Sequence Parallel 的 reduce-scatter 改写、Async TP 的 overlap 调度均只做了概览，未逐行核对。
- **未覆盖 nano-vllm TP 的正确性边界**：多 rank 下采样随机性是否同步（rank 0 独采应无问题，但 dropout/随机种子未确认）、preempt/抢占在 TP 下各 rank block_table 是否一致，未专项验证。
- **PP/DP/EP** 在本次对比中仅列举，未深入 vLLM 的具体实现，与 TP 的交互（如 TP×PP 的 2D 网格）留待后续。

---

## 参考来源

- nano-vllm 源码：`nanovllm/layers/linear.py`、`nanovllm/layers/embed_head.py`、`nanovllm/engine/model_runner.py`、`nanovllm/engine/llm_engine.py`、`nanovllm/models/qwen3.py`、`nanovllm/config.py`（commit `f8d495d`，phase4 分支）
- vLLM 源码（本地 checkout `/home/cb/work/vllm/vllm`）：`vllm/distributed/parallel_state.py`、`vllm/model_executor/layers/linear.py`、`vllm/model_executor/layers/vocab_parallel_embedding.py`、`vllm/distributed/communication_op.py`
- 算法背景：Megatron-LM 张量并行（Shoeybi et al., 2019）
- 相关文档：`docs/02_core_concepts.md`、`docs/prefix_caching-调研报告-20260614.md`、`docs/chunked_prefill-调研报告-20260614.md`
