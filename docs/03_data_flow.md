# 数据流与调度策略

## 一、完整 Prefill 数据流

以两个请求为例（无前缀缓存命中），完整追踪从调度到采样的每一步数据变换。

```
SeqA: prompt = [1, 2, 3, ..., 100]    # 100 个 token
SeqB: prompt = [5, 6, 7, ..., 84]     #  80 个 token
block_size = 256
```

### Step 1：Scheduler.schedule()

```python
# SeqA 首次调度
num_cached_blocks_A = can_allocate(seqA)  # = 0（无缓存）
block_manager.allocate(seqA, 0)
#   SeqA 需 1 块（ceil(100/256) = 1）
#   seqA.block_table = [3]   ← 分配物理块 3
#   seqA.num_cached_tokens = 0
seqA.num_scheduled_tokens = 100   # min(100, remaining=4096) = 100

# SeqB 首次调度
num_cached_blocks_B = can_allocate(seqB)  # = 0（无缓存）
block_manager.allocate(seqB, 0)
#   seqB.block_table = [7]   ← 分配物理块 7
seqB.num_scheduled_tokens = 80

# 本步处理完整个 prompt → 两者都移入 running
seqA.status = RUNNING;  seqB.status = RUNNING
return ([seqA, seqB], is_prefill=True)
```

### Step 2：ModelRunner.prepare_prefill()

```python
# 拼接所有 seq 的待处理 token
input_ids = [1,2,...,100, 5,6,...,84]    # shape [180]
positions  = [0,1,...,99, 0,1,...,79]    # 各 seq 的绝对位置，从 num_cached_tokens 开始

# cu_seqlens：描述拼接后 input_ids 中各 seq 的边界
cu_seqlens_q = [0, 100, 180]   # SeqA: input_ids[0:100]，SeqB: input_ids[100:180]
cu_seqlens_k = [0, 100, 180]   # 无前缀缓存，K 总长度 = Q 长度

# slot_mapping：每个 token 写入 KV cache 的物理 slot 位置
# SeqA 的 100 个 token → block_table[0]=3 → slot 3*256+0 到 3*256+99
slot_mapping = [768, 769, ..., 867,    # SeqA 100 个 slot（3*256=768 起）
                1792, 1793, ..., 1871] # SeqB 80 个 slot（7*256=1792 起）

# 无前缀缓存，block_tables = None
set_context(True, cu_seqlens_q, cu_seqlens_k, 100, 100, slot_mapping, None, None)
```

### Step 3：model.forward(input_ids=[180], positions=[180])

```
── embed_tokens ──
x = embed_tokens([1,2,...,100,5,...,84])   # [180, 2048]

── 第 0 层 Transformer（以下 28 层相同）──

x, residual = rms_norm(x)          # [180, 2048]
   ↑ 首层 residual=None，普通 rms_norm

qkv = qkv_proj(x)                  # [180, (16+8+8)*128/1] = [180, 4096]（tp=1）
q = qkv[:, :16*128].view(180, 16, 128)    # [180, 16, 128]  Q
k = qkv[:, 16*128:24*128].view(180, 8, 128)  # [180, 8, 128] K（GQA）
v = qkv[:, 24*128:].view(180, 8, 128)     # [180, 8, 128] V

q = q_norm(q)    # RMSNorm per head：[180, 16, 128]
k = k_norm(k)    # [180, 8, 128]

q, k = rotary_emb(positions, q, k)   # 应用 RoPE，形状不变

# ① 写 KV cache（Triton kernel）
store_kvcache(k, v, k_cache, v_cache, slot_mapping)
# k_cache[3, 0:100, :, :] ← SeqA 的 100 个 K
# k_cache[7, 0:80, :, :]  ← SeqB 的 80 个 K
# （slot_mapping 直接映射到展平后的位置）

# ② FlashAttention（prefill，可变长，因果 mask）
o = flash_attn_varlen_func(
    q, k, v,
    cu_seqlens_q=[0,100,180], cu_seqlens_k=[0,100,180],
    max_seqlen_q=100, max_seqlen_k=100,
    causal=True, block_table=None
)   # [180, 16, 128]

x = o_proj(o.view(180, -1))   # [180, 2048]，行并行 + all_reduce（tp>1时）

x, residual = post_norm(x, residual)  # Fused Add-RMSNorm

gate_up = gate_up_proj(x)    # [180, 2*11008/tp] = [180, 22016]（tp=1）
half = 11008
x = silu(gate_up[:, :half]) * gate_up[:, half:]   # SwiGLU，[180, 11008]
x = down_proj(x)             # [180, 2048] + all_reduce

── 最终归一化 ──
x, _ = final_norm(x, residual)   # [180, 2048]
return x
```

### Step 4：compute_logits()

```python
# 只取每个 seq 最后一个 token 的 hidden state
last_indices = cu_seqlens_q[1:] - 1   # [99, 179]
x = hidden_states[last_indices]        # [2, 2048]（两个 seq 各取最后一个）

# LM Head（并行矩阵乘）
logits_partial = x @ lm_head.weight.T   # [2, 151936/tp]（tp=1时 [2, 151936]）
# tp>1时：rank 0 gather → [2, 151936]
```

### Step 5：Sampler

```python
temperatures = [0.6, 0.6]
logits = logits / temperatures.unsqueeze(1)   # 温度缩放：[2, 151936]
probs = softmax(logits, dim=-1)               # [2, 151936]
noise = Exponential(1).clamp_min(1e-10)       # [2, 151936]
token_ids = (probs / noise).argmax(dim=-1)    # [2]  → 如 [42, 1337]
```

### Step 6：Scheduler.postprocess()

```python
for seq, token_id in zip([seqA, seqB], [42, 1337]):
    block_manager.hash_blocks(seq)
    # seqA：start=0//256=0, end=(0+100)//256=0 → 不满一块，无哈希注册
    # seqB：同上

    seq.num_cached_tokens += seq.num_scheduled_tokens
    # seqA: 0 + 100 = 100
    # seqB: 0 + 80 = 80

    seq.append_token(token_id)
    # seqA.token_ids = [1,...,100, 42]
    # seqB.token_ids = [5,...,84, 1337]

    if token_id == eos or completion_count == max_tokens:
        → FINISHED + deallocate
    else:
        → 留在 running，下步进入 decode
```

---

## 二、Decode 阶段数据流

```
状态：seqA(101 token), seqB(81 token) 均在 running
```

### Step 1：Scheduler.schedule()（decode 路径）

```python
# 无 waiting seq，进入 decode 调度
while running:
    seq = running.popleft()   # seqA
    can_append(seqA):
        len(seqA) % 256 = 101 % 256 = 101 ≠ 1
        → 不需要新块，返回 True（free_block_ids >= 0）
    may_append(seqA):
        101 % 256 != 1，不分配新块
    seqA.num_scheduled_tokens = 1

    seq = running.popleft()   # seqB
    len(seqB) % 256 = 81 % 256 = 81 ≠ 1 → 同上

running.extendleft(reversed([seqA, seqB]))   # 放回队列
return ([seqA, seqB], is_prefill=False)
```

### Step 2：ModelRunner.prepare_decode()

```python
input_ids  = [42, 1337]           # 各 seq 的最新 token，[2]
positions  = [100, 80]            # len(seq) - 1，[2]
context_lens = [101, 81]          # len(seq)，KV 总长度，[2]

# slot_mapping：新 token 写入的位置
# seqA: block_table[-1]=3，offset = len(seqA) % block_size - 1 = 101%256-1 = 100
slot_mapping = [
    3 * 256 + 100,  # = 868   seqA 的 token 42 写入 slot 868
    7 * 256 + 80,   # = 1872  seqB 的 token 1337 写入 slot 1872
]

# block_tables：[2, max_blocks]，-1 填充
block_tables = [[3, -1, -1, ...],   # seqA 只有 1 块
                [7, -1, -1, ...]]   # seqB 只有 1 块

set_context(False, slot_mapping, context_lens, block_tables)
```

### Step 3：model.forward(input_ids=[2], positions=[2])

```
x = embed_tokens([42, 1337])   # [2, 2048]

for layer in layers:
    qkv = qkv_proj(x)          # [2, 4096]
    q = qkv[:, :2048].view(2, 16, 128)
    k = qkv[:, 2048:3072].view(2, 8, 128)
    v = qkv[:, 3072:].view(2, 8, 128)

    # ① 写 KV cache（只写当前步的 K/V）
    store_kvcache(k, v, k_cache, v_cache, slot_mapping)
    # k_cache[3, 100, :, :] ← seqA 当前 K
    # k_cache[7, 80, :, :]  ← seqB 当前 K

    # ② FlashAttention（decode，从 kv_cache 读历史）
    o = flash_attn_with_kvcache(
        q.unsqueeze(1),    # [2, 1, 16, 128]（每 seq 1 个 query）
        k_cache,           # [num_blocks, 256, 8, 128]
        v_cache,
        cache_seqlens=[101, 81],    # attend to 101 和 81 个历史 K
        block_table=[[3,-1,...], [7,-1,...]],   # 分页地址
        causal=True
    )   # [2, 1, 16, 128]

    x = o_proj(o.view(2, -1))   # [2, 2048]
    # ... ffn ...

# compute_logits：decode 时不取 last，直接 [2, vocab]
logits = lm_head(x)
```

---

## 三、CUDA Graph Decode 数据流（decode + graph replay）

CUDA Graph 只在 decode 阶段且 `bs <= 512` 时使用。以 `bs=3`，`graph_bs=4` 为例：

### Capture 时（bs=4，静态）

```python
# 录制时的静态张量（地址固定，存入 CUDA graph）
input_ids_static   = [0, 0, 0, 0]        # 全零占位
positions_static   = [0, 0, 0, 0]
slot_mapping_static = [-1, -1, -1, -1]   # -1 = 跳过写入
context_lens_static = [0,  0,  0,  0]
block_tables_static = [[0,...], ...]      # 全零占位

# 录制 model forward（以上静态张量为输入）
with torch.cuda.graph(graph, pool):
    outputs_static = model(input_ids_static, positions_static)
# graph 内硬编码了 input_ids_static, slot_mapping_static 等的 GPU 地址
```

### Replay 时（实际 bs=3）

```python
bs = 3
graph = graphs[4]   # 找 >= 3 的最小 graph，= 4
gv = graph_vars

# 写入实际数据（修改静态张量的值，不改变地址）
gv["input_ids"][:3] = [42, 1337, 99]     # 前 3 个有实际 token
gv["input_ids"][3]  = 0                  # 第 4 个保持 0（dummy）

gv["slot_mapping"].fill_(-1)             # 全部重置为 -1（dummy 跳过写入）
gv["slot_mapping"][:3] = [868, 1872, 500]  # 前 3 个填真实 slot

gv["context_lens"].zero_()
gv["context_lens"][:3] = [101, 81, 50]  # 前 3 个有效

gv["block_tables"][:3] = [[3,-1,...], [7,-1,...], [5,2,...]]

graph.replay()    # 一次 GPU 调用，执行全部 500+ kernel

# dummy 第 4 个 seq 的处理结果：
#   slot=-1 → store_kvcache 跳过（不写 kv_cache）
#   context_lens=0 → FlashAttention 输出 0 或 NaN（会被截掉）

token_ids = compute_logits(gv["outputs"][:3])  # 只取前 3 个
```

### CUDA Graph 的 Context 交互

CUDA Graph 录制的是模型执行时的 GPU 指令，包括 Attention 层调用 `get_context()` 获取 `slot_mapping` 等。这些静态张量的 **GPU 地址** 被硬编码进 graph，replay 时 GPU 直接从同一地址读取数据——因此修改静态张量的"内容"等效于修改 graph 的"输入"。

---

## 四、前缀缓存 Prefill 数据流

情形：SeqA 和 SeqB 共享前 256 个 token（一个满块）。

```
SeqA 先完成，释放 Block 2（hash=0xABCD1234）
  hash_to_block_id[0xABCD1234] = 2

SeqC 到来，token_ids[0:256] = SeqA.token_ids[0:256]（完全相同）
```

### can_allocate(seqC)

```python
h = compute_hash(seqC.block(0), prefix=-1)   # = 0xABCD1234
block_id = hash_to_block_id.get(0xABCD1234)  # = 2（命中！）
blocks[2].token_ids == seqC.block(0)?  → True（碰撞校验通过）

num_cached_blocks = 1
num_new_blocks = seqC.num_blocks - 1    # 除缓存块外需新分配
return 1
```

### allocate(seqC, 1)

```python
# Block 2 在 free_block_ids（ref_count=0），重激活
blocks[2].ref_count = 1
free_block_ids.remove(2)
used_block_ids.add(2)
seqC.block_table = [2, new_block_id]

seqC.num_cached_tokens = 1 * 256 = 256
```

### prepare_prefill（seqC）

```python
start = seqC.num_cached_tokens = 256   # 从第 256 个 token 开始
input_ids = seqC.token_ids[256:]       # 只处理 256 之后的 token
positions  = range(256, len(seqC))

cu_seqlens_q = [0, len(seqC)-256]     # query 数 = 实际需处理 token
cu_seqlens_k = [0, len(seqC)]         # KV 总长 = 历史（256）+ 当前处理 token

# slot_mapping：只计算新写入的部分（num_cached_tokens 之后）
slot_mapping = ...   # 从 seqC.block_table[1] 开始

# cu_seqlens_k > cu_seqlens_q → 有前缀缓存 → 需要 block_tables
block_tables = [[2, new_block_id]]
set_context(True, cu_seqlens_q, cu_seqlens_k, ..., block_tables=block_tables)
```

### FlashAttention（前缀缓存路径）

```python
# attention.py
if context.block_tables is not None:
    k, v = k_cache, v_cache    # ← 使用 KV cache（含 Block 2 的历史 KV）
o = flash_attn_varlen_func(
    q,           # 只有本步的 query（len(seqC)-256 个）
    k, v,        # 是整个 kv_cache 张量（FlashAttention 按 block_table 读取）
    cu_seqlens_q=[0, len(seqC)-256],
    cu_seqlens_k=[0, len(seqC)],     # 告诉 FA：K 有 len(seqC) 个（含历史）
    block_table=block_tables,         # FA 按此映射读取 Block 2 的历史 KV
    causal=True
)
```

**效果**：SeqC 的前 256 token 完全跳过了 forward 计算（embedding、QKV 投影、RoPE、Attention 都没算），直接复用 SeqA 的 KV 结果。

---

## 五、Chunked Prefill 数据流

SeqA（2000 token prompt），max_num_batched_tokens=1024，running=[SeqB, SeqC]（decode 阶段）：

### Step 1（SeqA 第一个 chunk）

```
schedule() 返回 ([SeqA], is_prefill=True)
SeqA.num_scheduled_tokens = 1024
SeqA.num_cached_tokens = 0 → 处理后 = 1024
SeqA 留在 waiting（1024 < 2000）
```

> **SeqB 和 SeqC 不被调度**：prefill 和 decode 严格分步，waiting 非空时不做 decode。

### Step 2（SeqA 第二个 chunk）

```
schedule()：waiting 有 SeqA（分块续传）
  remaining = 1024
  num_tokens = 2000 - 1024 = 976（剩余）
  SeqA.num_scheduled_tokens = min(976, 1024) = 976

  976 + 1024 = 2000 == num_tokens → SeqA 移入 running，状态 RUNNING

return ([SeqA], is_prefill=True)
```

### Step 3（SeqA 完成 prefill，进入 decode）

```
waiting 为空 → 进入 decode
schedule() 返回 ([SeqA, SeqB, SeqC], is_prefill=False)
```

**注意 Step 2 中 postprocess 的特殊处理：**

```python
# scheduler.py: postprocess()
for seq, token_id in zip(seqs, token_ids):
    block_manager.hash_blocks(seq)
    seq.num_cached_tokens += seq.num_scheduled_tokens

    # 关键：chunked prefill 中间步不产出新 token
    if is_prefill and seq.num_cached_tokens < seq.num_tokens:
        continue   # ← 跳过 append_token

    seq.append_token(token_id)  # 只有最后一步才追加
```

**为什么中间步不产出 token？**
模型 compute_logits 在 prefill 时只取最后一个 token 的 hidden state，chunk 1 的"最后一个 token"是 token[1023]，这不是 prompt 最后一个 token，用它采样的 token 是无效的。

---

## 六、Scheduler 调度策略详解

### 调度优先级与吞吐权衡

```
优先 prefill 的理由：
  1. prefill 计算密度高（矩阵乘，GPU 利用率高）
  2. decode 中途插 prefill 会破坏 CUDA Graph（batch_size 变化）
  3. 简单优先规则避免饥饿（worst case：waiting 无限增长）

优先 prefill 的代价：
  1. decode 延迟（TPOT，Time Per Output Token）抖动
  2. 与 chunked prefill 结合后，prefill 步不做 decode
```

### Prefill 调度流程（完整）

```
while waiting and len(scheduled) < max_num_seqs:
    seq = waiting[0]   # 不 pop，先探测
    remaining = max_num_batched_tokens - num_batched_tokens
    if remaining == 0: break

    if seq.block_table == []:
        # 首次调度：探测前缀缓存
        num_cached = can_allocate(seq)   # -1 表示内存不足
        if num_cached == -1: break
        num_tokens = seq.num_tokens - num_cached * block_size
    else:
        # 分块续传：继续上次未完成的部分
        num_tokens = seq.num_tokens - seq.num_cached_tokens

    # 关键约束：只允许第一个 seq 分块
    if remaining < num_tokens and scheduled:
        break   # 第 2+ 个 seq 必须能完整 prefill 或等下一步

    if seq.block_table == []:
        block_manager.allocate(seq, num_cached)   # 正式分配

    seq.num_scheduled_tokens = min(num_tokens, remaining)
    num_batched_tokens += seq.num_scheduled_tokens

    if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
        seq.status = RUNNING
        waiting.popleft()
        running.append(seq)

    scheduled.append(seq)
```

### Decode 调度流程（完整）

```
while running and len(scheduled) < max_num_seqs:
    seq = running.popleft()

    # 内层循环：处理内存不足
    while not block_manager.can_append(seq):
        if running:
            preempt(running.pop())   # 抢占最晚加入 running 的 seq（优先级最低）
        else:
            preempt(seq)             # 连当前 seq 都装不下，seq 自己也被抢占
            seq = None               # 标记 seq 已被抢占
            break
    else:
        # can_append 成功
        seq.num_scheduled_tokens = 1
        block_manager.may_append(seq)   # 按需分配新块（仅在块边界时）
        scheduled.append(seq)

running.extendleft(reversed(scheduled))   # 保持 FIFO 顺序放回
```

### can_append 的精确逻辑

```python
def can_append(seq) -> bool:
    # 问：decode 这一步后，是否需要新块？
    # 答：如果 len(seq) % block_size == 1，说明上一步刚写满一块，
    #     这一步的新 token 是新块的第一个 token，需要分配
    return len(free_block_ids) >= (len(seq) % block_size == 1)
    # True  → 要么不需要新块，要么有空闲块可用
    # False → 需要新块但空闲块不足

# 注意：调用时机是 decode 前，len(seq) 是 append 当前 token 之前的长度
# 例：block_size=256，len(seq)=256 → 下一个 token 会是第 257 个，需要新块
#     256 % 256 = 0 ≠ 1 → 不需要新块？
# 仔细分析：when len(seq) % block_size == 1：
#   表示 seq 已有 k*256+1 个 token
#   即上一步 append 后刚好进入新块（第 k*256+1 个 token 是新块第 1 个）
#   这一步的 decode token 将写入第 k+1 块的第 2 个 slot
#   等等……让我重新分析

# may_append 调用时，len(seq) 已经加了新 token：
def may_append(seq):
    if len(seq) % block_size == 1:
        # 新 token 是某块的第 1 个 token，前一块已满
        # 需要为下一个 token 分配新块
        # 实际上：这里 len(seq) 是 append_token 之后的长度
        seq.block_table.append(_allocate_block())
```

### 抢占（Preemption）策略

```python
def preempt(seq):
    seq.status = WAITING
    seq.is_prefill = True
    block_manager.deallocate(seq)    # 释放 KV 块（ref_count--）
    waiting.appendleft(seq)          # 插到 waiting 头部（优先重新调度）

# 注意：
# 1. deallocate 后，seq 的 block_table 清空，num_cached_tokens 归零
# 2. 但 hash_to_block_id 中 seq 填满过的块的 hash 仍在！
#    → 下次重新 prefill 时，前缀缓存大概率命中，无需重新计算
# 3. 抢占顺序：running 队尾（LIFO）——最晚加入的，已 decode 最少 token
#    → 代价最小（重计算量少）
# 4. 当前不做 swap（换出到 CPU），直接 recompute
#    → 简单，但网络带宽大时 swap 更优（见 04_implementation_guide.md）
```

---

## 七、完整流程示意图

```
generate(prompts) ──────────────────────────────────────────────────►
                  │
                  ▼  tokenize + add Sequence to waiting
       ┌──────────────────────────────────────────────┐
       │ while not is_finished():                      │
       │                                               │
       │   ┌── schedule() ──────────────────────┐     │
       │   │  prefill:                           │     │
       │   │    can_allocate → allocate          │     │
       │   │    设置 num_scheduled_tokens        │     │
       │   │    → waiting/running 转移           │     │
       │   │  decode:                            │     │
       │   │    can_append → may_append          │     │
       │   │    → 必要时 preempt                 │     │
       │   └────────────────────────────────────┘     │
       │           │ seqs, is_prefill                  │
       │           ▼                                   │
       │   ┌── model_runner.call("run") ────────┐     │
       │   │  prepare_prefill/decode             │     │
       │   │    构造 input_ids, positions         │     │
       │   │    计算 slot_mapping                │     │
       │   │    set_context(...)                 │     │
       │   │  run_model                          │     │
       │   │    eager forward 或 CUDA graph     │     │
       │   │    → Triton KV 写入                 │     │
       │   │    → FlashAttention                 │     │
       │   │  sampler (rank 0)                   │     │
       │   │  reset_context()                    │     │
       │   │  return token_ids                   │     │
       │   └────────────────────────────────────┘     │
       │           │ token_ids                         │
       │           ▼                                   │
       │   ┌── postprocess() ───────────────────┐     │
       │   │  hash_blocks (注册满块哈希)         │     │
       │   │  num_cached_tokens 累积             │     │
       │   │  append_token                       │     │
       │   │  检查 EOS / max_tokens → FINISHED   │     │
       │   └────────────────────────────────────┘     │
       └──────────────────────────────────────────────┘
                  │
                  ▼  收集所有 FINISHED 的 output
       return [{"text": ..., "token_ids": ...}, ...]
```
