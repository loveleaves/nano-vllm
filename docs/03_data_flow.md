# 数据流与调度策略

> **统一连续批（A 轮对齐 V1）**：nano 已无 prefill/decode 阶段切换——一步内 decode（query=1）
> 与 prefill chunk（query>1）可混排在同一 varlen 批。下文"一、Prefill 数据流"与"二、Decode
> 数据流"分开讲解仅为**教学清晰**（对应一个新批次的首步 prefill 与后续 decode），实际调度二者混合。
> 数据通路：`EngineCore.step` → `scheduler.schedule()`(产 `SchedulerOutput`) →
> `executor.execute_model` → `ModelRunner.run`（`InputBatch.update`→`make_inputs`→`run_model`→
> `Sampler`）→ `scheduler.update_from_output`。注意力元数据 `AttentionMetadata` 经 forward 链
> **显式透传**（非全局 Context）；decode 与 prefill 统一走 `flash_attn_varlen_func`。

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
return SchedulerOutput(scheduled_seqs=[seqA, seqB],
                       num_scheduled_tokens={A:100, B:80}, finished_seq_ids=set())
```

### Step 2：InputBatch.update + make_inputs（构造输入张量）

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

# 无前缀缓存，block_table = None；以上数组写入 InputBatch 常驻 pinned 缓冲后单次异步 H2D，
# 打包成 AttentionMetadata 显式传入 forward（非全局 Context）
attn_md = AttentionMetadata(query_start_loc=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
                            max_query_len=100, max_seq_len=100,
                            slot_mapping=slot_mapping, block_table=None)
```

### Step 3：model.forward(input_ids[180], positions[180], attn_md)

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

### Step 5：Sampler（结构化采样层，rank0）

由 `prepare_sample` 按行序构造 `SamplingMetadata`，`Sampler` 逐行 `where(temp<eps, argmax, 随机)`：

```python
# temperature=0 → 真·greedy（argmax）；否则温度缩放 + 可选 top-k/top-p + 惩罚，再 Gumbel 随机采样
# 此例 temperature=0.6（纯随机路径，无 top-k/p/惩罚）：
logits = logits.float().div_(temperature.unsqueeze(1))   # [2, 151936]
probs = softmax(logits, dim=-1)
token_ids = (probs / Exponential(1).clamp_min(1e-10)).argmax(dim=-1)   # [2] → 如 [42, 1337]
# 行序 token 按 seq_id 映射回 scheduled_seqs 顺序后返回
```

### Step 6：Scheduler.update_from_output()

```python
for seq, token_id in zip([seqA, seqB], [42, 1337]):
    kv_cache_manager.hash_blocks(seq)
    # seqA：(0+100)//256=0 → 不满一块，无哈希注册；seqB 同

    seq.num_cached_tokens += seq.num_scheduled_tokens   # A:100  B:80
    if seq.is_prefill:        # chunk 未覆盖完整 prompt 时本步不产 token（此例已覆盖）
        continue
    seq.append_token(token_id)
    # seqA.token_ids = [1,...,100, 42]；seqB.token_ids = [5,...,84, 1337]
    if (not seq.ignore_eos and token_id == eos) or seq.num_completion_tokens == seq.max_tokens:
        → FINISHED + deallocate（并记入 scheduler.finished_req_ids，下步 InputBatch 回收其行）
    else:
        → 留在 running，下步 decode
```

---

## 二、Decode 阶段数据流

```
状态：seqA(101 token), seqB(81 token) 均在 running
```

### Step 1：Scheduler.schedule()（本批仅 decode，无 waiting）

```python
# RUNNING 段：每 seq decode 1 token（受 max_num_seqs / token 预算约束）
for seq in [seqA, seqB]:                # 经 may_append 在块边界按需分配新块
    can_append(seqA): 101 % 256 = 101 ≠ 1 → 不需新块；may_append 不分配
    seq.num_scheduled_tokens = 1
# WAITING 段为空 → 无 prefill chunk 混入
return SchedulerOutput(scheduled_seqs=[seqA, seqB], num_scheduled_tokens={A:1, B:1})
```

### Step 2：InputBatch.update + make_inputs（decode）

```python
# 持久行已存在（A/B 上步即在批中）→ update 仅按需 append_row 新块；decode 不重建块表
input_ids  = [42, 1337]           # 各 seq 的最新 token，[2]
positions  = [100, 80]            # num_cached_tokens（= 历史长度），[2]
cu_seqlens_q = [0, 1, 2]          # 每 seq query=1
cu_seqlens_k = [0, 101, 81+101]   # = [0,101,182]，KV 总长（历史+本步）

# slot_mapping：新 token 写入位置（向量化 compute_slot_mapping）
# seqA: block_table[100//256]=3 → 3*256+100=868；seqB: 7*256+80=1872
slot_mapping = [868, 1872]
block_table = block_table.gpu[:2]  # 常驻块表前 2 行：[[3,...],[7,...]]
attn_md = AttentionMetadata(query_start_loc=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
                            max_query_len=1, max_seq_len=101,
                            slot_mapping=slot_mapping, block_table=block_table)
```

### Step 3：model.forward(input_ids[2], positions[2], attn_md)

```
x = embed_tokens([42, 1337])   # [2, 2048]

for layer in layers:
    qkv = qkv_proj(x)          # [2, 4096]
    q = qkv[:, :2048].view(2, 16, 128); k = qkv[:, 2048:3072].view(2, 8, 128); v = qkv[:, 3072:].view(2, 8, 128)

    # ① 写 KV cache（只写当前步的 K/V）
    store_kvcache(k, v, k_cache, v_cache, slot_mapping)   # k_cache[3,100]←seqA；k_cache[7,80]←seqB

    # ② 统一 FlashAttention varlen（decode = query_len 1 的退化；按 block_table 读历史 KV）
    o = flash_attn_varlen_func(
        q, k, v,                       # [2, 16/8, 128]（每 seq 1 个 query）
        cu_seqlens_q=[0,1,2], cu_seqlens_k=[0,101,182],
        max_seqlen_q=1, max_seqlen_k=101,
        block_table=[[3,-1,...],[7,-1,...]], causal=True,
    )   # [2, 16, 128]

    x = o_proj(o.view(2, -1)); ...   # ffn

# compute_logits：用 query_start_loc[1:]-1=[0,1] 取每 seq 末 token → [2, vocab]
```

---

## 三、CUDA Graph Decode 数据流（decode + graph replay）

CUDA Graph 只在 decode 阶段且 `bs <= 512` 时使用。以 `bs=3`，`graph_bs=4` 为例：

### Capture 时（bs=4，静态）

```python
# 录制时的静态张量（地址固定，存入 CUDA graph）
input_ids_static    = [0, 0, 0, 0]       # 全零占位
positions_static    = [0, 0, 0, 0]
slot_mapping_static = [-1, -1, -1, -1]   # -1 = 跳过写入
cu_seqlens_q_static = [0, 1, 2, 3, 4]    # decode 恒为 arange（常量，replay 不更新）
cu_seqlens_k_static = [0, 0, 0, 0, 0]
block_tables_static = [[0,...], ...]      # 全零占位

# 录制 model forward（静态 AttentionMetadata 显式传入；max_seq_len 取 max_model_len）
attn_md = AttentionMetadata(cu_seqlens_q_static, cu_seqlens_k_static, 1, max_model_len,
                            slot_mapping_static, block_tables_static)
with torch.cuda.graph(graph, pool):
    outputs_static = model(input_ids_static, positions_static, attn_md)
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

gv["slot_mapping"].fill_(-1)               # 全部重置为 -1（dummy 跳过写入）
gv["slot_mapping"][:3] = [868, 1872, 500]  # 前 3 个填真实 slot

gv["cu_seqlens_k"].zero_()
gv["cu_seqlens_k"][:4] = [0, 101, 182, 232]  # 前 3 段有效（KV 累计长度）

gv["block_tables"][:3] = [[3,-1,...], [7,-1,...], [5,2,...]]

graph.replay()    # 一次 GPU 调用，执行全部 kernel

# dummy 第 4 个 seq 的处理结果：
#   slot=-1 → store_kvcache 跳过（不写 kv_cache）
#   cu_seqlens_k 末段长 0 → 其输出被 compute_logits 按 query_start_loc[1:]-1 取 token 时截掉

token_ids = compute_logits(gv["outputs"][:3], attn_md)  # 只取前 3 个
```

### CUDA Graph 与 AttentionMetadata 交互

CUDA Graph 录制的是模型执行时的 GPU 指令；Attention 层从**传入的** `AttentionMetadata` 读取
`slot_mapping`/`cu_seqlens_k`/`block_table`。这些静态张量的 **GPU 地址** 被硬编码进 graph，replay
时 GPU 从同一地址读取——因此修改静态张量的"内容"等效于修改 graph 的"输入"，无需重录。

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

### make_inputs（seqC）

```python
start = seqC.num_cached_tokens = 256   # 从第 256 个 token 开始
input_ids = seqC.token_ids[256:]       # 只处理 256 之后的 token
positions  = range(256, len(seqC))

cu_seqlens_q = [0, len(seqC)-256]     # query 数 = 实际需处理 token
cu_seqlens_k = [0, len(seqC)]         # KV 总长 = 历史（256）+ 当前处理 token

# slot_mapping：只计算新写入的部分（num_cached_tokens 之后），block_table 取常驻行
slot_mapping = ...   # 从 seqC.block_table[1] 开始
attn_md = AttentionMetadata(query_start_loc=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
                            max_query_len=len(seqC)-256, max_seq_len=len(seqC),
                            slot_mapping=slot_mapping, block_table=[[2, new_block_id]])
```

### FlashAttention（前缀缓存路径）

```python
# attention/flash_attn.py：block_table 非 None 即统一从分页 cache 读历史
o = flash_attn_varlen_func(
    q,           # 只有本步的 query（len(seqC)-256 个）
    k_cache, v_cache,                 # 整个 KV cache 张量（FA 按 block_table 读取）
    cu_seqlens_q=[0, len(seqC)-256],
    cu_seqlens_k=[0, len(seqC)],     # 告诉 FA：K 有 len(seqC) 个（含历史）
    block_table=attn_md.block_table,  # FA 按此映射读取 Block 2 的历史 KV
    causal=True,
)
```

**效果**：SeqC 的前 256 token 完全跳过了 forward 计算（embedding、QKV 投影、RoPE、Attention 都没算），直接复用 SeqA 的 KV 结果。

---

## 五、Chunked Prefill 数据流

SeqA（2000 token prompt），max_num_batched_tokens=1024，running=[SeqB, SeqC]（decode 中）：

### Step 1（decode + SeqA 第一个 chunk **混排**）

```
schedule()（统一连续批）：
  ① RUNNING：SeqB、SeqC 各 decode 1 token（占 2 token 预算）
  ② WAITING：SeqA chunk = min(2000-0, 1024-2) = 1022；处理后 num_cached=1022（< 2000，留 waiting 队首）
return SchedulerOutput(scheduled_seqs=[SeqB, SeqC, SeqA], num_scheduled_tokens={B:1, C:1, A:1022})
```

> SeqB/SeqC 的 decode **不再被 prefill 阻塞**——它们与 SeqA 的 prefill chunk 在同一 varlen 批内
> 一起前向（`query_start_loc=[0,1,2,1024]`）。

### Step 2（SeqA 后续 chunk，继续与 decode 混排）

```
schedule()：① SeqB/SeqC decode 各 1；② SeqA chunk = 2000-1022 = 978（≤ 剩余预算）
  1022 + 978 = 2000 == prompt → SeqA 转 running、本步产出首 token
```

### Step 3 起（SeqA 也进入 decode）

```
schedule() → scheduled_seqs=[SeqB, SeqC, SeqA]，全部 decode（num_scheduled 各 1）
```

**update_from_output 对 chunk 中间步的处理：**

```python
# sched/scheduler.py: update_from_output()
for seq, token_id in zip(scheduled_seqs, token_ids):
    kv_cache_manager.hash_blocks(seq)
    seq.num_cached_tokens += seq.num_scheduled_tokens
    if seq.is_prefill:          # chunk 未覆盖完整 prompt（num_cached < num_prompt）
        continue                # ← 本步不 append_token
    seq.append_token(token_id)  # 仅 prompt 覆盖完整后才追加
```

**为什么中间 chunk 不产出 token？** `compute_logits` 按 `query_start_loc[1:]-1` 取每段末 token，
chunk 1 的末 token 是 token[1021]，并非 prompt 最后一个 token，据它采样无意义；`is_prefill`
（由 `num_cached_tokens < num_prompt_tokens` 派生）为真时跳过。EngineCore.step 也据此不产增量输出。

---

## 六、Scheduler 调度策略详解

### 统一连续批调度（一步内 decode + prefill chunk）

`schedule()` 单步内分两段填充同一批（无 prefill/decode 阶段标志）：

```
① RUNNING 段（decode 优先，保证已在生成的请求低延迟推进）：
     每 running seq 调度 1 token；块边界时 may_append 分配新块；
     KV 不足则从 running 末尾 preempt（recompute）。
② WAITING 段（用剩余 token 预算做 prefill chunk）：
     队首 seq 首次调度时 can_allocate 探测前缀缓存命中块、allocate；
     可分块（任意队首 seq），chunk = min(剩余 prompt, 剩余预算)；
     覆盖完整 prompt 即转入 running。
```

decode 段先填，确保正在 decode 的请求不被长 prompt 的 prefill 饿死；二者混排在一个 varlen 批，
`query_start_loc` 区分各段。这与 vLLM 连续批一致；不再有"优先 prefill 独占步""仅第一个 seq 分块"。

### Prefill 段（WAITING）流程

```
while waiting and len(scheduled) < max_num_seqs:
    seq = waiting.peek_request()                  # 不 pop，先探测
    remaining = max_num_batched_tokens - num_batched_tokens
    if remaining <= 0: break
    if not seq.block_table:                       # 首次：探测前缀缓存
        num_cached = can_allocate(seq)            # -1 表示内存不足
        if num_cached == -1: break
        num_tokens = seq.num_tokens - num_cached * block_size
    else:                                         # 分块续传
        num_tokens = seq.num_tokens - seq.num_cached_tokens
    n = min(num_tokens, remaining)
    if n <= 0: break
    if not seq.block_table: allocate(seq, num_cached)
    seq.num_scheduled_tokens = n; num_batched_tokens += n
    if seq.num_cached_tokens + n == seq.num_tokens:   # prompt 覆盖完整 → 转 running
        seq.status = RUNNING; waiting.pop_request(); running.append(seq)
    scheduled.append(seq)
```

### Decode 段（RUNNING）流程

```
while running and len(scheduled) < max_num_seqs and num_batched_tokens + 1 <= budget:
    seq = running.popleft()
    while not can_append(seq):                    # KV 不足 → 抢占
        if running: preempt(running.pop()); 记入 preempted_seq_ids   # 末尾（重计算量最小）
        else:       preempt(seq); seq = None; break                  # 自己也装不下
    if seq is None: break
    seq.num_scheduled_tokens = 1; may_append(seq)   # 按需分配新块（块边界时）
    scheduled.append(seq)
running.extendleft(reversed(decode_scheduled))      # 保持 FIFO 顺序放回
```

> 被抢占（及上一步结束）的 seq_id 汇入 `SchedulerOutput.finished_seq_ids`，下一步由各 rank 的
> InputBatch 回收其持久行槽位。

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
    block_manager.deallocate(seq)    # 释放 KV 块（ref_count--），num_cached_tokens 归零
    waiting.prepend_request(seq)     # 插到 waiting 头部（优先重新调度）
    # is_prefill 是派生属性（num_cached_tokens < num_prompt_tokens），归零后自动恢复为 True

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
LLM.generate(prompts) ───────────────────────────────────────────────►
       │  Processor.process_inputs（tokenize）→ EngineCore.add_request（加入 waiting）
       │                                       + OutputProcessor.add_request
       ▼
   ┌──────────────────────────────────────────────────────────────┐
   │ while EngineCore.has_unfinished_requests():                   │
   │   ┌── EngineCore.step() → EngineCoreOutputs ──────────────┐  │
   │   │  schedule() → SchedulerOutput                          │  │
   │   │    RUNNING decode（can_append/may_append，必要时 preempt）│
   │   │    + WAITING prefill chunk（can_allocate/allocate）混排  │  │
   │   │    产 finished_seq_ids                                  │  │
   │   │        │ scheduled_seqs, finished_seq_ids              │  │
   │   │        ▼                                                │  │
   │   │  executor.execute_model(seqs, finished_seq_ids)         │  │
   │   │    (UniProc 内联 / MultiProc 隔离) → ModelRunner.run：   │  │
   │   │      InputBatch.update（增量行 + 回收）                 │  │
   │   │      → make_inputs（行序展开 + 单次 H2D，构 AttentionMetadata）│
   │   │      → run_model（eager / CUDA graph）→ Triton KV 写入 + FlashAttention varlen │
   │   │      → Sampler（rank0 结构化采样）→ 按 seq_id 映射回    │  │
   │   │        │ token_ids                                      │  │
   │   │        ▼                                                │  │
   │   │  update_from_output(): hash_blocks + num_cached 累积    │  │
   │   │      + append_token + EOS/max_tokens 检查 → FINISHED    │  │
   │   └─────────────────────────────────────────────────────────┘ │
   │   OutputProcessor.process_outputs(): 增量 detokenize + 停止串 → RequestOutput │
   └──────────────────────────────────────────────────────────────┘
       │
       ▼  收集所有 FINISHED 的 output（AsyncLLM 则逐步流式 yield）
   return [{"text": ..., "token_ids": ...}, ...]
```
