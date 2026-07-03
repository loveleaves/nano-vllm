# vLLM Disaggregated Prefilling(P/D 分离)技术调研报告

> 参考来源:vLLM 官方文档([Disaggregated Prefilling](https://docs.vllm.ai/en/latest/features/disagg_prefill/))、vLLM 源码(`vllm/distributed/kv_transfer`)、DeepWiki 架构解析,以及 DistServe、Splitwise、Mooncake 等相关学术论文。
> 报告版本对应 vLLM 最新开发版文档(该特性目前仍标记为 **experimental**)。

---

## 1. 背景与动机

### 1.1 LLM 推理的两阶段特性

一次 LLM 自回归生成请求可以拆分为两个计算特征截然不同的阶段:

| 阶段 | 计算特征 | 关键指标 | 典型瓶颈 |
|---|---|---|---|
| **Prefill(预填充)** | 一次性并行处理全部输入 token,计算 KV Cache | TTFT(Time To First Token,首 token 延迟) | **计算密集型(compute-bound)**,吞吐主要受 GEMM/FLOPs 限制 |
| **Decode(解码)** | 逐 token 自回归生成,每步只算 1 个新 token | TPOT / ITL(Time Per Output Token / Inter-Token Latency) | **访存密集型(memory-bandwidth-bound)**,主要受 KV Cache 读取带宽限制 |

在传统的"合并式"(co-located)vLLM 单实例部署中,prefill 和 decode 请求会被同一个调度器混合批处理(continuous batching)。这带来两个核心问题:

1. **阶段间干扰(interference)**:当一个新请求的 prefill 被插入到正在进行的 decode batch 中时(即便使用了 chunked prefill),会挤占本应连续、稳定输出的 decode 步骤,导致其他正在解码请求的 **tail ITL 抖动**变大,体验不稳定。
2. **资源配置耦合**:prefill 和 decode 对并行策略(TP/PP/DP)、批大小、硬件类型的最优选择往往不同,但在合并部署下两者被迫共享同一套配置,无法分别针对 TTFT 和 ITL 做独立优化。

### 1.2 Disaggregated Prefilling 的核心思想

vLLM 官方文档给出的动机可以概括为两点:

- **独立调优 TTFT 与 ITL**:将 prefill 阶段与 decode 阶段放到不同的 vLLM 实例上运行,可以为两者分别指定不同的并行策略(如 TP、PP 大小),从而在不影响 ITL 的前提下优化 TTFT,或反之。
- **控制尾部 ITL(tail ITL)**:不做分离时,vLLM 可能会在某个请求解码过程中插入其他请求的 prefill 任务,造成更高的尾延迟;分离部署可以从根本上避免这种"插队",把 tail ITL 控制住。

这与学术界在 2024 年前后掀起的 **Prefill/Decode(PD)分离** 研究浪潮(DistServe、Splitwise、TetriInfer、Mooncake 等)思路一致——用"目标达成率(Goodput)"取代单纯的吞吐量作为优化目标:即在同时满足 TTFT 和 TPOT SLO 的前提下,系统能承载的最大请求速率。

---

## 2. 学术脉络与业界方案对比

vLLM 的 Disaggregated Prefilling 并非孤立设计,而是与整个学术/工业界的 PD 分离方向同步演进:

| 系统 | 机构/来源 | 核心贡献 |
|---|---|---|
| **Splitwise**(ISCA'24) | Microsoft | 首批提出将 prompt 处理与生成阶段拆分到不同机器池,并可为两阶段配置不同硬件(如低算力/低成本 GPU 用于 decode);采用分层 KV Cache 传输设计以提升计算-通信重叠度 |
| **DistServe**(OSDI'24) | 北大团队 | 提出 **Goodput**(满足 TTFT 与 TPOT SLO 的最大请求速率)概念;通过为 prefill/decode 独立选择并行策略与 GPU 配比,实现相比同类系统最高 7.4× 的 goodput 提升或 12.6× 更紧的 SLO;论文验证在 NVLink 级带宽下 KV 传输开销可低至 <0.1% |
| **TetriInfer** | 学术界 | 面向混合负载场景的 PD 分离处理 |
| **Mooncake** | 月之暗面(Moonshot AI) | 以 **KV-cache 为中心**的分离式架构,支持 SLO 感知调度与 D2D/RDMA 传输,是 vLLM 生态中重要的第三方 Connector 实现来源之一 |
| **NVIDIA Dynamo** | NVIDIA | 面向数据中心规模的分离式推理编排框架,与 vLLM/TensorRT-LLM/SGLang 均有集成 |

**共识与分歧**:多篇论文(如 *Prefill-Decode Aggregation or Disaggregation?*, 2025)指出,PD 分离在 **TPOT 要求严格、TTFT 相对宽松**的场景下收益最大;而在 TTFT 要求极严格、TPOT 宽松的场景下,传统的"合并 + chunked prefill"反而可能更优。这也解释了为什么 vLLM 官方文档明确将该特性标注为 **experimental**,并强调它高度依赖具体基础设施(网络带宽、拓扑)。

---

## 3. vLLM 的整体架构设计

### 3.1 高层架构

vLLM 通过运行 **两类独立的 vLLM 实例** 来实现 PD 分离:

```
                     ┌─────────────┐        KV Cache (RDMA/NIXL/...)      ┌─────────────┐
   请求 ──▶ Proxy ──▶│ Prefill 实例 │ ─────────────────────────────────▶ │ Decode 实例  │──▶ 流式输出
                     │ (Producer)   │        + 元数据 (block ids 等)       │ (Consumer)   │
                     └─────────────┘                                      └─────────────┘
```

- **Prefill 实例(KV 生产者/producer)**:接收完整 prompt,执行一次前向计算得到全部 KV Cache(以及首 token),然后通过 **Connector** 将 KV Cache 块和相关元数据传输给 decode 实例。
- **Decode 实例(KV 消费者/consumer)**:接收到 KV Cache 后,直接跳过 prefill 计算,进入自回归解码循环,持续生成后续 token。
- **代理层(Proxy)**:负责编排请求路径——先将 prompt 发送给某个 prefill 实例,待其完成后,再将请求(附带 KV 传输所需的元数据)转发给某个 decode 实例继续生成。

vLLM 团队在源码中的定位非常明确:**分离式推理与具体基础设施(网络拓扑、互联硬件)强相关**,因此 vLLM 核心团队本身不绑定某一种传输实现,而是把该功能设计为一套**可插拔的 Connector 接口**,交给社区/厂商基于自身硬件条件去实现和优化,vLLM 团队负责评审并合并第三方 Connector 的 PR。所有实现均位于代码路径:

```
vllm/distributed/kv_transfer/
```

### 3.2 两种可扩展方式(官方文档给出)

vLLM 文档中给出了实现自定义 Connector 的两条路径:

1. **完全自定义 Connector(Fully-customized connector)**:直接实现自己的 `Connector`,调用任意第三方库来发送/接收 KV Cache,甚至可以修改 vLLM 的模型输入以执行定制化的 prefill 逻辑。灵活度最高,但存在与未来 vLLM 版本不兼容的风险。
2. **类数据库式 Connector(Database-like connector)**:实现自己的 `LookupBuffer`,并支持 `insert` 与 `drop_select` 两个类似 SQL 的接口——`LookupBuffer` 允许 KV 消费者(consumer)按需检索一批请求对应的 KV Cache,这种模式更像是把 KV Cache 当作一个可查询的存储服务。

---

## 4. 核心实现原理:KVConnectorBase_V1 接口

vLLM V1 引擎中,所有 Connector 都必须实现抽象基类 **`KVConnectorBase_V1`**(位于 `vllm/distributed/kv_transfer/kv_connector/v1/base.py`)。该接口最核心的设计是:**严格区分运行在 Scheduler 进程与 Worker 进程中的方法**。

### 4.1 两种角色(KVConnectorRole)

| 角色 | 运行位置 | 职责 |
|---|---|---|
| `KVConnectorRole.SCHEDULER` | 调度器(Scheduler)进程 | 请求状态跟踪、元数据组装、KV 块的持有/释放决策 |
| `KVConnectorRole.WORKER` | 每个 GPU Worker 进程 | 显存注册、异步数据传输发起、传输状态轮询 |

每个 Connector 会分别为这两个角色实例化一次。

### 4.2 Scheduler 侧 API

| 方法 | 作用 |
|---|---|
| `get_num_new_matched_tokens(request, num_computed_tokens)` | 返回可从外部 KV Cache 加载的 token 数量,用于 decode 实例判断可跳过多少本地计算 |
| `update_state_after_alloc(request, blocks, num_external_tokens)` | 在本地完成块分配后调用,记录哪些块需要被加载/保存 |
| `build_connector_meta(scheduler_output)` | 组装每一步(step)供 Worker 使用的传输元数据(`KVConnectorMetadata`) |
| `request_finished(request, block_ids)` | 请求完成时调用一次,返回 `(delay_free, kv_transfer_params)`,决定 KV 块是否延迟释放以等待远端消费者取走 |

### 4.3 Worker 侧 API

| 方法 | 作用 |
|---|---|
| `register_kv_caches(kv_caches)` | 启动时调用,传入每层的 KV Cache 张量,完成 GPU 显存注册(如 RDMA pin memory) |
| `register_cross_layers_kv_cache(kv_cache, attn_backend)` | 面向跨层连续张量布局的替代注册方式 |
| `start_load_kv(forward_context)` | 发起**异步**、非阻塞的 KV 数据加载传输 |
| `wait_for_layer_load(layer_name)` | 阻塞等待某一层的 KV 数据加载完成(用于逐层 overlap) |
| `save_kv_layer(layer_name, kv_layer, attn_metadata)` | 将某一层刚计算出的 KV 写出给 Connector(prefill 侧调用) |
| `wait_for_save()` | 阻塞等待所有进行中的保存操作完成 |
| `get_finished(finished_req_ids)` | 返回 `(finished_sending, finished_recving)` 两个请求 ID 集合,供调度器判断传输是否结束 |

这种"逐层(per-layer)"粒度的设计非常关键:它允许 KV Cache 的传输与模型前向计算 **overlap(重叠)**——例如在 prefill 侧,某一层刚算完 KV 就可以立即通过 `save_kv_layer` 发起写出,而不必等待整个前向传播结束,从而将通信开销尽量隐藏在计算之后的层里,降低有效 TTFT 增量。

### 4.4 请求生命周期(以 NIXL 为例的典型 workflow)

1. 请求进入 Proxy,被路由到某个 **Prefill 实例**。
2. Prefill 实例逐层计算 KV Cache,通过 `save_kv_layer` 将新计算的层写出(在 chunked prefill 场景下,KV 块会累积到最后一个 chunk 计算完才真正发起写传输)。
3. Prefill 完成后(状态为 `FINISHED_LENGTH_CAPPED`,即"因长度原因提前终止"以移交给 decode 侧继续生成),Scheduler 侧 `request_finished` 返回包含 `remote_block_ids`、`remote_engine_id`、`remote_request_id`、`remote_host`、`remote_port`、`tp_size` 等字段的 `kv_transfer_params`。
4. Proxy 将请求(携带上述元数据)转发给 **Decode 实例**。
5. Decode 实例的 Scheduler 通过 `get_num_new_matched_tokens` 判断可复用多少远端 KV,分配本地显存块后,Worker 通过 `start_load_kv` 发起对应的异步拉取/接收。
6. 数据到达后,Decode 实例跳过 prefill 直接开始自回归生成,并持续通过 `get_finished` 上报传输完成状态,便于 Prefill 侧安全释放已发送完毕的 KV 块。

---

## 5. Connector 工厂与官方/社区已支持的实现

`KVConnectorFactory`(`vllm/distributed/kv_transfer/kv_connector/factory.py`)维护了一个按字符串索引、**懒加载**的 Connector 注册表。已知(预注册或社区广泛使用)的实现包括:

| Connector 名称 | 传输/存储机制 | 主要用途 |
|---|---|---|
| **NixlConnector** | 基于 [NIXL](https://github.com/ai-dynamo/nixl)(NVIDIA 主导的异构内存/网络加速通信库)的 RDMA 风格 GPU 显存直传 | **生产级分离式 prefill 的主力实现** |
| **MooncakeConnector / MooncakeStoreConnector** | Mooncake 的 D2D/RDMA 传输 + KV-cache 中心化存储池 | 大规模、SLO 感知的 KV 迁移与共享 |
| **LMCacheMPConnector** | 多进程架构对接 LMCache 中心化缓存服务器 | 跨进程/跨实例 KV 复用与分离式服务 |
| **P2pNcclConnector** | 基于 NCCL 的点对点传输 | 无需 RDMA 网卡场景下的 GPU 间直传 |
| **MoRIIOConnector** | AMD MORI(Modular RDMA Interface)开源 RDMA 框架 | 面向 AMD GPU(如 MI300X)的单节点/多节点 PD 分离 |
| **FlexKVConnectorV1** | FlexKV 分布式 KV 存储与多级缓存管理 | 超大规模 LLM 推理下的多级 KV 缓存管理 |
| **MultiConnector** | 组合多个 Connector | 例如同时对接"分离式传输 + 前缀缓存卸载"等多种能力 |
| **OffloadingConnector** | GPU↔CPU 拷贝(CUDA Stream) | **单实例内**的 KV 显存卸载(与跨实例的 PD 分离是相邻但不同的能力) |

> 值得注意的是,官方文档特别强调:**分离式 prefill 的生产级能力依赖第三方 Connector**,vLLM 核心团队更多扮演"接口维护者 + PR 评审者"的角色,这与 vLLM 在其他特性(如量化、Attention Backend)上"官方深度实现"的策略有所不同,体现出该领域对底层硬件/网络高度定制化的现实。

### 5.1 NixlConnector 深入:握手协议与兼容性

作为目前生产环境中最主流的实现,`NixlConnector` 的关键机制包括:

- **`NixlConnectorScheduler`**(调度器进程):跟踪跨调度步骤的请求状态;`request_finished` 中若 `do_remote_decode=True` 且请求状态为 `FINISHED_LENGTH_CAPPED`,则将其加入待发送队列 `_reqs_need_send`。同时启动一个 ZMQ 监听守护线程(`_nixl_handshake_listener`),向发起连接的 decode worker 提供 `NixlHandshakePayload`。
- **`NixlConnectorWorker`**(每个 GPU worker 进程):持有 NIXL agent,负责真正的异步块传输。`register_kv_caches` 会依次调用 `get_reg_descs()` 获取每个 KV Cache 张量的描述符,再调用 `register_memory()` 完成显存 pin(RDMA 注册),并计算出供远端 worker 使用的握手元数据。
- **握手(Handshake)协议**:decode worker 首次向某个 prefill engine 发起传输前,需要先获取并注册对方 NIXL agent 的信息,该握手是**懒加载(lazy)**触发的,即在首次传输时才建立。
- **兼容性哈希(compatibility_hash)**:通过 `NIXL_CONNECTOR_VERSION`(当前为 2)结合 vLLM 版本、模型架构、KV head 数量等信息计算出一个哈希值,用来校验 prefill 与 decode 实例之间的互操作性,避免因模型/版本不一致导致传输错误。

### 5.2 与模型执行的集成

`KVConnectorModelRunnerMixin`(`vllm/v1/worker/kv_connector_model_runner_mixin.py`)将 Connector 生命周期嵌入到 `GPUModelRunner.execute_model()` 的每一步中,通过上下文管理器 `_get_kv_connector_output()` 封装"发起加载 → 逐层等待/保存 → 收尾"整个流程;`KVOutputAggregator` 负责在 TP(张量并行)场景下,将各 TP rank 产生的 `KVConnectorOutput` 聚合后再返回给调度器。

### 5.3 可观测性(Metrics)

vLLM 为 Connector 设计了统一的统计接口 `KVConnectorStats`,并为不同实现提供了专用子类,例如 `NixlKVConnectorStats`(记录传输耗时、字节数、描述符数量)、`MultiKVConnectorStats`(聚合多个子 Connector 的统计)、`OffloadingConnectorStats`(CPU/GPU 传输耗时与吞吐)。这为线上排查"KV 传输是否成为新瓶颈"提供了基础的监控手段。

---

## 6. 使用示例(Usage Example,概念性)

> 以下命令基于官方文档与示例目录(`examples/disaggregated/`)整理,具体参数请以所用 vLLM 版本的实际文档为准。

启动 Prefill 实例(生产者角色):

```bash
vllm serve <model> \
  --kv-transfer-config \
  '{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_rank":0,"kv_parallel_size":2}' \
  --port 8100
```

启动 Decode 实例(消费者角色):

```bash
vllm serve <model> \
  --kv-transfer-config \
  '{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_rank":1,"kv_parallel_size":2}' \
  --port 8200
```

再由一个 Proxy(反向代理/网关)按照"先转发给 Prefill 实例、再转发给 Decode 实例"的顺序编排请求路径,并透传 `kv_transfer_params` 元数据。

FlexKV 场景下的角色配置示例(前缀缓存 + 分离式共用一套 Connector):

```bash
--kv-transfer-config '{"kv_connector":"FlexKVConnectorV1","kv_role":"kv_both"}'
```

vLLM 官方示例目录中也提供了教学用的最小实现 `example_connector.py`,用于演示如何从零实现一个符合 `KVConnectorBase_V1` 接口的自定义 Connector。

---

## 7. 收益、代价与工程权衡

### 7.1 收益

- **TTFT/ITL 解耦调优**:prefill、decode 实例可独立选择 TP/PP 并行度、批大小甚至硬件型号(如 decode 用更便宜的卡)。
- **消除阶段间干扰**:decode 实例不再被临时插入的 prefill 任务打断,tail ITL 更稳定、更可预测。
- **独立弹性伸缩**:可根据流量特征(长 prompt 多 or 高并发生成多)分别扩缩 prefill/decode 实例数量。
- **学术验证的性能提升**:DistServe 等系统在特定 SLO 场景下报告了最高 7.4× 的 goodput 提升,或以同样吞吐实现 12.6× 更紧的延迟 SLO(需注意具体数字依赖硬件、模型与工作负载假设)。

### 7.2 代价与挑战

| 问题 | 说明 |
|---|---|
| **KV 传输开销成为新瓶颈** | 有实测数据显示,在 8×MI300X(4P-4D)配置下,KV 传输带来平均 1.4× 吞吐开销与 1.9× TTFT 开销,尤其对**短 prompt** 更不利(计算量小,传输占比相对更高,难以被计算重叠) |
| **显存翻倍** | prefill、decode 各自维护一份完整模型权重,GPU 显存消耗成倍增长,对小显卡不友好 |
| **依赖高速互联** | 论文普遍假设 NVLink 级(600GB/s)或至少 InfiniBand 级(800Gbps)带宽;跨集群/跨数据中心场景下(50–100GB/s)传输开销会显著放大,已成为新的研究热点(如 SplitZip 提出的无损 KV 压缩传输) |
| **长上下文场景压力更大** | 文档级问答、代码库理解、多文档摘要等长输入场景下,prefill 产生的 KV Cache 体量巨大,对传输带宽和延迟要求更高 |
| **单/双卡场景不适用** | 在只有 1-2 张 GPU 的部署里,拆分 prefill/decode 的设计空间会坍缩,分离几乎无法带来收益 |
| **鲁棒性问题** | decode 侧节点故障可能连锁阻塞多个与之绑定的 prefill 节点,容错机制仍是开放问题 |
| **多轮对话下的重复计算** | 现有 vLLM 分离方案(以 NIXL Connector 为例)的 KV 传输是**单向**的(P→D),这导致多轮对话、KV 被 decode 侧驱逐后需要 prefill 侧重新计算增量前缀等场景存在冗余计算——这正是社区正在讨论的"双向 KV 传输"RFC(#32733)所要解决的问题 |

### 7.3 vLLM 官方的现实定位

正因为上述权衡高度依赖具体基础设施与工作负载,vLLM 官方文档明确将该特性标记为 **experimental(实验性)**,并给出一个清晰的架构立场:**核心引擎只负责定义稳定、清晰的 Connector 抽象接口(调度器/Worker 分离、逐层异步传输 API),把"如何传得又快又省"这件强基础设施相关的事情,开放给社区与硬件厂商(NVIDIA/NIXL、AMD/MORI-IO、Moonshot/Mooncake、LMCache 等)去做深度优化**。

---

## 8. SGLang 的 PD 分离技术方案深度调研

SGLang 是当前与 vLLM 并列的另一主流开源 LLM 推理引擎,其 PD 分离功能(官方文档路径 `docs/advanced_features/pd_disaggregation.md`)自 2025 年初的 Roadmap Issue(#3554、#4655)开始迭代,目前已经历多轮生产打磨(容错、投机解码、结构化输出、异构 TP 等特性均已支持),是理解"业界共识实现范式"的重要参照系。

### 8.1 设计动机:与 vLLM 高度一致,但更强调 DP Attention 场景

SGLang 官方文档给出的动机与 vLLM 基本一致,但额外强调了一个 vLLM 文档未特别展开的问题:

1. **Prefill Interruption(prefill 打断)**:新到达的 prefill batch 频繁打断正在进行的 decode batch,导致 token 生成出现明显停顿——这与 vLLM 强调的"tail ITL 抖动"是同一个问题的两种表述。
2. **DP Attention Imbalance(数据并行注意力负载不均)**:在启用 **DP Attention**(SGLang 面向 MoE 大模型如 DeepSeek-V3 的核心并行策略)时,某个 DP worker 可能在处理 prefill batch 的同时,另一个 DP worker 在处理 decode batch,导致 decode 延迟被拖慢——这是 SGLang 结合自身 DP Attention/MoE 架构提出的特有问题,也解释了为什么 SGLang 的 PD 分离与其大规模 MoE(如 DeepSeek-R1/V3)部署实践结合得尤为紧密。

### 8.2 三组件架构:Proxy / Prefill Server / Decode Server

SGLang 采用与 vLLM 相似但组件划分更明确的三层架构:

```
                 ┌───────────────┐
  请求 ──▶ Proxy/Router (可选独立节点或与 Prefill 同机部署)
                 └──────┬────────┘
                         │ 1) 路由到某一对 (Prefill, Decode)
                 ┌───────▼────────┐   RDMA (Mooncake / NIXL)   ┌────────────────┐
                 │  Prefill Server │──────────────────────────▶│  Decode Server  │
                 │ (KV Sender)     │      KV Cache + aux data   │ (KV Receiver)   │
                 └─────────────────┘                            └────────────────┘
```

请求处理流程(综合官方文档与 NVIDIA Dynamo 对 SGLang 集成文档描述):

1. **建连阶段(一次性/懒加载)**:Decode 侧向 Prefill 侧注册 RDMA 连接信息(如 GPU 显存基址指针),这一步类似 vLLM NixlConnector 的握手(handshake)。
2. **请求到达**:Proxy/Router 依据负载均衡策略选出一对 Prefill/Decode Server。
3. **Decode 侧预分配**:Decode Server 先为该请求**预分配**好 KV Cache 显存槽位,然后通过 **bootstrap server**(内置于 `tokenizer_manager` 中的一个协调服务,通过 `bootstrap_room` 标识请求)通知 Prefill Server 可以开始计算。
4. **Prefill 计算与发送**:Prefill Server 执行前向计算得到 KV Cache 与首 token,随后通过 `KVSender` 把数据经 RDMA(Mooncake 或 NIXL 后端)写入 Decode 侧已预留好的显存地址。
5. **Decode 接收与生成**:`KVReceiver` 收到数据后,Decode Server 直接进入自回归解码循环,持续生成 token 并流式返回给 Proxy/客户端。

### 8.3 核心工程设计原则

SGLang 官方博客(与 AMD 联合发布的 MI300X 分离式部署实践)总结了四条关键设计原则,与 vLLM 的"逐层异步"思路互为印证:

| 设计原则 | 说明 |
|---|---|
| **动态连接(Dynamic Connection)** | 每个请求单独建立一对 Prefill↔Decode 连接,而非固定绑定,使得 Prefill/Decode 实例池可以独立弹性扩缩 |
| **非阻塞传输(Non-blocking Transfer)** | 发送/接收操作运行在**后台线程**中,主调度器事件循环(scheduler event loop)不会被 KV 传输阻塞,与 vLLM"逐层异步 + `wait_for_layer_load`"的目的相同 |
| **异构并行(Heterogeneous Parallelism)** | 支持 Prefill 与 Decode 使用**不同的 TP 大小**(如 Prefill TP=4、Decode 用 DP Attention 等效 TP=1),两侧可以分别选择最优并行策略 |
| **基于 RDMA 的传输(RDMA-Based Transfer)** | 利用 RDMA 的 queue pair 建立连接,并用 scatter-gather elements(SGE)高效搬运非连续显存块,避免逐 token 拷贝的开销 |

### 8.4 KVSender / KVReceiver 与 Connector 抽象

与 vLLM 类似,SGLang 也定义了一套可插拔的 `KVTransferConfig`(`kv_connector`、`kv_role` 等字段,取值 `kv_producer`/`kv_consumer`/`both`),并在 `python/sglang/srt/disaggregation/` 目录下实现了以 `KVSender`/`KVReceiver` 为核心的传输接口。这套抽象与 vLLM `KVConnectorBase_V1` 的 Scheduler/Worker 双角色划分理念相通,但落地形式略有差异:SGLang 更强调"每请求一条独立传输线程 + 后台轮询"的模型,并通过一个专门的 **bootstrap server**(早期版本依赖 etcd,后改为内置 ZMQ 协调服务)来完成 Prefill/Decode 双方的元数据握手(如 `bootstrap_room → [kv_indices, aux_token]` 的映射)。

### 8.5 支持的传输后端(Transfer Engine)

| 后端 | 特点 |
|---|---|
| **Mooncake**(`mooncake-transfer-engine`) | 目前 SGLang 生态中最成熟、生产验证最多的后端;通过 `--disaggregation-ib-device` 指定 RDMA 网卡(支持按 GPU 分别指定不同网卡);支持 NVLink/NVL72 场景下的定制内存池(`SGLANG_MOONCAKE_CUSTOM_MEM_POOL=NVLINK`),以及节点内 NVLink 传输(`INTRA_NODE_NVLINK`) |
| **NIXL** | 与 vLLM NixlConnector 使用同一套底层 NIXL 库(基于 UCX),默认使用 UCX 后端,可通过 `SGLANG_DISAGGREGATION_NIXL_BACKEND` 切换为 LIBFABRIC 等其他插件 |
| **Ascend(华为昇腾 NPU)** | 通过 `memfabric_hybrid` 或复用 Mooncake 后端,支持在昇腾 NPU 集群上做 PD 分离(如 DeepSeek-R1/MiMo 系列模型的多机部署) |

### 8.6 异构 TP 下的 GPU Staging Buffer(工程亮点)

当 Prefill 与 Decode 使用**不同的 TP 大小**时(例如 Prefill TP=4、Decode 采用 DP Attention 使等效 Attention TP=1),两侧 KV Cache 在显存中的头(head)切分布局并不一致,若逐 token 直接搬运效率很低。SGLang 为此设计了 **GPU Staging Buffer** 机制:先在 Prefill 侧把待发送的 KV head 切片**聚合(gather)**进一块连续显存缓冲区,批量发起一次 RDMA 传输,再在 Decode 侧**散射(scatter)**回正确的 KV Cache 分页位置。据官方文档,该机制相比默认的逐 token 切片方式,在高并发下可带来 **2–5× 的吞吐提升**,并能将异构 TP 场景的性能拉近到同构 TP 基线的 ~5% 以内(注:该特性目前仅适用于 GQA/MHA 等非 MLA 模型,DeepSeek-V2/V3 等 MLA 模型不建议开启)。

### 8.7 生产可用性相关能力

相比 vLLM 当前仍标注为 experimental 的状态,SGLang 的 PD 分离在以下工程能力上已经比较完善,体现了更强的生产化程度:

- **容错与重连**:支持传输失败自动 abort 请求(`[PD] Abort request if transfer fails`),支持 Decode 节点故障后与其绑定的 Prefill 侧安全重连而不影响其他实例(`Handle P/D failure and reconnect without affecting other instances`);Decode 侧还有周期性心跳检测机制(`SGLANG_DISAGGREGATION_HEARTBEAT_INTERVAL`)监控 Prefill bootstrap server 的存活状态。
- **超时可调**:提供 `SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT`(建连阶段超时)与 `SGLANG_DISAGGREGATION_WAITING_TIMEOUT`(等待 KV 到达超时)两个环境变量,可在延迟容忍度与故障恢复速度之间权衡。
- **与其他特性的组合**:已支持与投机解码(Speculative Decoding)、结构化输出(Structured Output)、logprob 返回、Decode 侧请求"回退"(retract)等特性组合使用,覆盖面比 vLLM 目前的 Connector 生态更广。
- **异步 KV 传输(Async Transfer)**:较新的特性(`--disaggregation-async-transfer`)允许元数据缓冲区的填充由传输线程异步完成,使 Prefill 计算与 KV 传输能进一步 overlap,减少同步等待。
- **路由层(Router / Model Gateway)**:官方提供 `sglang_router`(现称 SGLang Model Gateway),支持多种路由策略在 Prefill/Decode 实例池之间做负载均衡与容错路由,相当于 vLLM 生态中"Proxy"角色的官方标准实现(vLLM 目前更多依赖社区/示例级 Proxy)。

### 8.8 部署示例(概念性,节选自官方文档)

**单机 1P1D(Mooncake 后端)**:

```bash
# Prefill Server
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --disaggregation-mode prefill \
  --port 30000 \
  --disaggregation-ib-device mlx5_roce0

# Decode Server
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --disaggregation-mode decode \
  --port 30001 --base-gpu-id 1 \
  --disaggregation-ib-device mlx5_roce0

# Router(负责编排请求路径 + 负载均衡)
python -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill http://127.0.0.1:30000 \
  --decode http://127.0.0.1:30001 \
  --host 0.0.0.0 --port 8000
```

**多机 DeepSeek-V3(3P9D 类似生产拓扑,启用 DP Attention + DeepEP MoE 通信后端)**:实际生产部署(如行业报告披露的 DeepSeek-V3/R1 参考架构)通常采用类似 **3 台 Prefill 节点 + 9 台 Decode 节点(每台 8×H100)** 的非对称拓扑——Prefill 侧节点少但吞吐高、充分打满算力;Decode 侧节点多,以更多显存与并行度换取稳定可控的 tail ITL,这正是"独立弹性伸缩"这一 PD 分离核心收益在超大规模 MoE 模型上的典型体现。

---

## 9. 主流框架 PD 分离方案横向对比

结合前文对 vLLM、SGLang 的深入调研,以及公开资料对 NVIDIA Dynamo、TensorRT-LLM、Mooncake 的介绍,可以得到如下对比:

| 维度 | **vLLM** | **SGLang** | **NVIDIA Dynamo** | **TensorRT-LLM** |
|---|---|---|---|---|
| 定位 | 单一推理引擎内置的可插拔 Connector 接口 | 单一推理引擎内置的原生 PD 分离子系统 | **跨引擎**的分离式推理编排/路由层(可编排 vLLM/SGLang/TensorRT-LLM 作为后端) | NVIDIA 官方高性能推理引擎,PD 分离更贴近自有硬件栈 |
| 核心抽象 | `KVConnectorBase_V1`(Scheduler/Worker 双角色) | `KVSender`/`KVReceiver` + `KVTransferConfig`(`kv_producer`/`kv_consumer`) | 跨引擎统一的分离式服务编排 API,底层可复用各引擎的 Connector | 内部 Executor 级别的分离式调度 |
| 主流传输后端 | NIXL、Mooncake、LMCache、P2P-NCCL、MoRIIO、FlexKV | Mooncake、NIXL、Ascend(memfabric_hybrid) | 复用后端引擎自身的传输实现(常见为 NIXL) | NIXL 及 NVIDIA 自有互联(NVLink/InfiniBand) |
| 请求编排/路由 | 依赖社区/自建 Proxy(无官方标准实现) | 官方提供 `sglang_router`(Model Gateway),内置负载均衡与容错路由 | 提供官方 Router/Planner 组件,支持数据中心规模的动态调度 | 依赖 Triton Inference Server 等 NVIDIA 生态组件编排 |
| 异构并行支持 | 支持不同 TP/PP 配置(依赖具体 Connector 实现) | 显式支持异构 TP,并提供 **GPU Staging Buffer** 专门优化该场景 | 依赖后端引擎能力 | 支持,但配置更偏向 NVIDIA 官方参考架构 |
| 容错/重连 | 尚在完善中(依赖具体 Connector,官方标注 experimental) | 已支持传输失败中止、故障重连、心跳检测等较完整机制 | 定位为编排层,容错是其核心卖点之一 | 依赖 Triton/K8s 等外部编排层的容错能力 |
| MoE/大规模场景实践 | 有社区实践,但官方文档未突出 MoE 专项优化 | 与 DeepEP、DP Attention 深度结合,是 DeepSeek 系 MoE 模型的主流分离式部署方案之一 | 面向数据中心规模,理论上可编排任意后端的 MoE 分离部署 | 有 NVIDIA 官方 MoE 优化,但生态相对封闭 |
| 成熟度定位 | **Experimental**(官方明确标注) | 相对更生产化(容错、观测、路由更完整) | 编排层仍在快速演进 | 与硬件绑定较深,成熟度依场景而异 |

**关键结论**:

1. **技术路线趋同**:四者在"生产者/消费者角色划分 + 可插拔传输后端 + 逐层/异步非阻塞传输"这一核心范式上高度一致,印证了 PD 分离已成为大模型推理服务的**事实标准架构**,而非某一家框架的独有创新。
2. **差异集中在工程成熟度与生态定位**:vLLM 更强调接口的通用性和可扩展性,明确把生产级优化交给社区/硬件厂商;SGLang 则把 PD 分离作为核心特性做了更深的生产打磨(容错、路由、异构 TP 优化),并与自身的 DP Attention/MoE 通信栈(DeepEP)结合得更紧密;NVIDIA Dynamo 则跳出单一引擎,定位为可编排多种后端引擎的数据中心级分离式推理"操作系统"。
3. **Mooncake 是事实上的跨框架标准之一**:vLLM、SGLang 均已将 Mooncake 作为官方支持的传输后端,使其成为连接不同推理引擎的"通用 KV 传输层",这与 Mooncake 团队"以 KV Cache 为中心的分离式架构"设计理念一致——即 KV Cache 本身可以作为跨请求、跨实例甚至跨框架复用的一等系统资源。
4. **NIXL 正成为另一条跨框架路线**:同样地,NIXL(NVIDIA/AI-Dynamo 主导)也被 vLLM 与 SGLang 同时采用,两条技术路线(Mooncake vs NIXL)在社区中并存竞争,用户可根据自身硬件(是否有 RDMA 网卡、是否为 NVLink 拓扑等)与生态偏好选择。

---

## 10. 前沿与未来演进方向

1. **KV 传输压缩**:如 SplitZip 提出的无损 KV Cache 压缩方案,专门针对长输入、低带宽(跨集群)场景降低传输体积。
2. **共享内存/CXL 方案**:TraCT 等研究探索利用 CXL 机架级共享内存替代纯网络 RDMA 传输,减少 NIC 队列、主机 DRAM 缓冲、多层传输协议带来的额外跳数与延迟。
3. **双向 KV 传输**:vLLM 社区 RFC(#32733)提出让 Decode 节点也能将 KV 回传给 Prefill 节点,以消除多轮对话、缓存被驱逐后的重复前缀计算。
4. **动态 PD 比例调整**:如 DOPD 等工作探索根据实时负载动态调整 Prefill/Decode 实例数量比例,避免静态配比下的资源浪费或瞬时过载。
5. **单节点内的 PD 微服务化**:如 AMD 基于 MORI-IO 的方案表明,即便只有单台 8-GPU 服务器,也可以通过节点内 PD 拆分获得延迟收益,而不必等到多节点集群规模。
6. **GPU 内部资源级分离**:RAPID-Serve 等研究进一步探索在单块 GPU 内部通过 SM 级资源切分实现 P/D 隔离,减少节点间通信,是比"实例级分离"更细粒度的方向。

---

## 11. 总结

vLLM 的 Disaggregated Prefilling 本质上是把学术界"Prefill/Decode 物理分离以消除阶段间干扰、独立优化 Goodput"的思想,通过一套**清晰分层的 Connector 抽象**(Scheduler 侧负责调度决策与元数据、Worker 侧负责逐层异步显存搬运)工程化落地到 vLLM V1 引擎中。其核心价值在于:

- 将 TTFT 与 ITL 的优化空间解耦,支持针对两阶段的差异化并行策略与硬件选型;
- 通过逐层(per-layer)异步传输 API,把 KV Cache 搬运尽量与计算重叠,降低有效延迟增量;
- 通过工厂模式与可插拔接口,让 NIXL、Mooncake、LMCache、MORI-IO 等针对不同硬件/网络栈优化的第三方实现能够并存、竞争与演进。

但该特性目前仍是 **experimental** 状态,其收益高度依赖具体的网络带宽、模型规模、上下文长度与业务 SLO;在显存受限、互联带宽较低、或单/双卡部署等场景下,传统的合并式 + chunked prefill 调度可能仍是更稳妥的选择。是否引入 PD 分离,本质上是一个"用基础设施复杂度换取延迟可预测性"的工程权衡,而非放之四海皆准的性能优化。

再结合对 SGLang 等主流框架的调研可以看到:**PD 分离的核心范式(生产者/消费者角色划分、可插拔传输后端、异步非阻塞传输)已在业界形成共识**,各框架的竞争焦点正从"要不要做 PD 分离"转向"分离式架构的工程成熟度"——包括容错重连、异构并行下的传输效率(如 GPU Staging Buffer)、请求路由的智能化程度,以及与 MoE 大模型专用并行策略(如 DP Attention)的结合深度。对于生产环境选型,建议优先评估:(1)自身硬件是否具备 RDMA/NVLink 等高速互联条件;(2)业务的 TTFT/TPOT SLO 相对严格程度;(3)所选框架在容错、可观测性等生产化能力上的成熟度,而非仅比较论文中的峰值加速比。

---

### 参考资料

1. vLLM 官方文档 — [Disaggregated Prefilling (experimental)](https://docs.vllm.ai/en/latest/features/disagg_prefill/)
2. vLLM 源码 — `vllm/distributed/kv_transfer/`(GitHub: vllm-project/vllm)
3. DeepWiki — [KV Cache Transfer and Disaggregated Serving](https://deepwiki.com/vllm-project/vllm/9.4-kv-cache-transfer-and-disaggregated-serving)
4. Zhong et al., *DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving*, OSDI 2024 (arXiv:2401.09670)
5. Patel et al., *Splitwise: Efficient Generative LLM Inference Using Phase Splitting*, ISCA 2024
6. Qin et al., *Mooncake: A KV-Cache-centric Disaggregated Architecture for LLM Serving*
7. vLLM GitHub Issue #5557 — *[RFC]: Implement disaggregated prefilling via KV cache transfer*
8. vLLM GitHub Issue #32733 — *[RFC]: [P/D] Prefill compute optimizations with bi-directional KV cache transfers*
9. vLLM Blog — *Next-Level Inference: Why Your Single-Node vLLM Setup Needs Prefill-Decode Disaggregation*(MORI-IO Connector)
10. *SplitZip: Ultra Fast Lossless KV Compression for Disaggregated LLM Serving*(arXiv:2605.01708)
11. *RAPID-Serve: Resource-efficient and Accelerated P/D Intra-GPU Disaggregation*(arXiv:2601.11822)
12. *TraCT: Disaggregated LLM Serving with CXL Shared Memory KV Cache at Rack-Scale*(arXiv:2512.18194)
13. *Prefill-Decode Aggregation or Disaggregation? Unifying Both for Goodput-Optimized LLM Serving*(arXiv:2508.01989)