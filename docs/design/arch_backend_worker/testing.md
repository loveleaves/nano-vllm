# nano-vllm 架构对齐第三轮（C 多后端 + D Worker 解耦）— 测试文档

## 测试环境

| 项 | 值 |
|---|---|
| GPU | NVIDIA RTX 3060 Ti (8GB)，**单卡**（TP=2 无法真机验证）|
| torch / CUDA | 2.5.1 + cu121 |
| flash_attn | 2.8.3 |
| msgspec | 0.21.1（本轮新增依赖，RPC 序列化）|
| 模型 | Qwen3-1.7B（bf16）|
| 单测命令 | `pytest tests/ -q` |

## 测试用例清单

| 测试文件 | 类型 | 测试点 | 结果 |
|---|---|---|---|
| `tests/test_attention_backend.py` | Unit | backend 三件套类型、`get_kv_cache_shape`、builder 恒等、selector dispatch（cpu/cuda/env 覆盖/非法值）、Attention 层绑定与委派 | ✅ 10 passed |
| `tests/test_rpc.py` | Unit | `ShmTransport.encode/decode` 往返：prefill 全 token / decode 仅 last_token / 多 seq / 无参 / 字节输出 | ✅ 5 passed |
| `tests/test_attention.py` | Unit | 经 `Attention`→SDPA impl 的统一路径（split 后回归）| ✅ 7 passed |
| `tests/test_context.py` / `test_qwen3.py` / `test_embed_head.py` | Unit | `AttentionMetadata`/forward 链/LM head（split 后回归）| ✅ 全过 |
| 其余（scheduler/sequence/block_manager/...）| Unit | 未改动模块回归 | ✅ 全过 |
| **合计** | | | **✅ 164 passed, 4 skipped** |

> 相比第二轮 149，新增 10（backend）+ 5（rpc）= 164。

### 集成 / E2E（Qwen3-1.7B，greedy via temperature=0.01，max_tokens=48）

| 对比项 | 方法 | 结果 |
|---|---|---|
| **C+D(flash) == 原始提交基线 HEAD** | baseline worktree @HEAD（A+B+C+D 全链路）逐 token | ✅ 完全一致 |
| **C+D eager == C+D graph** | 后端抽象 + Worker/RPC 重构后 graph 路径 | ✅ 完全一致 |
| **后端一致性 flash vs torch_sdpa** | `NANOVLLM_ATTN_BACKEND=torch_sdpa` GPU 跑通；首分叉位置 | seq0@18 / seq1 全同 / seq2@42 |

**后端一致性说明**：flash 与 SDPA 在 bf16 下数值微差，仅在生成深处（token 18/42，或全程不分叉）于"top-2 logit 近似并列"处发生 argmax 翻转，属两种 kernel 的正常数值敏感性（vLLM 自身切换后端亦如此），**非正确性回归**。权威对齐路径为 flash，已与原始基线逐 token 一致。

## 验收标准对照（来自 PRD）

| 验收标准 | 测试方法 | 实测 | 达标 |
|---|---|---|---|
| 1. `Attention.forward` 无 flash/SDPA 直接调用，全经 `self.impl` | 代码审查：`layer.py` 仅 `self.impl.forward`；kernel 在 `flash_attn.py`/`torch_sdpa.py` | 通过 | ✅ |
| 2. ≥2 后端 + builder 抽象 + `get_attn_backend()` 正确 dispatch | `test_attention_backend.py`（10 用例）| 通过 | ✅ |
| 3. `ModelRunner` 不含 RPC（loop/read_shm/write_shm/call 迁出）；payload 结构化序列化 | 代码审查：`model_runner.py` 已删；`rpc.py::ShmTransport` 用 msgspec | 通过 | ✅ |
| 4. 全部单测通过 | `pytest tests/` | 164 passed, 4 skipped | ✅ |
| 5. Qwen3 端到端 TP=1 eager+graph 与重构前逐 token 一致 | 对原始 HEAD baseline diff | 完全一致 | ✅ |

## 关键风险验证结果

| 设计风险 | 验证 | 结论 |
|---|---|---|
| store_kvcache 移入 impl 后两后端写 cache 不一致 | 共享 `kv_ops.store_kvcache`；E2E flash 与基线一致、sdpa 跑通 | 一致 |
| CUDA graph 捕获 impl.forward 失败 | backend 在 `Attention.__init__` 绑定；E2E graph == eager | 通过 |
| 后端按机器 cuda 可用性误绑（CPU 测试被绑 flash）| selector 改按**默认设备**判定；CPU 单测走 SDPA | 已修复 |
| builder 增开销 | 两后端 build 恒等返回入参（无拷贝）| 可忽略 |
| msgspec 无法编码 state 元组 | `test_rpc.py` 往返一致；元组全 int/list[int] | 通过 |
| Worker 抽取破坏 TP 时序 | 平移原 barrier/shm 建序到 Worker；TP=1 全绿 | TP=1 通过 |

## 已知局限

1. **TP=2 未真机验证**：本机单卡。Worker/ShmTransport 的多进程路径靠 `test_rpc.py` 序列化往返 + TP=1 等价性间接覆盖；多进程端到端标注为**环境受限未验证**（代码平移自原可用实现，时序未改）。
2. **后端范围**：仅 FlashAttn + TorchSDPA 两个；FlashInfer/Triton-flash 等未实现（接口已预留，加一个三件套文件 + selector 一行即可）。
3. **D 务实边界**：保留同步 `step()` 驱动，未做 async/`AsyncLLM`/OpenAI server；RPC 传输仍为 SharedMemory（仅序列化 pickle→msgspec）。
4. **builder 恒等**：当前两后端共享 `CommonAttentionMetadata`，builder 不做转换；为未来需要重排元数据的后端预留接口。
5. **greedy 验证**：采样器无原生 greedy，E2E 用 `temperature=0.01` 近似确定性 argmax。
