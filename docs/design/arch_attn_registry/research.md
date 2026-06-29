# Attention 后端注册表 / 能力选择对齐 — V1 现状调研

> 对照基准：本机 `/home/cb/work/vllm/vllm` @ tag `v0.15.1`（V1）。
> 承接 C 轮（多后端 AttentionBackend 三件套）遗留项：`docs/nano_vs_vllm-架构对比` §5 标注的
> 两处差距——① 无 `registry.py` 动态注册 + 按 head_dim/dtype/platform 自动选最优后端；
> ② builder 恒等（V1 各后端 builder 做实质重排）。本轮补齐 ①。

## 背景：什么是注意力后端注册表 + 能力选择

**问题**：注意力有多种实现（后端）——FlashAttention（最快，但要求特定 head_size/dtype/有 CUDA +
装了 flash-attn）、PyTorch SDPA（兜底，任意平台/精度）等。它们性能与适用范围各异，硬编码"二选一"
不可扩展、不可按环境自适应。

**核心思想——注册表 + 能力驱动选择**：
- **注册表**：用枚举把"后端名 → 实现类路径"登记成表，`get_class()` **惰性导入**（用时才 import，
  避免无谓依赖）；`register_backend` 可运行时覆盖（接第三方后端 / 测试替身）。
- **能力查询**：每个后端声明自己支持的 `head_size / dtype / 平台`（`is_available` /
  `supports_head_size` / `supports_dtype`）。
- **自动选择**：`get_attn_backend(head_size, dtype, device)` 按优先级 [flash, sdpa] 选出第一个
  "可用且支持当前形状/精度"的后端；环境变量可显式强制。

**作用 / 收益**：按硬件/模型自适应选最优后端，新增后端只需注册一行，CPU/无 flash-attn 环境自动
回退 SDPA（保证可测）。

## V1 组件

| 文件 | 职责 | nano 对应 |
|---|---|---|
| `v1/attention/backends/registry.py` | `AttentionBackendEnum`（名→类路径）、`register_backend` 运行时覆盖、`enum.get_class()` 惰性解析 | `layers/attention/registry.py`（FLASH_ATTN/TORCH_SDPA 两枚） |
| `v1/attention/selector.py::get_attn_backend` | 按 head_size/dtype/kv_cache_dtype/block_size/平台/计算能力筛选后端 | `layers/attention/selector.py`（按 head_size/dtype/平台） |
| `v1/attention/backend.py::AttentionBackend` | `supports_head_size` / `supports_dtype` / `supports_compute_capability` / `get_supported_head_sizes` … 能力查询 | `AttentionBackend` 基类能力 classmethod（精简子集） |

## 关键观察

1. **枚举 + 类路径字符串**：后端在枚举里以全限定路径登记，`get_class()` 惰性 import 解析。
   好处：① 不在导入期触发未用后端的依赖（如 flashinfer/flash_attn）；② `register_backend`
   可运行时覆盖某枚举的实现（第三方后端、测试替身）。
2. **能力驱动选择**：`get_attn_backend` 不是写死 if/else，而是逐后端查询
   `supports_head_size(head)/supports_dtype(dtype)/supports_compute_capability(cap)`，
   选首个全满足者；不满足则回退（如 head_size 超限或 fp32 → 退非 flash 后端）。
3. **builder 的后端专属重排**：V1 各后端 `AttentionMetadataBuilder.build` 会做 kernel 专属的
   元数据变换（如 reorder、cu_seqlens 重算）。

## nano 对齐前状态（C 轮）

`selector.py` 用写死的 `_BACKENDS` dict + `is_cuda and HAS_FLASH_ATTN` 二选一；无枚举、无注册
覆盖、无 head_size/dtype 能力筛选。`AttentionBackend` 基类只有 get_name/impl/builder/kv_shape。

## nano 取舍（不引入）

- **builder 仍恒等**（差距 ②不补）：nano 两后端（flash varlen / SDPA 兜底）消费**同一**
  `CommonAttentionMetadata`，无需后端专属重排——恒等 build 是正确而非偷懒（见 design.md）。
- 不引入 kv_cache_dtype/block_size/compute_capability/MLA/sparse 等 V1 的多维能力轴与几十个后端。
