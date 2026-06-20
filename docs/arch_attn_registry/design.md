# nano-vllm Attention 后端注册表 / 能力选择对齐 V1 — 详细设计

> 基于 `research.md`。目标：把 C 轮写死的后端二选一升级为 V1 风格的**枚举注册表 +
> 能力驱动选择**——`AttentionBackendEnum` + `register_backend` 覆盖 + `get_attn_backend`
> 按 (平台, head_size, dtype) 筛选。

## 范围决策（与 V1 的取舍）

| V1 特性 | 是否对齐 | 说明 |
|---|---|---|
| `AttentionBackendEnum`（名→类路径）+ 惰性 `get_class()` | ✅ | FLASH_ATTN / TORCH_SDPA 两枚 |
| `register_backend` 运行时覆盖（含装饰器形式） | ✅ | 第三方后端 / 测试替身 |
| 能力查询 `supports_head_size` / `supports_dtype` / `is_available` | ✅ | 基类默认 + 子类覆盖 |
| `get_attn_backend` 按能力筛选 + 回退 | ✅ | 优先级 [flash, sdpa] |
| kv_cache_dtype / block_size / compute_capability / MLA / sparse 多维能力 | ❌ | nano 单一标准 attention |
| builder 后端专属重排 | ❌ | 两后端共享元数据，恒等 build（见下） |

## Architecture

```
layers/attention/
├── registry.py     # AttentionBackendEnum + register_backend + resolve_obj_by_qualname + _ATTN_OVERRIDES
├── selector.py     # get_attn_backend(head_size, dtype, device_type[, is_cuda])：能力筛选 + 回退
├── backend.py      # AttentionBackend 基类 + 能力 classmethod（is_available/supports_head_size/supports_dtype）
├── flash_attn.py   # FlashAttentionBackend：supported_dtypes=[fp16,bf16]；is_available=cuda∧HAS_FLASH；head≤256∧%8
├── torch_sdpa.py   # TorchSDPABackend：任意平台/head_size，dtypes=[fp16,bf16,fp32]
└── layer.py        # Attention.__init__ 传 head_dim/默认dtype/默认device 给 get_attn_backend
```

### 选择流程

```
get_attn_backend(head_size, dtype, device_type):
  1) env NANOVLLM_ATTN_BACKEND → AttentionBackendEnum.from_name → get_class()   # 显式强制，绕过能力检查
  2) for member in [FLASH_ATTN, TORCH_SDPA]:
        backend = member.get_class()            # 惰性 import，尊重 register_backend 覆盖
        if backend.is_available(device_type)
           and (head_size is None or supports_head_size)
           and (dtype is None or supports_dtype): return backend
  3) 都不满足 → ValueError
```

`Attention.__init__` 以**当前默认设备/dtype**（即模型构建环境）+ head_dim 调用，后端在 __init__
绑定、forward 期不再分发——CUDA graph 捕获稳定（沿用 C 轮约束）。

### 关键设计点

- **惰性解析**：枚举值是类路径字符串，`get_class()` 才 `importlib.import_module`。CPU 单测/无
  flash 环境不会在导入期触发 `import flash_attn`（flash_attn.py 本身也已 try/except 保护）。
- **覆盖优先于默认**：`_ATTN_OVERRIDES` dict 由 `register_backend` 写入，`get_path/get_class`
  优先读取；`clear_override` 还原。支持装饰器与直接两种调用。
- **能力子类覆盖**：FlashAttn 覆盖 `is_available`（cuda ∧ 已安装）与 `supports_head_size`
  （≤256 且 8 的倍数）；SDPA 全许可（兜底）。故 cuda 上 head=300 或 fp32 会**自动回退 SDPA**。
- **向后兼容**：保留 `get_attn_backend(is_cuda=...)` 旧签名（device_type 未给时由 is_cuda 推断），
  C 轮测试与调用零改动。

### 为何 builder 仍恒等（差距 ② 不补）

nano 只有两个后端：FlashAttn（统一 varlen，直接吃 cu_seqlens_q/cu_seqlens_k/block_table）与
TorchSDPA（CPU/无 flash 兜底，从同一组字段逐序列重建）。二者消费**同一** `CommonAttentionMetadata`，
无 kernel 专属布局差异，故 `build` 恒等是**正确**的（V1 的重排源于其后端各有专属 metadata 类）。
强行造重排只会引入无意义复杂度。保留接口，未来新增异构后端时再落实。
