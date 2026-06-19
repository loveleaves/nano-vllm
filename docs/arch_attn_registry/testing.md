# Attention 后端注册表 / 能力选择对齐 — 测试设计

## 测试矩阵（test_attention_backend.py，全 CPU）

| 分组 | 覆盖点 |
|---|---|
| `TestBackendTriad` | 三件套类型（C 轮，不变） |
| `TestSelector` | is_cuda 旧签名向后兼容：cpu→sdpa、cuda→flash(若装)、env 强制、非法 env 报错 |
| `TestCapabilitySelection`（新） | flash `supports_head_size`(128✓/320✗/100✗) / `supports_dtype`(bf16✓/fp32✗) / `is_available("cpu")`✗；sdpa 全许可；**cuda+head=300→回退 sdpa**；**cuda+fp32→回退 sdpa**，bf16+128→flash |
| `TestRegistry`（新） | `enum.get_class()` 解析；`from_name` 大小写不敏感 + 非法报错；`register_backend` 覆盖后 get_class 解析到替身、`clear_override` 还原 |
| `TestAttentionLayerBinding` | Attention.__init__ 在 CPU 绑定 SDPAImpl；forward 委派形状 |

## 关键校验

- **能力回退**：cuda 上不被 flash 支持的 head_size(300)/dtype(fp32) **自动回退 SDPA** —— 证明
  选择由能力查询驱动而非写死 is_cuda。
- **注册覆盖**：`register_backend(FLASH_ATTN, "...TorchSDPABackend")` 后 `FLASH_ATTN.get_class()`
  返回替身；`clear_override()` 还原默认 —— 证明动态注册生效且可逆。
- **惰性解析**：枚举类路径经 `importlib` 解析，CPU 测试不触发 flash_attn 依赖。

## GPU 端到端

`example.py` 派生脚本：TP=1 默认构建后，反射模型所有 Attention 层的 `impl` 类型集合 =
`{FlashAttentionImpl}`（head_dim=128、bf16、cuda 经新能力选择器选中 flash），greedy 生成与
对齐前同样连贯 —— 证明默认 GPU 路径选择不变、无回归。

## 回归

- 全量套件：**245 passed, 4 skipped**（test_attention_backend 8→17，+9：能力筛选 4 + 注册表 3 +
  既有 2 微调）。其余文件不变。
