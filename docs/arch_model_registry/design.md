# nano-vllm 模型动态注册表 / 惰性加载对齐 V1 — 详细设计

> 基于 `research.md`。目标：把硬编码的 `Qwen3ForCausalLM(hf_config)` 升级为 V1 风格的
> **架构名注册表 + 惰性导入 + 运行期解析**——从 HF config 的 `architectures` 字段查表，
> 命中后才 `importlib` 导入对应模型类。权重映射层（packed_modules_mapping + weight_loader）
> 本就存在，不在本轮改动范围。

## 范围决策（与 V1 的取舍）

| V1 特性 | 是否对齐 | 说明 |
|---|---|---|
| `_ModelRegistry`（架构名 → 已注册模型） | ✅ | `models: dict[str, _Lazy/_Registered]` |
| `_LazyRegisteredModel`（`"module:class"` 字符串惰性导入） | ✅ | `load_model_cls()` 用时才 `importlib.import_module` |
| `_RegisteredModel`（直接持有类对象） | ✅ | 外部以 `nn.Module` 子类注册时用 |
| `register_model` 注册 / 覆盖（str 或 类） | ✅ | 允许覆盖内置实现 |
| `resolve_model_cls(architectures)` 逐架构解析 | ✅ | str/list 入参，首个命中优先，全未命中报错列支持架构 |
| 子进程 `_ModelInfo` 探测 + 磁盘 hash 缓存 | ❌ | 海量模型类启动开销优化，nano 单模型属过度设计 |
| transformers / terratorch 后备、架构归一化 | ❌ | 超出精简实现定位 |

## Architecture

```
models/
├── registry.py     # _LazyRegisteredModel / _RegisteredModel / _ModelRegistry
│                   # + 内置表 _BUILTIN_MODELS + 全局单例 ModelRegistry
│                   # + 模块级 register_model / resolve_model_cls
└── qwen3.py        # Qwen3ForCausalLM（含 packed_modules_mapping，第③层）

engine/model_runner.py
  __init__:
    architectures = getattr(hf_config, "architectures", None) or ["Qwen3ForCausalLM"]
    model_cls, _arch = resolve_model_cls(architectures)   # 查表 + 惰性导入
    self.model = model_cls(hf_config)
    load_model(self.model, config.model)                  # 第③层权重映射
```

### 注册表数据结构

```python
@dataclass(frozen=True)
class _LazyRegisteredModel:           # 惰性：仅存字符串
    module_name: str
    class_name: str
    def load_model_cls(self): return getattr(import_module(self.module_name), self.class_name)

@dataclass(frozen=True)
class _RegisteredModel:               # 直接持有类
    model_cls: type[nn.Module]
    def load_model_cls(self): return self.model_cls

@dataclass
class _ModelRegistry:
    models: dict[str, _Lazy | _Registered]
    def register_model(arch, cls):    # cls 为 "module:class" → _Lazy；nn.Module 子类 → _Registered
    def resolve_model_cls(archs):     # str|list → (cls, 命中arch)；首个命中惰性导入
    def get_supported_archs(): ...
```

### 内置表（架构名 → "模块:类名"，惰性）

```python
_BUILTIN_MODELS = {"Qwen3ForCausalLM": "nanovllm.models.qwen3:Qwen3ForCausalLM"}
ModelRegistry = _ModelRegistry()   # 全局单例，启动时灌入内置表
```

加新模型只需在 `_BUILTIN_MODELS` 加一行，引擎代码零改动；外部可
`from nanovllm.models.registry import register_model` 注入/覆盖。

## 解析流程

```
resolve_model_cls(architectures):
  archs = [architectures] if str else architectures
  if not archs: raise ValueError                # 空架构
  for arch in archs:
      if arch in models: return models[arch].load_model_cls(), arch   # 首个命中惰性导入
  raise ValueError(支持架构=get_supported_archs())                    # 全未命中
```

`model_runner` 取 `hf_config.architectures`（HF 标准字段），缺失时回退到 `["Qwen3ForCausalLM"]`
以兼容裸 config。

## 校验

- `register_model`：`model_arch` 非 str → `TypeError`；字符串无 `:` → `ValueError`；
  `model_cls` 非 str/非 nn.Module 子类 → `TypeError`。
- 覆盖语义：同名 arch 再注册直接替换（对齐 V1，便于第三方替换内置实现 / 测试替身）。
