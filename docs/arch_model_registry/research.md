# 模型实例化 / 动态注册表对齐 — V1 现状调研

> 对照基准：本机 `/home/cb/work/vllm/vllm` @ tag `v0.15.1`（V1）。
> phase6 起点：nano-vllm 的模型实例化为**硬编码直连**——`ModelRunner.__init__` 顶层
> `from nanovllm.models.qwen3 import Qwen3ForCausalLM` 后直接 `Qwen3ForCausalLM(hf_config)`，
> 整个引擎只认识一个模型类。本轮补齐 V1 的"动态注册 + 惰性加载"两层。

## V1 三层机制

| 层 | V1 组件 | 职责 | nano 起点 |
|---|---|---|---|
| ① 动态注册 | `model_executor/models/registry.py::_ModelRegistry` + `_VLLM_MODELS` 字典 | 架构名（HF `architectures`）→ `(module, class)`，数百架构，多架构名可复用同一实现，`register_model` 外部覆盖 | ❌ 无注册表 |
| ② 惰性加载 | `_LazyRegisteredModel.load_model_cls()`（`importlib.import_module`） | 表里存字符串，用到时才 import；避免主进程过早初始化 CUDA（forked 子进程报 `Cannot re-initialize CUDA`）、省启动期导入开销 | ❌ 顶层 import |
| ③ 权重映射 | 模型类 `packed_modules_mapping` + 各 Parameter `weight_loader` | HF 权重名 → 融合参数名 + 分片加载 | ✅ 已实现（`utils/loader.py`） |

## V1 关键观察

- `resolve_model_cls(architectures, model_config)`（`registry.py:1022`）：逐个架构查表，支持
  归一化、transformers 后备实现、terratorch、外部覆盖，返回 `(cls, arch)`。
- `_LazyRegisteredModel.inspect_model_cls()`：在**子进程**（`_run_in_subprocess`）探测模型元信息
  `_ModelInfo`，并按源文件 hash 落盘缓存（`VLLM_CACHE_ROOT/modelinfos/*.json`），避免每次启动
  import 全部模型。
- `register_model(model_arch, model_cls)`：`model_cls` 可为 `"<module>:<class>"` 字符串（惰性）
  或 `nn.Module` 子类；已存在则告警并覆盖。

## 与 nano 的差距（本轮范围）

| V1 特性 | 是否对齐 | 说明 |
|---|---|---|
| 架构名 → 模型类 注册表 | ✅ | `_ModelRegistry` + 内置表 `_BUILTIN_MODELS` |
| 字符串惰性导入 `_LazyRegisteredModel` | ✅ | `importlib`，resolve 时才导 |
| `register_model` 外部覆盖（str / 类） | ✅ | 允许覆盖内置实现 |
| `resolve_model_cls` 逐架构解析 + 报错列出支持架构 | ✅ | str/list 入参，首个命中优先 |
| 子进程 `_ModelInfo` 探测 + 磁盘 hash 缓存 | ❌ | 为海量模型类启动开销服务，nano 单模型不需要 |
| transformers / terratorch 后备实现 | ❌ | nano 无通用后备路径 |
| 架构名归一化 / convert_type / runner_type 多路解析 | ❌ | 超出精简实现范围 |
