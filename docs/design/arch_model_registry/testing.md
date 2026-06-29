# 模型动态注册表 — 测试

> 单测：`tests/test_model_registry.py`（13 例，`-m unit`）。
> 权重映射层（第③层）测试见既有 `tests/test_model_loader.py`，本轮未改动。

## 运行

```bash
source .venv/bin/activate
pytest tests/test_model_registry.py -m unit -v
# 全量回归
pytest -m unit -q          # 287 passed
```

## 覆盖矩阵

| 用例 | 验证点 |
|---|---|
| `test_builtin_qwen3_registered` | 内置表已注册 `Qwen3ForCausalLM` |
| `test_string_registration_is_lazy` | 字符串注册存为 `_LazyRegisteredModel(module, class)`，**不**触发导入 |
| `test_lazy_import_only_on_resolve` | 模块仅在 `resolve_model_cls` 时被 import（注册阶段不导入） |
| `test_register_class_directly` | `nn.Module` 子类直接注册 → `_RegisteredModel` |
| `test_register_overwrites` | 同名 arch 再注册覆盖前者 |
| `test_bad_string_format_raises` | 无 `:` 的字符串 → `ValueError` |
| `test_bad_model_cls_type_raises` | `model_cls` 非 str/非 Module → `TypeError` |
| `test_bad_arch_type_raises` | `model_arch` 非 str → `TypeError` |
| `test_resolve_accepts_str_and_list` | str 与 list 入参均可解析 |
| `test_resolve_first_match_wins` | `["Unknown", "Known"]` 命中第二个 |
| `test_resolve_empty_raises` | 空架构列表 → `ValueError` |
| `test_resolve_unsupported_raises_with_list` | 不支持架构 → `ValueError`，消息含架构名 |
| `test_module_level_helpers` | 模块级 `register_model` / `resolve_model_cls` 走全局单例 |

## 结果

- `tests/test_model_registry.py`：13 passed
- 全量 `-m unit`：**287 passed**，4 deselected，无回归。
