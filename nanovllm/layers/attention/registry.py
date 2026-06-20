"""
注意力后端注册表（对齐 vLLM V1 `v1/attention/backends/registry.py`）。

把"后端名 → 实现类路径"的映射固化成枚举，并支持运行时 `register_backend` 覆盖（第三方/
自定义后端，或测试替身）。枚举值为类的全限定路径，`get_class()` 按需惰性 import 解析——
避免在不使用 flash_attn 的环境（如 CPU 单测）于导入期触发其依赖。

nano 仅 FLASH_ATTN / TORCH_SDPA 两枚；接口为未来后端（MLA、FlexAttention…）预留。
"""
import importlib
from enum import Enum


def resolve_obj_by_qualname(qualname: str):
    """按全限定名惰性解析对象（module.path.ClassName → 类）。"""
    module_name, obj_name = qualname.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, obj_name)


# 运行时覆盖表：register_backend 写入，get_path/get_class 优先读取
_ATTN_OVERRIDES: dict["AttentionBackendEnum", str] = {}


class AttentionBackendEnum(Enum):
    """所有支持的注意力后端（枚举值为默认实现类路径，可被 register_backend 覆盖）。"""

    FLASH_ATTN = "nanovllm.layers.attention.flash_attn.FlashAttentionBackend"
    TORCH_SDPA = "nanovllm.layers.attention.torch_sdpa.TorchSDPABackend"

    def get_path(self) -> str:
        """该后端的实现类路径（尊重运行时覆盖）。"""
        return _ATTN_OVERRIDES.get(self, self.value)

    def get_class(self):
        """解析并返回后端实现类（尊重覆盖，惰性 import）。"""
        return resolve_obj_by_qualname(self.get_path())

    def is_overridden(self) -> bool:
        return self in _ATTN_OVERRIDES

    def clear_override(self) -> None:
        _ATTN_OVERRIDES.pop(self, None)

    @classmethod
    def from_name(cls, name: str) -> "AttentionBackendEnum":
        """按后端名（如 "flash_attn" / "torch_sdpa"，大小写不敏感）取枚举成员。"""
        try:
            return cls[name.upper()]
        except KeyError:
            valid = ", ".join(m.name.lower() for m in cls)
            raise ValueError(
                f"未知注意力后端 {name!r}，可选：{valid}") from None


def register_backend(backend: AttentionBackendEnum, class_path: str | None = None):
    """注册 / 覆盖某后端的实现类。

    直接调用：register_backend(AttentionBackendEnum.FLASH_ATTN, "pkg.mod.Cls")
    作装饰器：@register_backend(AttentionBackendEnum.FLASH_ATTN)
              class MyFlashAttn: ...
    """
    if class_path is not None:
        _ATTN_OVERRIDES[backend] = class_path
        return None

    def _decorator(cls):
        _ATTN_OVERRIDES[backend] = f"{cls.__module__}.{cls.__qualname__}"
        return cls

    return _decorator
