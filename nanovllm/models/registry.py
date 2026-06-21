"""
动态模型注册表（对齐 vLLM 0.15.1 的 Model Registry 机制）。

三层设计：
  1. 动态注册（register）：架构名 → (模块路径, 类名) 的映射表。
       HF config 的 `architectures` 字段（如 "Qwen3ForCausalLM"）即查表 key。
       加新模型只需往表里加一行，引擎代码零改动；外部还可调用 register_model 注入/覆盖。
  2. 惰性加载（lazy import）：表里存的是字符串而非类对象，真正用到时才 importlib 导入。
       目的是避免在主进程 import 模型时过早初始化 CUDA（forked 子进程会报
       "Cannot re-initialize CUDA"），同时省去启动期导入全部模型类的开销。
  3. 权重映射（weight loader）：由模型类自身的 packed_modules_mapping + 各 Parameter 的
       weight_loader 承担，见 nanovllm/utils/loader.py，本模块不涉及。

与 vLLM 的差异：nano-vllm 不做子进程元信息探测 / 磁盘缓存（_ModelInfo cache），
保留注册 + 惰性导入 + 架构解析这三件核心能力，契合精简实现的定位。
"""
import importlib
from dataclasses import dataclass, field

from torch import nn


@dataclass(frozen=True)
class _LazyRegisteredModel:
    """以字符串记录的模型，真正使用时才 importlib 导入（惰性加载）。"""

    module_name: str
    class_name: str

    def load_model_cls(self) -> type[nn.Module]:
        module = importlib.import_module(self.module_name)
        return getattr(module, self.class_name)


@dataclass(frozen=True)
class _RegisteredModel:
    """直接持有类对象的模型（外部以 nn.Module 子类注册时使用，无惰性导入）。"""

    model_cls: type[nn.Module]

    def load_model_cls(self) -> type[nn.Module]:
        return self.model_cls


@dataclass
class _ModelRegistry:
    """架构名 → 已注册模型 的映射表，提供注册与解析能力。"""

    # key 为 HF 架构名（architectures 字段元素）
    models: dict[str, "_LazyRegisteredModel | _RegisteredModel"] = field(default_factory=dict)

    def get_supported_archs(self) -> list[str]:
        return sorted(self.models.keys())

    def register_model(self, model_arch: str, model_cls: "type[nn.Module] | str") -> None:
        """
        注册（或覆盖）一个模型架构。

        model_cls 可为：
          - 字符串 "<module>:<class>"：惰性导入，避免过早初始化 CUDA（推荐）。
          - nn.Module 子类：直接持有类对象。
        """
        if not isinstance(model_arch, str):
            raise TypeError(f"model_arch 须为 str，而非 {type(model_arch)}")

        if model_arch in self.models:
            # 与 vLLM 行为一致：允许覆盖，便于外部替换内置实现
            pass

        if isinstance(model_cls, str):
            parts = model_cls.split(":")
            if len(parts) != 2:
                raise ValueError("字符串格式须为 `<module>:<class>`")
            self.models[model_arch] = _LazyRegisteredModel(parts[0], parts[1])
        elif isinstance(model_cls, type) and issubclass(model_cls, nn.Module):
            self.models[model_arch] = _RegisteredModel(model_cls)
        else:
            raise TypeError(f"model_cls 须为 str 或 nn.Module 子类，而非 {type(model_cls)}")

    def resolve_model_cls(self, architectures: "str | list[str]") -> tuple[type[nn.Module], str]:
        """
        从 HF config 的 architectures 解析出模型类（命中即惰性导入）。

        返回 (模型类, 命中的架构名)。逐个尝试，全部未命中则报错并列出支持架构。
        """
        if isinstance(architectures, str):
            architectures = [architectures]
        if not architectures:
            raise ValueError("未指定任何模型架构（architectures 为空）")

        for arch in architectures:
            registered = self.models.get(arch)
            if registered is not None:
                return registered.load_model_cls(), arch

        raise ValueError(
            f"模型架构 {architectures} 暂不支持。已支持架构：{self.get_supported_archs()}"
        )


# ─── 内置模型表（架构名 → "模块:类名"，惰性导入） ──────────────────────────────
_BUILTIN_MODELS: dict[str, str] = {
    "Qwen3ForCausalLM": "nanovllm.models.qwen3:Qwen3ForCausalLM",
    # Qwen3.5（VLM 包装）：nano 仅取文本主干（dense 混合线性注意力）
    "Qwen3_5ForConditionalGeneration": "nanovllm.models.qwen35:Qwen35ForCausalLM",
}

ModelRegistry = _ModelRegistry()
for _arch, _path in _BUILTIN_MODELS.items():
    ModelRegistry.register_model(_arch, _path)


# ─── 模块级便捷函数 ──────────────────────────────────────────────────────────
def register_model(model_arch: str, model_cls: "type[nn.Module] | str") -> None:
    """向全局注册表注入/覆盖模型，供外部扩展。"""
    ModelRegistry.register_model(model_arch, model_cls)


def resolve_model_cls(architectures: "str | list[str]") -> tuple[type[nn.Module], str]:
    """从全局注册表解析模型类（惰性导入）。"""
    return ModelRegistry.resolve_model_cls(architectures)
