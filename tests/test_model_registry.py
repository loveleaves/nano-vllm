"""
动态模型注册表单元测试（注册 + 惰性导入 + 架构解析）。

运行：pytest tests/test_model_registry.py -m unit -v
"""
import sys
import types

import pytest
import torch.nn as nn

from nanovllm.models.registry import (
    ModelRegistry,
    _LazyRegisteredModel,
    _ModelRegistry,
    register_model,
    resolve_model_cls,
)


class TestLazyImport:

    @pytest.mark.unit
    def test_builtin_qwen3_registered(self):
        assert "Qwen3ForCausalLM" in ModelRegistry.get_supported_archs()

    @pytest.mark.unit
    def test_string_registration_is_lazy(self):
        """字符串注册仅存模块/类名，不触发导入。"""
        reg = _ModelRegistry()
        reg.register_model("FooForCausalLM", "nanovllm.models.qwen3:Qwen3ForCausalLM")
        entry = reg.models["FooForCausalLM"]
        assert isinstance(entry, _LazyRegisteredModel)
        assert entry.module_name == "nanovllm.models.qwen3"
        assert entry.class_name == "Qwen3ForCausalLM"

    @pytest.mark.unit
    def test_lazy_import_only_on_resolve(self):
        """模块在 resolve 前不应被导入，resolve 后才出现在 sys.modules。"""
        mod_name = "nanovllm._fake_lazy_model"
        sys.modules.pop(mod_name, None)

        fake = types.ModuleType(mod_name)

        class FakeModel(nn.Module):
            pass

        fake.FakeModel = FakeModel
        # 预置到 sys.modules，模拟"已可导入但尚未被 registry 加载"
        reg = _ModelRegistry()
        reg.register_model("FakeArch", f"{mod_name}:FakeModel")
        # 注册阶段不导入
        sys.modules.pop(mod_name, None)
        sys.modules[mod_name] = fake  # importlib.import_module 命中缓存即可

        cls, arch = reg.resolve_model_cls("FakeArch")
        assert cls is FakeModel
        assert arch == "FakeArch"


class TestRegisterModel:

    @pytest.mark.unit
    def test_register_class_directly(self):
        reg = _ModelRegistry()

        class MyModel(nn.Module):
            pass

        reg.register_model("MyArch", MyModel)
        cls, arch = reg.resolve_model_cls("MyArch")
        assert cls is MyModel and arch == "MyArch"

    @pytest.mark.unit
    def test_register_overwrites(self):
        reg = _ModelRegistry()

        class A(nn.Module):
            pass

        class B(nn.Module):
            pass

        reg.register_model("Arch", A)
        reg.register_model("Arch", B)
        cls, _ = reg.resolve_model_cls("Arch")
        assert cls is B

    @pytest.mark.unit
    def test_bad_string_format_raises(self):
        reg = _ModelRegistry()
        with pytest.raises(ValueError):
            reg.register_model("Arch", "no_colon_here")

    @pytest.mark.unit
    def test_bad_model_cls_type_raises(self):
        reg = _ModelRegistry()
        with pytest.raises(TypeError):
            reg.register_model("Arch", 123)

    @pytest.mark.unit
    def test_bad_arch_type_raises(self):
        reg = _ModelRegistry()

        class M(nn.Module):
            pass

        with pytest.raises(TypeError):
            reg.register_model(123, M)


class TestResolve:

    @pytest.mark.unit
    def test_resolve_accepts_str_and_list(self):
        reg = _ModelRegistry()

        class M(nn.Module):
            pass

        reg.register_model("Arch", M)
        assert reg.resolve_model_cls("Arch")[0] is M
        assert reg.resolve_model_cls(["Arch"])[0] is M

    @pytest.mark.unit
    def test_resolve_first_match_wins(self):
        reg = _ModelRegistry()

        class M(nn.Module):
            pass

        reg.register_model("Known", M)
        cls, arch = reg.resolve_model_cls(["Unknown", "Known"])
        assert cls is M and arch == "Known"

    @pytest.mark.unit
    def test_resolve_empty_raises(self):
        with pytest.raises(ValueError):
            ModelRegistry.resolve_model_cls([])

    @pytest.mark.unit
    def test_resolve_unsupported_raises_with_list(self):
        with pytest.raises(ValueError) as exc:
            ModelRegistry.resolve_model_cls("DefinitelyNotAModel")
        assert "DefinitelyNotAModel" in str(exc.value)

    @pytest.mark.unit
    def test_module_level_helpers(self):
        class M(nn.Module):
            pass

        register_model("ModuleLevelArch", M)
        cls, _ = resolve_model_cls("ModuleLevelArch")
        assert cls is M
