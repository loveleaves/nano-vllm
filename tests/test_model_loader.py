"""
权重加载单元测试（Phase 3：safetensors 加载、packed 权重分发）

GPU 集成测试：pytest tests/test_model_loader.py -m gpu -v  (需设置 NANO_VLLM_MODEL)
"""
import os
import tempfile
import pytest
import torch
import torch.nn as nn


# ─── utils/loader.py 单元测试 ────────────────────────────────────────────────


class TestLoadModel:

    def _make_safetensors(self, tmpdir: str, tensors: dict) -> str:
        from safetensors.torch import save_file
        path = os.path.join(tmpdir, "model.safetensors")
        save_file(tensors, path)
        return path

    @pytest.mark.unit
    def test_default_weight_loader(self):
        from nanovllm.utils.loader import load_model, default_weight_loader

        class SimpleModel(nn.Module):
            packed_modules_mapping = {}

            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.zeros(4, 4))
                self.weight.weight_loader = default_weight_loader

        model = SimpleModel()
        w = torch.randn(4, 4)
        with tempfile.TemporaryDirectory() as tmpdir:
            self._make_safetensors(tmpdir, {"weight": w})
            load_model(model, tmpdir)
        assert torch.allclose(model.weight.data, w)

    @pytest.mark.unit
    def test_packed_qkv_weight_loader(self):
        from nanovllm.utils.loader import load_model
        from nanovllm.layers.linear import QKVParallelLinear

        class FakeModel(nn.Module):
            packed_modules_mapping = {
                "q_proj": ("qkv_proj", "q"),
                "k_proj": ("qkv_proj", "k"),
                "v_proj": ("qkv_proj", "v"),
            }

            def __init__(self):
                super().__init__()
                self.qkv_proj = QKVParallelLinear(8, 4, 2, 1)

        model = FakeModel()
        q_w = torch.ones(2 * 4, 8)
        k_w = torch.ones(1 * 4, 8) * 2
        v_w = torch.ones(1 * 4, 8) * 3

        with tempfile.TemporaryDirectory() as tmpdir:
            from safetensors.torch import save_file
            save_file({"q_proj.weight": q_w, "k_proj.weight": k_w,
                       "v_proj.weight": v_w}, os.path.join(tmpdir, "model.safetensors"))
            load_model(model, tmpdir)

        q_size = 2 * 4
        kv_size = 1 * 4
        assert torch.allclose(model.qkv_proj.weight.data[:q_size], q_w)
        assert torch.allclose(model.qkv_proj.weight.data[q_size: q_size + kv_size], k_w)
        assert torch.allclose(model.qkv_proj.weight.data[q_size + kv_size:], v_w)

    @pytest.mark.unit
    def test_packed_gate_up_weight_loader(self):
        from nanovllm.utils.loader import load_model
        from nanovllm.layers.linear import MergedColumnParallelLinear

        class FakeModel(nn.Module):
            packed_modules_mapping = {
                "gate_proj": ("gate_up_proj", 0),
                "up_proj":   ("gate_up_proj", 1),
            }

            def __init__(self):
                super().__init__()
                self.gate_up_proj = MergedColumnParallelLinear(4, [8, 8])

        model = FakeModel()
        gate_w = torch.ones(8, 4)
        up_w = torch.zeros(8, 4) + 2

        with tempfile.TemporaryDirectory() as tmpdir:
            from safetensors.torch import save_file
            save_file({"gate_proj.weight": gate_w, "up_proj.weight": up_w},
                      os.path.join(tmpdir, "model.safetensors"))
            load_model(model, tmpdir)

        assert torch.allclose(model.gate_up_proj.weight.data[:8], gate_w)
        assert torch.allclose(model.gate_up_proj.weight.data[8:], up_w)

    @pytest.mark.unit
    def test_no_safetensors_raises(self):
        from nanovllm.utils.loader import load_model

        class EmptyModel(nn.Module):
            packed_modules_mapping = {}

        model = EmptyModel()
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(AssertionError):
                load_model(model, tmpdir)

    @pytest.mark.unit
    def test_multiple_shards_loaded(self):
        from nanovllm.utils.loader import load_model, default_weight_loader

        class TwoParamModel(nn.Module):
            packed_modules_mapping = {}

            def __init__(self):
                super().__init__()
                self.a = nn.Parameter(torch.zeros(2, 2))
                self.b = nn.Parameter(torch.zeros(3, 3))
                self.a.weight_loader = default_weight_loader
                self.b.weight_loader = default_weight_loader

            def get_parameter(self, name: str):
                return getattr(self, name)

        model = TwoParamModel()
        wa = torch.randn(2, 2)
        wb = torch.randn(3, 3)

        with tempfile.TemporaryDirectory() as tmpdir:
            from safetensors.torch import save_file
            save_file({"a": wa}, os.path.join(tmpdir, "model-00001-of-00002.safetensors"))
            save_file({"b": wb}, os.path.join(tmpdir, "model-00002-of-00002.safetensors"))
            load_model(model, tmpdir)

        assert torch.allclose(model.a.data, wa)
        assert torch.allclose(model.b.data, wb)


# ─── 前缀剥离 + skip_prefixes（loader 扩展功能）────────────────────────────────

class TestLoaderPrefixStrip:

    def _save(self, tmpdir, tensors):
        from safetensors.torch import save_file
        path = os.path.join(tmpdir, "model.safetensors")
        save_file(tensors, path)

    @pytest.mark.unit
    def test_prefix_stripped_on_load(self):
        """weight_prefix_to_strip 剥离后参数名正确匹配。"""
        from nanovllm.utils.loader import load_model, default_weight_loader

        class PrefixModel(nn.Module):
            weight_prefix_to_strip = "model.language_model."
            weight_skip_prefixes   = ()
            packed_modules_mapping = {}

            def __init__(self):
                super().__init__()
                self.fc = nn.Parameter(torch.zeros(3, 3))
                self.fc.weight_loader = default_weight_loader

        model = PrefixModel()
        w = torch.randn(3, 3)
        with tempfile.TemporaryDirectory() as tmpdir:
            self._save(tmpdir, {"model.language_model.fc": w})
            load_model(model, tmpdir)
        assert torch.allclose(model.fc.data, w)

    @pytest.mark.unit
    def test_skip_prefixes_not_loaded(self):
        """skip_prefixes 命中的权重不写入模型，其余参数正常加载。"""
        from nanovllm.utils.loader import load_model, default_weight_loader

        class SkipModel(nn.Module):
            weight_prefix_to_strip = ""
            weight_skip_prefixes   = ("model.visual.", "mtp.")
            packed_modules_mapping = {}

            def __init__(self):
                super().__init__()
                self.lm = nn.Parameter(torch.zeros(2, 2))
                self.lm.weight_loader = default_weight_loader

        model = SkipModel()
        lm_w = torch.ones(2, 2) * 5
        with tempfile.TemporaryDirectory() as tmpdir:
            self._save(tmpdir, {
                "lm": lm_w,
                "model.visual.patch": torch.randn(4, 4),
                "mtp.head": torch.randn(4, 4),
            })
            load_model(model, tmpdir)
        assert torch.allclose(model.lm.data, lm_w)

    @pytest.mark.unit
    def test_no_prefix_strip_unchanged(self):
        """weight_prefix_to_strip='' 时行为与旧版本完全一致。"""
        from nanovllm.utils.loader import load_model, default_weight_loader

        class PlainModel(nn.Module):
            weight_prefix_to_strip = ""
            weight_skip_prefixes   = ()
            packed_modules_mapping = {}

            def __init__(self):
                super().__init__()
                self.w = nn.Parameter(torch.zeros(2, 2))
                self.w.weight_loader = default_weight_loader

        model = PlainModel()
        data = torch.randn(2, 2)
        with tempfile.TemporaryDirectory() as tmpdir:
            self._save(tmpdir, {"w": data})
            load_model(model, tmpdir)
        assert torch.allclose(model.w.data, data)

    @pytest.mark.unit
    def test_prefix_strip_with_packed_mapping(self):
        """前缀剥离后仍能正确触发 packed_modules_mapping。"""
        from nanovllm.utils.loader import load_model
        from nanovllm.layers.linear import MergedColumnParallelLinear

        class PrefixPackedModel(nn.Module):
            weight_prefix_to_strip = "lm."
            weight_skip_prefixes   = ()
            packed_modules_mapping = {
                "gate_proj": ("gate_up_proj", 0),
                "up_proj":   ("gate_up_proj", 1),
            }

            def __init__(self):
                super().__init__()
                self.gate_up_proj = MergedColumnParallelLinear(4, [4, 4])

        model = PrefixPackedModel()
        gate_w = torch.ones(4, 4)
        up_w   = torch.ones(4, 4) * 2
        with tempfile.TemporaryDirectory() as tmpdir:
            from safetensors.torch import save_file
            save_file({"lm.gate_proj.weight": gate_w,
                       "lm.up_proj.weight": up_w},
                      os.path.join(tmpdir, "model.safetensors"))
            load_model(model, tmpdir)
        assert torch.allclose(model.gate_up_proj.weight.data[:4], gate_w)
        assert torch.allclose(model.gate_up_proj.weight.data[4:], up_w)


# ─── GPU 集成测试（需真实 Qwen3 权重） ─────────────────────────────────────────


MODEL_PATH = os.environ.get("NANO_VLLM_MODEL", "")


@pytest.mark.gpu
@pytest.mark.skipif(not MODEL_PATH or not os.path.isdir(MODEL_PATH),
                    reason="需要设置 NANO_VLLM_MODEL 环境变量指向真实模型路径")
class TestLLMEngineIntegration:

    @pytest.fixture(scope="class")
    def engine(self):
        from nanovllm.engine.llm_engine import LLMEngine
        eng = LLMEngine(MODEL_PATH, enforce_eager=True, max_model_len=512)
        return eng

    def test_kv_cache_allocated(self, engine):
        assert engine.model_runner.config.num_kvcache_blocks > 0

    def test_single_prompt_generation(self, engine):
        from nanovllm.sampling_params import SamplingParams
        result = engine.generate(["Hello, world!"], SamplingParams(max_tokens=10), use_tqdm=False)
        assert len(result) == 1
        assert "text" in result[0]
        assert len(result[0]["token_ids"]) <= 10

    def test_multiple_prompts_ordering(self, engine):
        from nanovllm.sampling_params import SamplingParams
        prompts = ["A", "B", "C"]
        results = engine.generate(prompts, SamplingParams(max_tokens=5), use_tqdm=False)
        assert len(results) == 3

    def test_max_tokens_respected(self, engine):
        from nanovllm.sampling_params import SamplingParams
        sp = SamplingParams(max_tokens=5, ignore_eos=True)
        result = engine.generate(["test"], sp, use_tqdm=False)
        assert len(result[0]["token_ids"]) == 5
