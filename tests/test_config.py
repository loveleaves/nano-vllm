"""
Config / SamplingParams 单元测试
"""
import pytest
import tempfile

from nanovllm.sampling_params import SamplingParams


class TestSamplingParams:

    @pytest.mark.unit
    def test_default_values(self):
        sp = SamplingParams()
        assert sp.temperature == 1.0
        assert sp.max_tokens == 64
        assert not sp.ignore_eos

    @pytest.mark.unit
    def test_custom_values(self):
        sp = SamplingParams(temperature=0.7, max_tokens=128, ignore_eos=True)
        assert sp.temperature == 0.7
        assert sp.max_tokens == 128
        assert sp.ignore_eos

    @pytest.mark.unit
    def test_zero_temperature_raises(self):
        with pytest.raises(AssertionError):
            SamplingParams(temperature=0.0)

    @pytest.mark.unit
    def test_negative_temperature_raises(self):
        with pytest.raises(AssertionError):
            SamplingParams(temperature=-1.0)

    @pytest.mark.unit
    def test_zero_max_tokens_raises(self):
        with pytest.raises(AssertionError):
            SamplingParams(max_tokens=0)

    @pytest.mark.unit
    def test_negative_max_tokens_raises(self):
        with pytest.raises(AssertionError):
            SamplingParams(max_tokens=-5)

    @pytest.mark.unit
    def test_very_small_positive_temperature_allowed(self):
        # 极小正值应允许
        sp = SamplingParams(temperature=1e-9)
        assert sp.temperature == 1e-9

    @pytest.mark.unit
    def test_large_max_tokens_allowed(self):
        sp = SamplingParams(max_tokens=100000)
        assert sp.max_tokens == 100000


class TestConfig:
    """Config 仅在有效模型目录下可实例化；此处测试参数验证逻辑。"""

    @pytest.mark.unit
    def test_invalid_model_path_raises(self):
        from nanovllm.config import Config
        with pytest.raises(AssertionError):
            Config(model="/nonexistent/path")

    @pytest.mark.unit
    def test_valid_dir_creates_config(self):
        from nanovllm.config import Config
        with tempfile.TemporaryDirectory() as d:
            config = Config(model=d)
            assert config.model == d

    @pytest.mark.unit
    def test_default_field_values(self):
        from nanovllm.config import Config
        with tempfile.TemporaryDirectory() as d:
            config = Config(model=d)
            assert config.max_num_batched_tokens == 16384
            assert config.max_num_seqs == 512
            assert config.gpu_memory_utilization == 0.9
            assert config.tensor_parallel_size == 1
            assert not config.enforce_eager
            assert config.kvcache_block_size == 256

    @pytest.mark.unit
    def test_custom_fields(self):
        from nanovllm.config import Config
        with tempfile.TemporaryDirectory() as d:
            config = Config(model=d, max_num_seqs=128, enforce_eager=True)
            assert config.max_num_seqs == 128
            assert config.enforce_eager

    @pytest.mark.unit
    def test_invalid_block_size_raises(self):
        from nanovllm.config import Config
        with tempfile.TemporaryDirectory() as d:
            with pytest.raises(AssertionError):
                Config(model=d, kvcache_block_size=100)  # 不是 256 的倍数

    @pytest.mark.unit
    def test_invalid_gpu_utilization_raises(self):
        from nanovllm.config import Config
        with tempfile.TemporaryDirectory() as d:
            with pytest.raises(AssertionError):
                Config(model=d, gpu_memory_utilization=0.0)

    @pytest.mark.unit
    def test_invalid_tp_size_raises(self):
        from nanovllm.config import Config
        with tempfile.TemporaryDirectory() as d:
            with pytest.raises(AssertionError):
                Config(model=d, tensor_parallel_size=9)
