"""
CPU 执行后端测试（对齐 vLLM V1 CpuPlatform / CPUModelRunner）。

两部分：
  1. Config 层：device='cpu' 的门控（强制 eager、拒绝 mp/swap/TP>1）——纯 CPU、零依赖。
  2. 端到端：在临时目录搭一个微型 Qwen3（config.json + safetensors），用 device='cpu'
     跑通 EngineCore 的 prefill+decode 完整推理，证明**无 GPU、无 flash-attn、无 Triton**
     也能产出 token，且贪心确定（两次构建逐 token 一致）。

CPU 后端不读真实模型/分词器，故整文件标 unit，纳入 `pytest -m unit` 常规回归。
"""
import json
import os

import pytest
import torch

from nanovllm.config import Config


# ─── 1. Config 门控 ──────────────────────────────────────────────────────────


class TestCPUConfigGating:

    def _cfg(self, tmp_path, **kw):
        # 不触发 AutoConfig（无 config.json 时 __post_init__ 静默跳过 hf_config 加载）
        return Config(str(tmp_path), **kw)

    @pytest.mark.unit
    def test_cpu_forces_eager(self, tmp_path):
        cfg = self._cfg(tmp_path, device="cpu", enforce_eager=False)
        assert cfg.enforce_eager is True   # CPU 无 CUDA graph，强制 eager

    @pytest.mark.unit
    def test_cpu_rejects_mp(self, tmp_path):
        with pytest.raises(AssertionError):
            self._cfg(tmp_path, device="cpu", distributed_executor_backend="mp")

    @pytest.mark.unit
    def test_cpu_rejects_tp(self, tmp_path):
        with pytest.raises(AssertionError):
            self._cfg(tmp_path, device="cpu", tensor_parallel_size=2)

    @pytest.mark.unit
    def test_cpu_rejects_swap(self, tmp_path):
        with pytest.raises(AssertionError):
            self._cfg(tmp_path, device="cpu", num_swap_blocks=8)

    @pytest.mark.unit
    def test_invalid_device(self, tmp_path):
        with pytest.raises(AssertionError):
            self._cfg(tmp_path, device="tpu")

    @pytest.mark.unit
    def test_cuda_default_unchanged(self, tmp_path):
        cfg = self._cfg(tmp_path)
        assert cfg.device == "cuda" and cfg.enforce_eager is False


# ─── 2. 端到端：微型 Qwen3 on CPU ─────────────────────────────────────────────

_TINY_CONFIG = {
    "architectures": ["Qwen3ForCausalLM"],
    "model_type": "qwen3",
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 8,
    "vocab_size": 100,
    "max_position_embeddings": 128,
    "rms_norm_eps": 1e-6,
    "hidden_act": "silu",
    "rope_theta": 10000.0,
    "attention_bias": False,
    "tie_word_embeddings": False,
    "torch_dtype": "float32",
}


def _build_tiny_model_dir(path: str) -> None:
    """在 path 下写 config.json + safetensors（HF 命名，经 weight_loader 还原 qkv/gate_up）。"""
    from safetensors.torch import save_file
    from nanovllm.layers.rotary_embedding import get_rope
    from nanovllm.models.qwen3 import Qwen3ForCausalLM

    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(_TINY_CONFIG, f)

    # 用 nano 模型自身的参数当随机权重，再拆成 HF 命名落盘（确保 load 后逐位一致 → 确定）
    cfg = type("C", (), {k: v for k, v in _TINY_CONFIG.items()})()
    get_rope.cache_clear()
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(cfg)

    h, kv = _TINY_CONFIG["num_attention_heads"], _TINY_CONFIG["num_key_value_heads"]
    hd, inter = _TINY_CONFIG["head_dim"], _TINY_CONFIG["intermediate_size"]
    q_size, kv_size = h * hd, kv * hd

    hf_weights: dict[str, torch.Tensor] = {}
    for name, p in model.named_parameters():
        w = p.detach().clone().contiguous()
        if name.endswith("self_attn.qkv_proj.weight"):
            base = name[: -len("qkv_proj.weight")]
            q, k, v = torch.split(w, [q_size, kv_size, kv_size], dim=0)
            hf_weights[base + "q_proj.weight"] = q.contiguous()
            hf_weights[base + "k_proj.weight"] = k.contiguous()
            hf_weights[base + "v_proj.weight"] = v.contiguous()
        elif name.endswith("mlp.gate_up_proj.weight"):
            base = name[: -len("gate_up_proj.weight")]
            gate, up = torch.split(w, [inter, inter], dim=0)
            hf_weights[base + "gate_proj.weight"] = gate.contiguous()
            hf_weights[base + "up_proj.weight"] = up.contiguous()
        else:
            hf_weights[name] = w

    save_file(hf_weights, os.path.join(path, "model.safetensors"))


def _run_once(model_dir: str, prompt: list[int], max_tokens: int) -> list[int]:
    """用 device='cpu' 起 EngineCore，跑完一条贪心请求，返回生成的 token 序列。"""
    from nanovllm.engine.core import EngineCore
    from nanovllm.engine.core_types import EngineCoreRequest
    from nanovllm.sampling_params import SamplingParams

    config = Config(model_dir, device="cpu", max_model_len=128,
                    max_num_batched_tokens=256, max_num_seqs=4, cpu_kvcache_gb=0.05)
    core = EngineCore(config)
    try:
        sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True)
        core.add_request(EngineCoreRequest("req0", list(prompt), sp))
        produced: list[int] = []
        steps = 0
        while core.has_unfinished_requests() and steps < max_tokens + 5:
            out = core.step()
            for o in out.outputs:
                produced.extend(o.new_token_ids)
            steps += 1
        return produced
    finally:
        core.exit()


@pytest.mark.unit
def test_cpu_end_to_end_runs_without_gpu(tmp_path):
    """无 GPU 跑通微型模型：KV cache 成功分配、贪心生成出 max_tokens 个 token。"""
    model_dir = str(tmp_path)
    _build_tiny_model_dir(model_dir)
    out = _run_once(model_dir, prompt=[1, 2, 3, 4, 5], max_tokens=8)
    assert len(out) == 8
    assert all(0 <= t < _TINY_CONFIG["vocab_size"] for t in out)


@pytest.mark.unit
def test_cpu_greedy_deterministic(tmp_path):
    """同权重两次构建 + 贪心生成 → 逐 token 一致（CPU 后端可复现）。"""
    model_dir = str(tmp_path)
    _build_tiny_model_dir(model_dir)
    a = _run_once(model_dir, prompt=[1, 2, 3, 4, 5], max_tokens=8)
    b = _run_once(model_dir, prompt=[1, 2, 3, 4, 5], max_tokens=8)
    assert a == b
