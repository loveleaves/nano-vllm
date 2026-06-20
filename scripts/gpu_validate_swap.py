"""GPU 验证 swap 抢占（需显卡 + ~/model/Qwen3-1.7B）。

  1) 张量往返：填已知值 → swap_out(D2H) → 清零 GPU → swap_in(H2D) → 校验还原
  2) 端到端：钳制 num_kvcache_blocks 强制抢占，swap vs recompute 的 greedy 输出应一致

运行：python scripts/gpu_validate_swap.py
"""
import os
import torch

from nanovllm import LLM, SamplingParams
from nanovllm.engine import model_runner as mr_mod
from transformers import AutoTokenizer

MODEL = os.path.expanduser("~/model/Qwen3-1.7B/")


def test_tensor_roundtrip():
    print("=== 1) swap_out/swap_in 张量往返 ===")
    llm = LLM(MODEL, enforce_eager=True, tensor_parallel_size=1, num_swap_blocks=8)
    runner = llm.engine_core.executor.worker.model_runner
    assert runner.cpu_kv_cache is not None, "num_swap_blocks>0 应分配 CPU swap 区"

    gpu_blocks = [3, 5, 7]
    swap_slots = [0, 1, 2]
    # 给目标 GPU 块写入可识别值
    for i, g in enumerate(gpu_blocks):
        runner.kv_cache[:, :, g].fill_(float(i + 1))
    ref = runner.kv_cache[:, :, gpu_blocks].clone()

    mapping = list(zip(gpu_blocks, swap_slots))
    runner.swap_out(mapping)                       # D2H
    runner.kv_cache[:, :, gpu_blocks] = 0          # 抹掉 GPU（index 赋值，非 copy）
    assert runner.kv_cache[:, :, gpu_blocks].abs().sum().item() == 0
    runner.swap_in(mapping)                        # H2D 还原

    restored = runner.kv_cache[:, :, gpu_blocks]
    ok = torch.equal(restored, ref)
    print(f"   还原一致: {ok}")
    assert ok
    llm.exit()
    print("   PASS\n")


def _run(prompts, num_swap_blocks, clamp_blocks):
    """钳制 KV 块数构造引擎并跑 greedy，返回每个 prompt 的 token 序列。"""
    orig = mr_mod.ModelRunner.allocate_kv_cache

    def clamped(self):
        orig(self)
        if self.config.num_kvcache_blocks > clamp_blocks:
            self.config.num_kvcache_blocks = clamp_blocks
            # 重切 KV 张量到钳制后的块数
            import torch as _t
            from nanovllm.layers.attention import Attention
            shape = self.kv_cache_spec.kv_cache_shape(clamp_blocks)[1:]
            num_layers = self.config.hf_config.num_hidden_layers
            self.kv_cache = _t.empty(2, num_layers, *shape, device="cuda",
                                     dtype=self.kv_cache.dtype)
            lid = 0
            for m in self.model.modules():
                if isinstance(m, Attention):
                    m.k_cache = self.kv_cache[0, lid]
                    m.v_cache = self.kv_cache[1, lid]
                    lid += 1

    mr_mod.ModelRunner.allocate_kv_cache = clamped
    try:
        llm = LLM(MODEL, enforce_eager=True, tensor_parallel_size=1,
                  num_swap_blocks=num_swap_blocks, max_model_len=1024)
        sp = SamplingParams(temperature=0.0, max_tokens=64)
        outs = llm.generate(prompts, sp)
        toks = [o["token_ids"] for o in outs]
        llm.exit()
    finally:
        mr_mod.ModelRunner.allocate_kv_cache = orig
    return toks


def test_end_to_end_equivalence():
    print("=== 2) swap vs recompute greedy 等价（强制抢占）===")
    tok = AutoTokenizer.from_pretrained(MODEL)
    raw = ["introduce yourself", "list all prime numbers within 100",
           "explain gravity in one paragraph"]
    prompts = [tok.apply_chat_template([{"role": "user", "content": p}],
                                       tokenize=False, add_generation_prompt=True)
               for p in raw]

    # 钳到很少的块，多条并发 → 必然触发抢占
    recompute = _run(prompts, num_swap_blocks=0, clamp_blocks=6)
    swap = _run(prompts, num_swap_blocks=32, clamp_blocks=6)

    all_ok = True
    for i, (a, b) in enumerate(zip(recompute, swap)):
        ok = a == b
        all_ok &= ok
        print(f"   prompt[{i}] len recompute={len(a)} swap={len(b)} 一致={ok}")
        if not ok:
            # 找首个分歧位
            for j, (x, y) in enumerate(zip(a, b)):
                if x != y:
                    print(f"      首个分歧 @ {j}: {x} vs {y}")
                    break
    print(f"   {'PASS' if all_ok else 'FAIL'}\n")
    assert all_ok


if __name__ == "__main__":
    test_tensor_roundtrip()
    test_end_to_end_equivalence()
    print("ALL GPU SWAP VALIDATIONS PASSED")
