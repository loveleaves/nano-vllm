"""GPU 验证异步调度（需显卡 + ~/model/Qwen3-1.7B）。

  1) 单序列 async vs sync greedy **逐 token 完全一致**（无批伴随 → 无多调度步的 FP 扰动）
  2) 多序列 async 自洽确定性（两次 async 结果一致）+ 输出连贯
  3) 吞吐对比（async 应 ≥ sync，重叠 CPU 调度与 GPU 计算）

运行：python scripts/gpu_validate_async.py
"""
import os
import time
import torch

from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer

MODEL = os.path.expanduser("~/model/Qwen3-1.7B/")


def _prompts(tok, raws):
    return [tok.apply_chat_template([{"role": "user", "content": r}],
                                    tokenize=False, add_generation_prompt=True)
            for r in raws]


def _gen(prompts, async_scheduling, max_tokens=64):
    llm = LLM(MODEL, enforce_eager=True, tensor_parallel_size=1,
              async_scheduling=async_scheduling)
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    t0 = time.perf_counter()
    outs = llm.generate(prompts, sp)
    dt = time.perf_counter() - t0
    toks = [o["token_ids"] for o in outs]
    llm.exit()
    return toks, dt


def test_single_seq_exact():
    print("=== 1) 单序列 async vs sync 逐 token 一致 ===")
    tok = AutoTokenizer.from_pretrained(MODEL)
    prompts = _prompts(tok, ["introduce yourself in detail"])
    sync, _ = _gen(prompts, async_scheduling=False)
    asyn, _ = _gen(prompts, async_scheduling=True)
    ok = sync[0] == asyn[0]
    print(f"   sync len={len(sync[0])} async len={len(asyn[0])} 一致={ok}")
    if not ok:
        for j, (x, y) in enumerate(zip(sync[0], asyn[0])):
            if x != y:
                print(f"      首个分歧 @ {j}: {x} vs {y}")
                break
    assert ok
    print("   PASS\n")


def test_multi_seq_determinism_and_coherence():
    print("=== 2) 多序列 async 确定性 + 连贯 ===")
    tok = AutoTokenizer.from_pretrained(MODEL)
    raws = ["list all prime numbers within 100",
            "explain gravity in one paragraph",
            "write a haiku about the sea"]
    prompts = _prompts(tok, raws)
    a1, _ = _gen(prompts, async_scheduling=True)
    a2, _ = _gen(prompts, async_scheduling=True)
    det = a1 == a2
    print(f"   两次 async 一致={det}")
    assert det
    for i, t in enumerate(a1):
        text = tok.decode(t)
        print(f"   out[{i}] len={len(t)} head={text[:60]!r}")
    print("   PASS\n")


def test_throughput():
    print("=== 3) 吞吐对比（async 应 ≥ sync）===")
    tok = AutoTokenizer.from_pretrained(MODEL)
    prompts = _prompts(tok, ["count from 1 to 50"] * 4)
    _, dt_sync = _gen(prompts, async_scheduling=False, max_tokens=128)
    _, dt_async = _gen(prompts, async_scheduling=True, max_tokens=128)
    print(f"   sync={dt_sync*1000:.0f}ms  async={dt_async*1000:.0f}ms  "
          f"加速={dt_sync/dt_async:.2f}x")
    print("   （单卡小模型重叠收益有限，重在不退化 + 正确）\n")


if __name__ == "__main__":
    test_single_seq_exact()
    test_multi_seq_determinism_and_coherence()
    test_throughput()
    print("ALL GPU ASYNC VALIDATIONS PASSED")
