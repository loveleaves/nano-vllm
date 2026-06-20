"""CPU 推理示例：无 GPU（也无需 flash-attn / Triton）也能跑通完整推理。

与 example.py 的唯一区别是 `device="cpu"`：注意力自动退回 SDPA、KV 写入退回 naive
scatter、禁用 CUDA graph、KV cache 内存由 cpu_kvcache_gb 显式预留、模型以 float32 运行。

  python example_cpu.py            # 默认 ~/model/Qwen3-1.7B/
  NANO_VLLM_MODEL=/path python example_cpu.py
"""
import os

from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def main():
    path = os.environ.get("NANO_VLLM_MODEL", os.path.expanduser("~/model/Qwen3-1.7B/"))
    tokenizer = AutoTokenizer.from_pretrained(path)

    # device="cpu" 触发 CPUModelRunner；cpu_kvcache_gb 为 KV cache 预留内存（GB）
    llm = LLM(path, device="cpu", cpu_kvcache_gb=2.0, max_model_len=2048)

    sampling_params = SamplingParams(temperature=0.0, max_tokens=64)
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True,
        )
        for prompt in ["introduce yourself", "list all prime numbers within 50"]
    ]
    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
