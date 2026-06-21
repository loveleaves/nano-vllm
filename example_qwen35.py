"""Qwen3.5 dense（混合线性注意力）推理示例。

Qwen3.5 = GatedDeltaNet 线性注意力层 + 全注意力层（每 4 层一个）+ dense MLP 的混合架构。
线性注意力层维护**每序列递归状态**（conv + recurrent），由引擎按 max_num_seqs 分配状态池，
故 max_num_seqs 越大占用越高——示例用较小值。CPU 后端下统一 fp32、强制 eager。

  python example_qwen35.py
  NANO_VLLM_MODEL=/path/to/Qwen3.5-2B python example_qwen35.py
"""
import os

from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def main():
    path = os.environ.get("NANO_VLLM_MODEL", os.path.expanduser("~/model/Qwen3.5-2B/"))
    tokenizer = AutoTokenizer.from_pretrained(path)

    # max_num_seqs 小 → 递归状态池小；cpu_kvcache_gb 仅供全注意力层的分页 KV cache
    llm = LLM(path, device="cpu", cpu_kvcache_gb=1.0, max_model_len=2048, max_num_seqs=4)

    sampling_params = SamplingParams(temperature=0.0, max_tokens=48)
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True,
        )
        for prompt in ["introduce yourself", "What is the capital of France?"]
    ]
    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
