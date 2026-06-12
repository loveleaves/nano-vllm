"""E2E 验证：Qwen3.5-35B-A3B 减层（3 层 GDN）模型在 8GB 显存上推理。

运行：source .venv/bin/activate && python tests/e2e_qwen35_moe_3l.py
"""
import os
import torch
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def main():
    path = os.path.expanduser("~/model/Qwen3.5-35B-A3B-3L/")
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(
        path,
        enforce_eager=True,
        tensor_parallel_size=1,
        max_num_seqs=4,
        max_model_len=512,
        max_num_batched_tokens=512,
    )

    free, total = torch.cuda.mem_get_info()
    print(f"\n[显存] 已用 {(total - free) / 1e9:.2f} GB / 共 {total / 1e9:.2f} GB")

    sampling_params = SamplingParams(temperature=0.7, max_tokens=32)
    prompts = ["introduce yourself", "1+1="]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in prompts
    ]
    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print(f"\nPrompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")

    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"\n[显存] 推理峰值 {peak:.2f} GB")
    print("\nE2E PASS：模型加载 + prefill + decode 全流程无错误")


if __name__ == "__main__":
    main()
