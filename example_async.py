"""AsyncLLM 端到端示例：异步流式生成。

与 example.py（同步 LLM.generate）对照，本脚本用 AsyncLLM 的 async generator
逐 token 流式拿到增量文本（delta_text），并让多条请求并发跑在同一事件循环上，
共享同一个 EngineCore（后台 handler 把它们混排进连续批）。

运行：python example_async.py
"""
import asyncio
import os

from transformers import AutoTokenizer

from nanovllm import AsyncLLM, SamplingParams


async def stream_one(llm: AsyncLLM, tag: str, prompt: str, sp: SamplingParams):
    """消费单条请求的流式输出，逐增量打印。"""
    print(f"\n[{tag}] >>> 开始")
    text = ""
    async for ro in llm.generate(prompt, sp, request_id=tag):
        text = ro.text
        # delta_text 为本步新增文本，适合 streaming 落地（这里直接打印）
        print(f"[{tag}] +{ro.delta_text!r}", flush=True)
        if ro.finished:
            print(f"[{tag}] <<< 结束 (finish_reason={ro.finish_reason})")
    return text


async def main():
    path = os.path.expanduser("~/model/Qwen3-1.7B/")
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = AsyncLLM(path, enforce_eager=True, tensor_parallel_size=1)

    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in prompts
    ]

    # 多条请求并发：共享同一 EngineCore，后台 handler 将其混排进连续批
    results = await asyncio.gather(*[
        stream_one(llm, f"req{i}", p, sampling_params)
        for i, p in enumerate(prompts)
    ])

    print("\n==== 最终结果 ====")
    for i, (prompt, text) in enumerate(zip(prompts, results)):
        print(f"\n[req{i}] Prompt: {prompt!r}")
        print(f"[req{i}] Completion: {text!r}")

    llm.exit()


if __name__ == "__main__":
    asyncio.run(main())
