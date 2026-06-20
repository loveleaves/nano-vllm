"""OpenAI server 命令行参数（对齐 vLLM `entrypoints/openai/cli_args.py` 的精简子集）。"""
import argparse


def make_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="nano-vllm OpenAI 兼容 API 服务")
    parser.add_argument("--model", required=True, help="模型权重目录路径")
    parser.add_argument("--served-model-name", default=None,
                        help="对外暴露的模型名（默认取 --model）")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)

    # 引擎配置（透传给 Config，与 LLM/AsyncLLM 同名）
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--enforce-eager", action="store_true")
    return parser


def engine_kwargs_from_args(args) -> dict:
    """从已解析的 args 提取 Config 字段。"""
    return {
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "tensor_parallel_size": args.tensor_parallel_size,
        "enforce_eager": args.enforce_eager,
    }
