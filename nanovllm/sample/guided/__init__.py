"""
引导 / 结构化解码（对齐 vLLM V1 `v1/structured_output/` + `v1/sample/logits_processor`）。

机制：每个受约束请求持一个 Grammar（FSM）；每步采样前由 GuidedDecodingLogitsProcessor
把"当前状态下不允许的 token"的 logit 置 -inf，采样后推进 FSM 状态。语法完成后只允许 EOS，
请求自然停止。

nano 自包含实现 ChoiceGrammar（输出须等于给定候选之一，trie 实现），零额外依赖、纯 CPU 可测，
演示与 vLLM xgrammar/outlines 后端一致的"逐步 token 掩码"机制。真实文法/正则后端可按同一
Grammar 接口接入。

范围：引导状态为每请求 Python 对象，挂在 Sequence 上、跨步在引擎进程内推进——仅 UniProc
（同进程）路径支持；MP/进程隔离下不可用（grammar 不随 Sequence 序列化）。
"""
from nanovllm.sample.guided.grammar import ChoiceGrammar, Grammar, build_grammar
from nanovllm.sample.guided.processor import GuidedDecodingLogitsProcessor

__all__ = [
    "Grammar",
    "ChoiceGrammar",
    "build_grammar",
    "GuidedDecodingLogitsProcessor",
]
