"""
采样元数据（对齐 vLLM V1 `v1/sample/metadata.py::SamplingMetadata`）。

把"本批每个序列的采样配置"固化成一个结构体，按行（与 InputBatch 行序一致）批量持有
temperature / top-p / top-k / 各类惩罚 / 历史 token，交给 Sampler 一次性向量化处理。

nano 取最小子集：不含 generators(逐请求种子)、bad_words、allowed_token_ids、min_p、
logits_processors、spec_token_ids（见 docs/arch_sampler/design.md 的范围决策）。
"""
from dataclasses import dataclass

import torch


@dataclass
class SamplingMetadata:
    """一批序列的采样配置（字段按行对齐，行序 = InputBatch 行序）。

    temperature        — [n] 温度；< eps 视为 greedy
    all_greedy         — 整批均为 greedy（temperature < eps）
    all_random         — 整批均为随机采样（无 greedy 行）
    top_p / top_k      — [n] 或 None（None 表示整批未启用，跳过该步）
    no_penalties       — 整批无惩罚（freq/pres==0 且 rep==1），可整段跳过
    prompt_token_ids   — list[list[int]]，惩罚用（repetition 同时看 prompt+output）；no_penalties 时 None
    output_token_ids   — list[list[int]]，已生成 token（频率/存在/重复惩罚）
    frequency/presence/repetition_penalties — [n]
    max_num_logprobs   — 需返回的 top-logprobs 个数；None 表示不要 logprobs
    """
    temperature: torch.Tensor
    all_greedy: bool
    all_random: bool

    top_p: torch.Tensor | None = None
    top_k: torch.Tensor | None = None
    min_p: torch.Tensor | None = None        # [n] 最小概率阈值系数；None 表示整批关闭

    # 行 → 随机数生成器（按请求 seed 持久化，跨步续流）；None/空表示无种子
    generators: dict[int, torch.Generator] | None = None
    # 行 → 该请求的禁止 token 序列列表；None 表示整批无 bad_words
    bad_words_token_ids: dict[int, list[list[int]]] | None = None

    no_penalties: bool = True
    prompt_token_ids: list[list[int]] | None = None
    output_token_ids: list[list[int]] | None = None
    frequency_penalties: torch.Tensor | None = None
    presence_penalties: torch.Tensor | None = None
    repetition_penalties: torch.Tensor | None = None

    # ── LogitsProcessor 框架字段（None/空 → 对应处理器整批 no-op）──────────────
    logit_bias: dict[int, dict[int, float]] | None = None   # 行 → {token_id: bias}
    min_tokens: dict[int, int] | None = None                # 行 → 最小生成长度（未达则抑制 EOS）
    eos_token_id: int | None = None                         # min_tokens / 引导完成用
    grammars: dict[int, object] | None = None               # 行 → Grammar（引导解码，仅 UniProc）

    max_num_logprobs: int | None = None
