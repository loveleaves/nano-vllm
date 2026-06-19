from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    """
    请求状态机：
      WAITING  — 等待调度（尚未完成 prefill，或被抢占后重新入队）
      RUNNING  — prefill 完成，正在逐 token decode
      FINISHED — 命中 EOS 或达到 max_tokens
    """
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    """
    一次推理请求的完整生命周期状态。

    类变量：
      block_size — KV cache 物理块大小（token 数），由 LLMEngine 从 Config 同步
      counter    — 全局自增 ID，保证 seq_id 唯一且有序

    关键字段：
      token_ids             — 完整 token 序列（prompt + 已生成）
      num_cached_tokens     — 已写入 KV cache 的 token 数（含前缀缓存命中部分）
      num_scheduled_tokens  — 当前步调度的 token 数（scheduler 设置，postprocess 后清零）
      block_table           — 逻辑块索引 → 物理块 ID 的映射列表
    """
    block_size: int = 256
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params: SamplingParams = None,
                 priority: int = 0):
        if sampling_params is None:
            sampling_params = SamplingParams()
        self.seq_id = next(Sequence.counter)
        self.priority = priority   # 调度优先级（值越小越先；PriorityRequestQueue 用）
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
        self.num_scheduled_tokens = 0
        self.block_table: list[int] = []
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos
        # 采样配置（仅 rank0 采样用，不入 __getstate__）
        self.top_p = sampling_params.top_p
        self.top_k = sampling_params.top_k
        self.presence_penalty = sampling_params.presence_penalty
        self.frequency_penalty = sampling_params.frequency_penalty
        self.repetition_penalty = sampling_params.repetition_penalty
        self.logprobs = sampling_params.logprobs

    def __len__(self) -> int:
        return self.num_tokens

    def __getitem__(self, key):
        """支持 seq[start:end] 切片，用于 prepare_prefill 提取待处理 token。"""
        return self.token_ids[key]

    @property
    def is_finished(self) -> bool:
        return self.status == SequenceStatus.FINISHED

    @property
    def is_prefill(self) -> bool:
        """prompt 尚未全部写入 KV cache → 仍处于 prefill 阶段（含 chunked prefill 中途）。

        统一连续批后不再有显式 prefill/decode 标志，该属性由调度进度派生，
        仅用于 __getstate__ 的 pickle 优化（decode 只传 last_token）。
        """
        return self.num_cached_tokens < self.num_prompt_tokens

    @property
    def num_completion_tokens(self) -> int:
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self) -> list[int]:
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self) -> list[int]:
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_blocks(self) -> int:
        """当前序列占用的逻辑块数量（向上取整）。"""
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self) -> int:
        """最后一个逻辑块中已使用的 token 数（1~block_size）。"""
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i: int) -> list[int]:
        """返回第 i 个逻辑块的 token_ids，用于前缀缓存哈希计算。"""
        assert 0 <= i < self.num_blocks
        return self.token_ids[i * self.block_size: (i + 1) * self.block_size]

    def append_token(self, token_id: int):
        """decode 步完成后追加新生成的 token。"""
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def __getstate__(self):
        """
        自定义 pickle 序列化（进程间通信优化）：
          prefill 时序列化完整 token_ids；decode 时只序列化 last_token。
        """
        last_state = self.last_token if not self.is_prefill else self.token_ids
        return (self.seq_id, self.num_tokens, self.num_prompt_tokens,
                self.num_cached_tokens, self.num_scheduled_tokens,
                self.block_table, last_state)

    def __setstate__(self, state):
        (self.seq_id, self.num_tokens, self.num_prompt_tokens,
         self.num_cached_tokens, self.num_scheduled_tokens,
         self.block_table, last_state) = state
        if isinstance(last_state, list):
            self.token_ids = last_state
            self.last_token = self.token_ids[-1]
        else:
            self.token_ids = []
            self.last_token = last_state
