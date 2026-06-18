import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.worker import Worker


class LLMEngine:
    """
    推理引擎主入口（Phase 4：多进程 TP）。

    多进程架构（Worker/ModelRunner 分层 + ShmTransport RPC）：
      rank 0（主进程）：调度 + Worker.call（broadcast + 本地推理 + 采样）
      rank 1..N（子进程）：Worker.loop()，经 ShmTransport(SharedMemory+Event+msgspec) 等待指令
    """

    def __init__(self, model: str, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        Sequence.block_size = config.kvcache_block_size

        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")

        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=Worker, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)

        self.worker = Worker(config, 0, self.events)

        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id

        self.scheduler = Scheduler(
            num_kvcache_blocks=config.num_kvcache_blocks,
            block_size=config.kvcache_block_size,
            max_num_seqs=config.max_num_seqs,
            max_num_batched_tokens=config.max_num_batched_tokens,
            eos=config.eos,
        )
        atexit.register(self.exit)

    def exit(self):
        self.worker.call("exit")
        del self.worker
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self) -> tuple[list[tuple], int]:
        seqs, num_scheduled = self.scheduler.schedule()
        if not seqs:
            return [], 0
        # 吞吐显示：批内含 prefill chunk（任一 seq 调度 >1 token）记为 prefill，
        # 否则为纯 decode（每 seq 1 token）。统一连续批下二者可混排，此处仅用于展示。
        total = sum(num_scheduled.values())
        is_prefill_step = any(n > 1 for n in num_scheduled.values())
        num_tokens = total if is_prefill_step else -len(seqs)
        token_ids = self.worker.call("run", seqs)
        self.scheduler.postprocess(seqs, token_ids, num_scheduled)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        return outputs, num_tokens

    def is_finished(self) -> bool:
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[dict]:
        pbar = tqdm(total=len(prompts), desc="Generating",
                    dynamic_ncols=True, disable=not use_tqdm)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)

        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)

        outputs = {}
        prefill_throughput = decode_throughput = 0.0
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if num_tokens > 0:
                prefill_throughput = num_tokens / (perf_counter() - t)
            elif num_tokens < 0:
                decode_throughput = -num_tokens / (perf_counter() - t)
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                pbar.update(1)

        pbar.close()
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        return [
            {"text": self.tokenizer.decode(tids), "token_ids": tids}
            for tids in outputs
        ]
