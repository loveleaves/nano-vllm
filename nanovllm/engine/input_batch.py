"""
持久化输入批（对齐 vLLM V1 `v1/worker/gpu_input_batch.py::InputBatch`）。

V1 ModelRunner 不在每步从零构造批输入，而是维护一个**跨步常驻**的 InputBatch：每个
请求占一个持久行槽位（seq_id → row），decode 只增量追加新块、整批输入写入常驻 pinned
缓冲后单次异步 H2D。请求结束/被抢占后其行变空，`condense` 把末尾活跃行滑入空洞回收槽位。

与 V1 取舍：
  - nano 单组同构 full-attention、无 LoRA/spec/多模态/prompt-embeds，故只保留行管理
    （add/remove/condense）+ 块表（BlockTable）+ 每步展开缓冲。
  - **行序即模型批序**：每步把被调度序列按其行号排序后展开，模型按行序前向、采样得到行序
    token，再按 seq_id 映射回调度顺序返回（runner 输出按 req_id 对齐）。nano 的连续批每步
    调度整个活跃集合，故"逐出已结束行 + 补入新行"后批集合恰等于本步调度集合（make_inputs
    内有断言保证）。
"""
import numpy as np
import torch

from nanovllm.engine.block_table import BlockTable, CpuGpuBuffer
from nanovllm.engine.sequence import Sequence
from nanovllm.utils.context import AttentionMetadata


class InputBatch:

    def __init__(self, max_num_reqs: int, max_num_blocks_per_req: int,
                 max_num_batched_tokens: int, block_size: int,
                 device, pin_memory: bool = True):
        self.max_num_reqs = max_num_reqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.block_size = block_size

        # 持久行槽位：行号 ↔ seq_id（_req_ids[row] is None 表示空洞，待 condense 回收）
        self._req_ids: list[int | None] = []
        self.req_id_to_index: dict[int, int] = {}

        self.block_table = BlockTable(
            block_size, max_num_reqs, max_num_blocks_per_req,
            max_num_batched_tokens, device, pin_memory)

        # 每步展开的常驻缓冲（int64 token/position，int32 累积长度）
        self.input_ids = CpuGpuBuffer(
            max_num_batched_tokens, dtype=torch.int64, device=device, pin_memory=pin_memory)
        self.positions = CpuGpuBuffer(
            max_num_batched_tokens, dtype=torch.int64, device=device, pin_memory=pin_memory)
        self.query_start_loc = CpuGpuBuffer(
            max_num_reqs + 1, dtype=torch.int32, device=device, pin_memory=pin_memory)
        self.seq_lens = CpuGpuBuffer(
            max_num_reqs + 1, dtype=torch.int32, device=device, pin_memory=pin_memory)

    @property
    def num_reqs(self) -> int:
        return len(self.req_id_to_index)

    # ── 行管理（add / remove / condense）──────────────────────────────────────
    def add_request(self, seq: Sequence) -> int:
        """为新请求分配行槽位（condense 后批致密，新行追加在末尾）。"""
        row = len(self._req_ids)
        self._req_ids.append(seq.seq_id)
        self.req_id_to_index[seq.seq_id] = row
        self.block_table.add_row(seq.block_table, row)
        return row

    def remove_request(self, seq_id: int) -> int | None:
        """请求结束 / 抢占：释放其行（置空，等待 condense 回收）。返回行号或 None。"""
        row = self.req_id_to_index.pop(seq_id, None)
        if row is None:
            return None
        self._req_ids[row] = None
        return row

    def condense(self) -> None:
        """把末尾活跃行滑入低位空洞，保持 [0, num_reqs) 致密（对齐 V1 condense）。

        只搬动必要的行——稳定存活的请求保持原行号，故 decode 增量追加块不受影响。
        """
        empty = [i for i, sid in enumerate(self._req_ids) if sid is None]
        if not empty:
            return
        empty.sort(reverse=True)            # 升序空洞，从尾部弹出最小空洞
        last = len(self._req_ids) - 1
        while empty:
            while last >= 0 and self._req_ids[last] is None:
                last -= 1
            empty_index = empty[-1]
            if empty_index >= last:
                break
            empty.pop()
            sid = self._req_ids[last]
            self._req_ids[empty_index] = sid
            self._req_ids[last] = None
            self.req_id_to_index[sid] = empty_index
            self.block_table.move_row(last, empty_index)
            last -= 1
        # 截断末尾空洞行
        del self._req_ids[self.num_reqs:]

    def clear(self) -> None:
        self._req_ids.clear()
        self.req_id_to_index.clear()
        self.block_table.clear()

    # ── 每步同步 + 展开 ───────────────────────────────────────────────────────
    def update(self, scheduled_seqs: list[Sequence],
               finished_seq_ids: set[int] | None) -> None:
        """逐出已结束/被抢占行 → condense 回收 → 补入新请求 / decode 追加新块。"""
        removed = False
        if finished_seq_ids:
            for sid in finished_seq_ids:
                if self.remove_request(sid) is not None:
                    removed = True
        if removed:
            self.condense()
        for seq in scheduled_seqs:
            row = self.req_id_to_index.get(seq.seq_id)
            if row is None:
                self.add_request(seq)
            else:
                # decode / chunked-prefill 续算：仅追加本步新分配的块
                cur = self.block_table.num_blocks_per_row[row]
                if len(seq.block_table) > cur:
                    self.block_table.append_row(seq.block_table[cur:], row)

    def make_inputs(self, scheduled_seqs: list[Sequence]):
        """把被调度序列按行号排序后展开成模型输入（写常驻缓冲 + 单次 H2D）。

        返回 (input_ids, positions, attn_md, ordered_seqs)，其中 ordered_seqs 为模型实际
        前向/采样的行序，调用方据此把行序输出按 seq_id 映射回原调度顺序。
        """
        assert self.num_reqs == len(scheduled_seqs), \
            "批集合须等于本步调度集合（逐出已结束行 + 补入新行后应一致）"
        ordered = sorted(scheduled_seqs, key=lambda s: self.req_id_to_index[s.seq_id])
        n = len(ordered)

        input_ids: list[int] = []
        positions: list[int] = []
        req_indices: list[int] = []
        cu_q = [0]
        cu_k = [0]
        max_q = max_k = 0
        has_cache = bool(self.block_table.num_blocks_per_row[:n].any())

        for row, seq in enumerate(ordered):
            start = seq.num_cached_tokens
            q = seq.num_scheduled_tokens
            end = start + q
            if seq.token_ids:
                input_ids.extend(seq[start:end])
            else:
                input_ids.append(seq.last_token)   # rank>0 decode：仅 last_token
            positions.extend(range(start, end))
            req_indices.extend([row] * q)
            cu_q.append(cu_q[-1] + q)
            cu_k.append(cu_k[-1] + end)             # KV 总长 = 已缓存 + 本步
            max_q = max(max_q, q)
            max_k = max(max_k, end)

        total = len(input_ids)
        self.input_ids.np[:total] = input_ids
        self.positions.np[:total] = positions
        self.query_start_loc.np[:n + 1] = cu_q
        self.seq_lens.np[:n + 1] = cu_k

        if has_cache:
            req_indices_arr = np.asarray(req_indices, dtype=np.int64)
            self.block_table.compute_slot_mapping(
                req_indices_arr, self.positions.np[:total])
            sm_gpu = self.block_table.commit_slot_mapping(total)
            block_tables = self.block_table.commit_block_table(n)
        else:
            self.block_table.slot_mapping.np[:total] = -1
            sm_gpu = self.block_table.commit_slot_mapping(total)
            block_tables = None

        input_ids_gpu = self.input_ids.copy_to_gpu(total)
        positions_gpu = self.positions.copy_to_gpu(total)
        cu_q_gpu = self.query_start_loc.copy_to_gpu(n + 1)
        cu_k_gpu = self.seq_lens.copy_to_gpu(n + 1)
        attn_md = AttentionMetadata(
            query_start_loc=cu_q_gpu, cu_seqlens_k=cu_k_gpu,
            max_query_len=max_q, max_seq_len=max_k,
            slot_mapping=sm_gpu, block_table=block_tables,
        )
        return input_ids_gpu, positions_gpu, attn_md, ordered
