"""
持久化块表（对齐 vLLM V1 `v1/worker/block_table.py::BlockTable` + `v1/utils.py::CpuGpuBuffer`）。

V1 ModelRunner 不在每步重建 padded block_table 再整表 H2D，而是把"逻辑块 → 物理块"
映射常驻一对 CPU(pinned)+GPU 缓冲：
  - prefill / 行重写：`add_row` 覆盖整行；
  - decode：`append_row` 仅追加本步新分配的块（绝大多数步只写 1 个 int）；
  - `commit_block_table(n)` 只把前 n 行异步刷到 GPU。
slot_mapping 同样常驻缓冲，按 (req_indices, positions) 向量化计算（compute_slot_mapping）。

nano 单组同构 full-attention，不含 V1 的 hybrid/kernel_block 拆分与 CP/DCP，故大幅精简。
"""
import numpy as np
import torch


class CpuGpuBuffer:
    """一对 CPU(pinned)+GPU 张量，外加 CPU 端 numpy 视图，便于增量写后整体异步上传。

    对齐 V1 `v1/utils.py::CpuGpuBuffer`。int64/int32 等可直接转 numpy；写 numpy 视图
    等价于写 CPU 张量，`copy_to_gpu(n)` 把前 n 个元素（首维）异步拷到 GPU。
    """

    def __init__(self, *size: int, dtype: torch.dtype, device, pin_memory: bool):
        self.cpu = torch.zeros(*size, dtype=dtype, device="cpu", pin_memory=pin_memory)
        self.gpu = torch.zeros(*size, dtype=dtype, device=device)
        self.np = self.cpu.numpy()

    def copy_to_gpu(self, n: int | None = None) -> torch.Tensor:
        if n is None:
            return self.gpu.copy_(self.cpu, non_blocking=True)
        return self.gpu[:n].copy_(self.cpu[:n], non_blocking=True)


class BlockTable:
    """常驻的 (max_num_reqs × max_num_blocks_per_req) 块表 + slot_mapping 缓冲。

    行索引 row_idx 即 InputBatch 中的持久槽位；num_blocks_per_row[row] 记录该行已写块数，
    使 decode 能只追加新块。
    """

    def __init__(self, block_size: int, max_num_reqs: int,
                 max_num_blocks_per_req: int, max_num_batched_tokens: int,
                 device, pin_memory: bool):
        self.block_size = block_size
        self.max_num_reqs = max_num_reqs
        self.max_num_blocks_per_req = max_num_blocks_per_req
        self.device = device
        self.pin_memory = pin_memory

        self.block_table = CpuGpuBuffer(
            max_num_reqs, max_num_blocks_per_req,
            dtype=torch.int32, device=device, pin_memory=pin_memory)
        self.num_blocks_per_row = np.zeros(max_num_reqs, dtype=np.int32)
        self.slot_mapping = CpuGpuBuffer(
            max_num_batched_tokens, dtype=torch.int32,
            device=device, pin_memory=pin_memory)

    # ── 行增量更新 ────────────────────────────────────────────────────────────
    def append_row(self, block_ids: list[int], row_idx: int) -> None:
        """在 row_idx 行尾追加 block_ids（decode：仅本步新分配的块）。"""
        if not block_ids:
            return
        num = len(block_ids)
        start = self.num_blocks_per_row[row_idx]
        self.num_blocks_per_row[row_idx] += num
        self.block_table.np[row_idx, start:start + num] = block_ids

    def add_row(self, block_ids: list[int], row_idx: int) -> None:
        """用 block_ids 覆盖 row_idx 整行（prefill / 行重写）。"""
        self.num_blocks_per_row[row_idx] = 0
        self.append_row(block_ids, row_idx)

    def move_row(self, src: int, tgt: int) -> None:
        """把 src 行整体搬到 tgt 行（condense 回收空洞用）。"""
        num = self.num_blocks_per_row[src]
        self.block_table.np[tgt, :num] = self.block_table.np[src, :num]
        self.num_blocks_per_row[tgt] = num

    # ── slot_mapping 向量化计算 ───────────────────────────────────────────────
    def compute_slot_mapping(self, req_indices: np.ndarray,
                             positions: np.ndarray) -> None:
        """slot = block_table[row, pos//block_size] * block_size + pos%block_size。

        req_indices / positions 为按 token 展开的两个等长数组（行索引、绝对位置）。
        结果写入 slot_mapping.np 前 len(positions) 个元素。
        """
        block_table_indices = (req_indices * self.max_num_blocks_per_req
                               + positions // self.block_size)
        block_numbers = self.block_table.np.ravel()[block_table_indices]
        block_offsets = positions % self.block_size
        np.add(block_numbers * self.block_size, block_offsets,
               out=self.slot_mapping.np[:positions.shape[0]])

    # ── 提交到 GPU ────────────────────────────────────────────────────────────
    def commit_block_table(self, num_reqs: int) -> torch.Tensor:
        self.block_table.copy_to_gpu(num_reqs)
        return self.block_table.gpu[:num_reqs]

    def commit_slot_mapping(self, num_tokens: int) -> torch.Tensor:
        self.slot_mapping.copy_to_gpu(num_tokens)
        return self.slot_mapping.gpu[:num_tokens]

    def clear(self) -> None:
        self.num_blocks_per_row.fill(0)
        self.block_table.np.fill(0)

    def get_device_tensor(self, num_reqs: int) -> torch.Tensor:
        return self.block_table.gpu[:num_reqs]
