"""BlockTable 单元测试（CPU，device="cpu" + pin_memory=False，不依赖 GPU）。"""
import numpy as np
import pytest

from nanovllm.engine.block_table import BlockTable, CpuGpuBuffer


def _bt(block_size=4, max_num_reqs=8, max_blocks=16, max_tokens=64):
    return BlockTable(block_size, max_num_reqs, max_blocks, max_tokens,
                      device="cpu", pin_memory=False)


class TestCpuGpuBuffer:

    @pytest.mark.unit
    def test_np_view_aliases_cpu_and_copy_to_gpu(self):
        import torch
        buf = CpuGpuBuffer(5, dtype=torch.int32, device="cpu", pin_memory=False)
        buf.np[:3] = [1, 2, 3]
        assert buf.cpu[2].item() == 3        # numpy 视图与 CPU 张量同一内存
        out = buf.copy_to_gpu(3)
        assert out.tolist() == [1, 2, 3]


class TestBlockTableRows:

    @pytest.mark.unit
    def test_add_and_append_row(self):
        bt = _bt()
        bt.add_row([10, 11], row_idx=0)
        assert bt.num_blocks_per_row[0] == 2
        bt.append_row([12], row_idx=0)        # decode：仅追加新块
        assert bt.num_blocks_per_row[0] == 3
        assert bt.block_table.np[0, :3].tolist() == [10, 11, 12]

    @pytest.mark.unit
    def test_add_row_overwrites(self):
        bt = _bt()
        bt.add_row([10, 11, 12], 0)
        bt.add_row([5], 0)                     # 整行重写：长度归零再写
        assert bt.num_blocks_per_row[0] == 1
        assert bt.block_table.np[0, 0] == 5

    @pytest.mark.unit
    def test_append_empty_is_noop(self):
        bt = _bt()
        bt.add_row([1], 0)
        bt.append_row([], 0)
        assert bt.num_blocks_per_row[0] == 1

    @pytest.mark.unit
    def test_move_row(self):
        bt = _bt()
        bt.add_row([7, 8, 9], 3)
        bt.move_row(3, 1)                      # condense 回收：把 row3 滑到 row1
        assert bt.num_blocks_per_row[1] == 3
        assert bt.block_table.np[1, :3].tolist() == [7, 8, 9]


class TestComputeSlotMapping:

    @pytest.mark.unit
    def test_slot_mapping_single_req(self):
        bt = _bt(block_size=4)
        bt.add_row([10, 11], 0)               # 6 token 占 2 块
        positions = np.arange(6, dtype=np.int64)
        req_indices = np.zeros(6, dtype=np.int64)
        bt.compute_slot_mapping(req_indices, positions)
        # block 10 → slots 40..43；block 11 → slots 44,45
        assert bt.slot_mapping.np[:6].tolist() == [40, 41, 42, 43, 44, 45]

    @pytest.mark.unit
    def test_slot_mapping_two_reqs(self):
        bt = _bt(block_size=4)
        bt.add_row([10], 0)                    # req0：2 token 在块 10
        bt.add_row([20], 1)                    # req1：1 token 在块 20
        positions = np.array([0, 1, 0], dtype=np.int64)
        req_indices = np.array([0, 0, 1], dtype=np.int64)
        bt.compute_slot_mapping(req_indices, positions)
        assert bt.slot_mapping.np[:3].tolist() == [40, 41, 80]

    @pytest.mark.unit
    def test_commit_and_clear(self):
        bt = _bt()
        bt.add_row([1, 2], 0)
        dev = bt.commit_block_table(1)
        assert dev.shape[0] == 1
        bt.clear()
        assert bt.num_blocks_per_row[0] == 0
        assert int(bt.block_table.np.sum()) == 0
