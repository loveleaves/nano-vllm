"""InputBatch 单元测试（CPU，device="cpu" + pin_memory=False）。

覆盖：持久行槽位 add/remove/condense 回收、decode 增量追加块、make_inputs 行序展开
与 slot_mapping/cu_seqlens 构造、批集合==调度集合断言。
"""
import pytest

from nanovllm.engine.input_batch import InputBatch
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams


def _ib(block_size=4, max_num_reqs=8, max_blocks=16, max_tokens=64):
    Sequence.block_size = block_size
    return InputBatch(max_num_reqs, max_blocks, max_tokens, block_size,
                      device="cpu", pin_memory=False)


def _seq(tokens, block_table, num_cached, num_scheduled):
    s = Sequence(tokens, SamplingParams())
    s.block_table = list(block_table)
    s.num_cached_tokens = num_cached
    s.num_scheduled_tokens = num_scheduled
    return s


class TestRowManagement:

    @pytest.mark.unit
    def test_add_request_assigns_dense_rows(self):
        ib = _ib()
        a, b, c = _seq([1], [10], 0, 1), _seq([2], [11], 0, 1), _seq([3], [12], 0, 1)
        assert ib.add_request(a) == 0
        assert ib.add_request(b) == 1
        assert ib.add_request(c) == 2
        assert ib.num_reqs == 3
        assert ib.req_id_to_index == {a.seq_id: 0, b.seq_id: 1, c.seq_id: 2}

    @pytest.mark.unit
    def test_remove_and_condense_recycles_row(self):
        ib = _ib()
        a, b, c = _seq([1], [10], 0, 1), _seq([2], [20], 0, 1), _seq([3], [30], 0, 1)
        ib.add_request(a); ib.add_request(b); ib.add_request(c)
        # 移除中间行 b → 留下空洞 row1
        assert ib.remove_request(b.seq_id) == 1
        ib.condense()                            # 末尾活跃行 c 滑入 row1
        assert ib.num_reqs == 2
        assert ib.req_id_to_index[a.seq_id] == 0
        assert ib.req_id_to_index[c.seq_id] == 1
        assert c.seq_id in ib.req_id_to_index and b.seq_id not in ib.req_id_to_index
        # c 的块表也随之搬到 row1
        assert ib.block_table.block_table.np[1, 0] == 30

    @pytest.mark.unit
    def test_remove_unknown_is_noop(self):
        ib = _ib()
        assert ib.remove_request(999) is None

    @pytest.mark.unit
    def test_condense_no_holes_noop(self):
        ib = _ib()
        a, b = _seq([1], [10], 0, 1), _seq([2], [20], 0, 1)
        ib.add_request(a); ib.add_request(b)
        ib.condense()
        assert ib.num_reqs == 2

    @pytest.mark.unit
    def test_clear(self):
        ib = _ib()
        ib.add_request(_seq([1], [10], 0, 1))
        ib.clear()
        assert ib.num_reqs == 0 and ib._req_ids == []


class TestUpdate:

    @pytest.mark.unit
    def test_update_adds_new_and_evicts_finished(self):
        ib = _ib()
        a, b = _seq([1], [10], 0, 1), _seq([2], [20], 0, 1)
        ib.update([a, b], finished_seq_ids=None)
        assert ib.num_reqs == 2
        # 下一步：a 结束，新增 c
        c = _seq([3], [30], 0, 1)
        ib.update([b, c], finished_seq_ids={a.seq_id})
        assert ib.num_reqs == 2
        assert a.seq_id not in ib.req_id_to_index
        assert b.seq_id in ib.req_id_to_index and c.seq_id in ib.req_id_to_index

    @pytest.mark.unit
    def test_update_decode_appends_only_new_blocks(self):
        ib = _ib(block_size=4)
        # prefill：4 token 占 1 块
        a = _seq([1, 2, 3, 4], [10], 0, 4)
        ib.update([a], None)
        assert ib.block_table.num_blocks_per_row[0] == 1
        # decode 跨块：分配新块 11，block_table 变 [10, 11]
        a.block_table = [10, 11]
        a.num_cached_tokens = 4
        a.num_scheduled_tokens = 1
        ib.update([a], None)
        assert ib.block_table.num_blocks_per_row[0] == 2   # 仅 +1
        assert ib.block_table.block_table.np[0, :2].tolist() == [10, 11]


class TestMakeInputs:

    @pytest.mark.unit
    def test_make_inputs_orders_by_row_and_builds_cu_seqlens(self):
        ib = _ib(block_size=4)
        a = _seq([1, 2], [10], 0, 2)          # row0：prefill 2 token
        b = _seq([5, 6, 7], [20], 0, 3)       # row1：prefill 3 token
        ib.update([a, b], None)
        # 以乱序传入，make_inputs 应按行号排序
        input_ids, positions, attn_md, ordered = ib.make_inputs([b, a])
        assert [s.seq_id for s in ordered] == [a.seq_id, b.seq_id]
        assert input_ids[:5].tolist() == [1, 2, 5, 6, 7]
        assert positions[:5].tolist() == [0, 1, 0, 1, 2]
        assert attn_md.query_start_loc[:3].tolist() == [0, 2, 5]
        assert attn_md.cu_seqlens_k[:3].tolist() == [0, 2, 5]
        assert attn_md.max_query_len == 3

    @pytest.mark.unit
    def test_make_inputs_slot_mapping_with_cache(self):
        ib = _ib(block_size=4)
        a = _seq([1, 2, 3, 4, 5, 6], [10, 11], 0, 6)
        ib.update([a], None)
        _, _, attn_md, _ = ib.make_inputs([a])
        assert attn_md.slot_mapping[:6].tolist() == [40, 41, 42, 43, 44, 45]
        assert attn_md.block_table is not None

    @pytest.mark.unit
    def test_make_inputs_no_cache_warmup_path(self):
        ib = _ib(block_size=4)
        a = _seq([1, 2, 3], [], 0, 3)         # 无 KV 块（warmup）
        ib.update([a], None)
        _, _, attn_md, _ = ib.make_inputs([a])
        assert attn_md.block_table is None
        assert attn_md.slot_mapping[:3].tolist() == [-1, -1, -1]

    @pytest.mark.unit
    def test_make_inputs_asserts_batch_equals_scheduled(self):
        ib = _ib()
        a, b = _seq([1], [10], 0, 1), _seq([2], [20], 0, 1)
        ib.update([a, b], None)
        with pytest.raises(AssertionError):
            ib.make_inputs([a])               # 漏传 b → 批集合 != 调度集合

    @pytest.mark.unit
    def test_make_inputs_decode_only_last_token(self):
        ib = _ib(block_size=4)
        # 模拟 rank>0 decode：token_ids 为空，仅 last_token 可用
        s = Sequence([1, 2, 3, 4], SamplingParams())
        s.block_table = [10]
        s.num_cached_tokens = 4
        s.num_scheduled_tokens = 1
        s.token_ids = []
        s.last_token = 99
        ib.update([s], None)
        input_ids, positions, _, _ = ib.make_inputs([s])
        assert input_ids[:1].tolist() == [99]
        assert positions[:1].tolist() == [4]
