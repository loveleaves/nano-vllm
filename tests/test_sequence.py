"""
Sequence 状态机单元测试
"""
import pickle
import pytest

from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.sampling_params import SamplingParams


def setup_function():
    Sequence.block_size = 4


class TestSequenceBasic:

    @pytest.mark.unit
    def test_initial_attributes(self):
        seq = Sequence([1, 2, 3, 4, 5])
        assert seq.num_tokens == 5
        assert seq.num_prompt_tokens == 5
        assert seq.num_completion_tokens == 0
        assert seq.status == SequenceStatus.WAITING
        assert not seq.is_finished
        assert seq.block_table == []

    @pytest.mark.unit
    def test_token_ids_are_copied(self):
        original = [1, 2, 3]
        seq = Sequence(original)
        original.append(99)
        assert seq.num_tokens == 3  # 不受外部修改影响

    @pytest.mark.unit
    def test_seq_id_unique_and_monotone(self):
        ids = [Sequence([1]).seq_id for _ in range(5)]
        assert len(set(ids)) == 5
        assert ids == sorted(ids)

    @pytest.mark.unit
    def test_getitem_slice(self):
        seq = Sequence([10, 20, 30, 40, 50])
        assert seq[1:3] == [20, 30]
        assert seq[0:5] == [10, 20, 30, 40, 50]

    @pytest.mark.unit
    def test_default_sampling_params(self):
        seq = Sequence([1, 2])
        assert seq.temperature == 1.0
        assert seq.max_tokens == 64


class TestSequenceBlockProperties:

    @pytest.mark.unit
    def test_num_blocks_ceiling_division(self):
        Sequence.block_size = 4
        assert Sequence([0] * 1).num_blocks == 1
        assert Sequence([0] * 4).num_blocks == 1
        assert Sequence([0] * 5).num_blocks == 2
        assert Sequence([0] * 8).num_blocks == 2
        assert Sequence([0] * 9).num_blocks == 3

    @pytest.mark.unit
    def test_last_block_num_tokens(self):
        Sequence.block_size = 4
        assert Sequence([0] * 4).last_block_num_tokens == 4
        assert Sequence([0] * 5).last_block_num_tokens == 1
        assert Sequence([0] * 7).last_block_num_tokens == 3

    @pytest.mark.unit
    def test_block_method_correct_tokens(self):
        Sequence.block_size = 4
        seq = Sequence([10, 20, 30, 40, 50, 60, 70])
        assert seq.block(0) == [10, 20, 30, 40]
        assert seq.block(1) == [50, 60, 70]

    @pytest.mark.unit
    def test_block_out_of_range_raises(self):
        seq = Sequence([1, 2, 3])
        with pytest.raises(AssertionError):
            seq.block(1)


class TestSequenceTokenOperations:

    @pytest.mark.unit
    def test_append_token(self):
        seq = Sequence([1, 2, 3])
        seq.append_token(99)
        assert seq.num_tokens == 4
        assert seq.last_token == 99
        assert seq.token_ids[-1] == 99
        assert seq.num_completion_tokens == 1

    @pytest.mark.unit
    def test_prompt_and_completion_ids(self):
        seq = Sequence([1, 2, 3])
        seq.append_token(100)
        seq.append_token(200)
        assert seq.prompt_token_ids == [1, 2, 3]
        assert seq.completion_token_ids == [100, 200]

    @pytest.mark.unit
    def test_multiple_append_tokens(self):
        seq = Sequence([0])
        for i in range(10):
            seq.append_token(i)
        assert seq.num_tokens == 11
        assert seq.num_completion_tokens == 10


class TestSequencePickle:

    @pytest.mark.unit
    def test_pickle_prefill_state_serializes_full_tokens(self):
        seq = Sequence([1, 2, 3, 4])
        seq.block_table = [0, 1]
        seq.num_cached_tokens = 2
        seq.num_scheduled_tokens = 2
        # num_cached_tokens(2) < num_prompt_tokens(4) → is_prefill 派生为 True
        state = seq.__getstate__()
        data = pickle.dumps(state)
        state2 = pickle.loads(data)
        seq2 = Sequence.__new__(Sequence)
        seq2.__setstate__(state2)
        assert seq2.num_tokens == seq.num_tokens
        assert seq2.token_ids == seq.token_ids

    @pytest.mark.unit
    def test_pickle_decode_state_only_last_token(self):
        seq = Sequence([1, 2, 3, 4])
        # num_cached_tokens(4) >= num_prompt_tokens(4) → is_prefill 派生为 False（decode）
        seq.num_cached_tokens = 4
        seq.last_token = 42
        state = seq.__getstate__()
        data = pickle.dumps(state)
        state2 = pickle.loads(data)
        seq2 = Sequence.__new__(Sequence)
        seq2.__setstate__(state2)
        assert seq2.last_token == 42
        assert seq2.token_ids == []  # decode 时不序列化完整 token_ids

    @pytest.mark.unit
    def test_pickle_restores_block_table(self):
        seq = Sequence([1, 2, 3])
        seq.block_table = [5, 7]
        # 默认 num_cached_tokens(0) < num_prompt_tokens(3) → is_prefill 派生为 True
        seq2 = Sequence.__new__(Sequence)
        seq2.__setstate__(seq.__getstate__())
        assert seq2.block_table == [5, 7]
