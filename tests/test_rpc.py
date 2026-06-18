"""
ShmTransport 序列化往返单元测试（不依赖多进程 / SharedMemory）。
"""
import pytest

from nanovllm.engine.rpc import ShmTransport
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams


class TestRpcEncodeDecode:

    @pytest.mark.unit
    def test_roundtrip_prefill_seq(self):
        Sequence.block_size = 256
        seq = Sequence([1, 2, 3, 4], SamplingParams(max_tokens=8))
        seq.block_table = [5, 9]
        seq.num_cached_tokens = 0          # prefill：序列化完整 token_ids
        seq.num_scheduled_tokens = 4

        data = ShmTransport.encode("run", [seq])
        method, seqs = ShmTransport.decode(data)

        assert method == "run"
        assert len(seqs) == 1
        r = seqs[0]
        assert r.num_tokens == seq.num_tokens
        assert r.token_ids == seq.token_ids      # prefill 全 token 还原
        assert r.block_table == [5, 9]
        assert r.num_cached_tokens == 0
        assert r.num_scheduled_tokens == 4

    @pytest.mark.unit
    def test_roundtrip_decode_seq_only_last_token(self):
        Sequence.block_size = 256
        seq = Sequence([1, 2, 3, 4], SamplingParams(max_tokens=8))
        seq.num_cached_tokens = 4          # decode：is_prefill 派生 False → 只传 last_token
        seq.last_token = 42

        method, seqs = ShmTransport.decode(ShmTransport.encode("run", [seq]))
        r = seqs[0]
        assert r.last_token == 42
        assert r.token_ids == []           # decode 不还原完整 token_ids
        assert r.num_cached_tokens == 4

    @pytest.mark.unit
    def test_roundtrip_multiple_seqs(self):
        Sequence.block_size = 256
        seqs = [Sequence([1, 2, 3]), Sequence([4, 5])]
        for s in seqs:
            s.num_scheduled_tokens = s.num_tokens
        method, out = ShmTransport.decode(ShmTransport.encode("run", seqs))
        assert method == "run" and len(out) == 2
        assert out[0].num_tokens == 3 and out[1].num_tokens == 2

    @pytest.mark.unit
    def test_roundtrip_no_args(self):
        data = ShmTransport.encode("exit", None)
        method, seqs = ShmTransport.decode(data)
        assert method == "exit" and seqs is None

    @pytest.mark.unit
    def test_encoded_is_bytes_and_compact(self):
        Sequence.block_size = 256
        seq = Sequence([1, 2, 3, 4])
        seq.num_cached_tokens = 4
        seq.last_token = 9
        data = ShmTransport.encode("run", [seq])
        assert isinstance(data, (bytes, bytearray))
