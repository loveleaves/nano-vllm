"""
ShmTransport 序列化往返 + ResultChannel 回传往返单元测试。
（ShmTransport 编解码不依赖多进程；ResultChannel 用进程内 Event+SharedMemory 自收发。）
"""
import multiprocessing as mp
import os

import pytest

from nanovllm.engine.rpc import ResultChannel, ShmTransport
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
        method, seqs, _ = ShmTransport.decode(data)

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

        method, seqs, _ = ShmTransport.decode(ShmTransport.encode("run", [seq]))
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
        method, out, _ = ShmTransport.decode(ShmTransport.encode("run", seqs))
        assert method == "run" and len(out) == 2
        assert out[0].num_tokens == 3 and out[1].num_tokens == 2

    @pytest.mark.unit
    def test_roundtrip_no_args(self):
        data = ShmTransport.encode("exit", None)
        method, seqs, finished = ShmTransport.decode(data)
        assert method == "exit" and seqs is None and finished is None

    @pytest.mark.unit
    def test_roundtrip_finished_seq_ids(self):
        Sequence.block_size = 256
        seq = Sequence([1, 2, 3])
        seq.num_scheduled_tokens = 3
        data = ShmTransport.encode("run", [seq], {3, 7})
        method, seqs, finished = ShmTransport.decode(data)
        assert method == "run" and len(seqs) == 1
        assert finished == {3, 7}

    @pytest.mark.unit
    def test_encoded_is_bytes_and_compact(self):
        Sequence.block_size = 256
        seq = Sequence([1, 2, 3, 4])
        seq.num_cached_tokens = 4
        seq.last_token = 9
        data = ShmTransport.encode("run", [seq])
        assert isinstance(data, (bytes, bytearray))


class TestResultChannel:
    """输出 rank → executor 回传通道的收发往返（进程内自收发）。"""

    @pytest.mark.unit
    @pytest.mark.parametrize("payload", [[7, 42, 1000], None, 123, [0]])
    def test_send_recv_roundtrip(self, payload):
        # 用唯一 name，避免与并发运行的引擎实例（固定名 nanovllm_result）撞段
        name = f"nanovllm_test_{os.getpid()}_a"
        event = mp.get_context("spawn").Event()
        chan = ResultChannel(event, create=True, name=name)
        try:
            chan.send(payload)          # 模拟输出 worker 写入
            assert chan.recv() == payload   # executor 读取
        finally:
            chan.close()
            chan.unlink()

    @pytest.mark.unit
    def test_event_cleared_after_recv(self):
        name = f"nanovllm_test_{os.getpid()}_b"
        event = mp.get_context("spawn").Event()
        chan = ResultChannel(event, create=True, name=name)
        try:
            chan.send([1, 2])
            chan.recv()
            assert not event.is_set()   # recv 后事件复位，供下一轮 wait
        finally:
            chan.close()
            chan.unlink()
