"""
EngineCore ↔ Worker 的结构化 RPC 传输层。

从原 ModelRunner 抽出（D：解耦执行器与通信）。传输用 SharedMemory + Event，
序列化用 msgspec.msgpack（替换裸 pickle）：载荷复用 Sequence.__getstate__ 的轻量
元组（全 int / list[int]），msgspec 原生可编码，更快更结构化且无任意代码执行风险。

协议：rank0 broadcast(method, seqs) → 各 rank>0 recv() 得到 (method, seqs)。
  seqs=None 表示无参方法（如 "exit"）。
"""
import msgspec
from multiprocessing.shared_memory import SharedMemory

from nanovllm.engine.sequence import Sequence


class ShmTransport:
    SHM_NAME = "nanovllm"
    SHM_SIZE = 2 ** 20

    def __init__(self, rank: int, events, create: bool):
        """
        rank0：events 为 list[Event]（逐个 set 通知子进程）。
        rank>0：events 为本进程单个 Event（wait/clear）。
        """
        self.rank = rank
        self.events = events
        self.shm = (SharedMemory(name=self.SHM_NAME, create=True, size=self.SHM_SIZE)
                    if create else SharedMemory(name=self.SHM_NAME))

    # ── 序列化（静态、可独立单测）──────────────────────────────────────────────
    @staticmethod
    def encode(method: str, seqs: list[Sequence] | None) -> bytes:
        states = [s.__getstate__() for s in seqs] if seqs is not None else None
        return msgspec.msgpack.encode((method, states))

    @staticmethod
    def decode(data: bytes) -> tuple[str, list[Sequence] | None]:
        method, states = msgspec.msgpack.decode(data)
        seqs = None
        if states is not None:
            seqs = []
            for st in states:
                seq = Sequence.__new__(Sequence)
                seq.__setstate__(st)
                seqs.append(seq)
        return method, seqs

    # ── 传输 ────────────────────────────────────────────────────────────────
    def broadcast(self, method: str, seqs: list[Sequence] | None = None):
        """rank0：写入 shm 并通知所有子进程。"""
        data = self.encode(method, seqs)
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n + 4] = data
        for event in self.events:
            event.set()

    def recv(self) -> tuple[str, list[Sequence] | None]:
        """rank>0：等待并读取一条指令。"""
        self.events.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method, seqs = self.decode(bytes(self.shm.buf[4:n + 4]))
        self.events.clear()
        return method, seqs

    def close(self):
        self.shm.close()

    def unlink(self):
        """仅 rank0 调用：释放共享内存段（需在所有 rank close 之后）。"""
        self.shm.unlink()
