"""
Executor ↔ Worker 的结构化 RPC 传输层。

从原 ModelRunner 抽出（D：解耦执行器与通信）。传输用 SharedMemory + Event，
序列化用 msgspec.msgpack（替换裸 pickle）：载荷复用 Sequence.__getstate__ 的轻量
元组（全 int / list[int]），msgspec 原生可编码，更快更结构化且无任意代码执行风险。

两条单向通道（进程隔离后 executor 不在 NCCL 组内、不内联任何 Worker）：
  ShmTransport   — 广播：executor → 所有 worker rank。executor 为创建方（events 为各
                   worker 的 Event 列表），worker 为打开方（events 为自身单个 Event）。
                   broadcast(method, seqs, finished) → 各 worker recv() 得到三元组。
  ResultChannel  — 回传：输出 rank（rank0，唯一采样者）→ executor。executor recv() 取回
                   该 rank 的结果（"run"→token_ids，"num_kvcache_blocks"→int）。
"""
import msgspec
from multiprocessing.shared_memory import SharedMemory

from nanovllm.engine.sequence import Sequence


class ShmTransport:
    SHM_NAME = "nanovllm"
    SHM_SIZE = 2 ** 20

    def __init__(self, rank: int, events, create: bool):
        """
        executor（create=True）：events 为 list[Event]（逐个 set 通知各 worker）。
        worker（create=False）：events 为本进程单个 Event（wait/clear）。
        """
        self.rank = rank
        self.events = events
        self.shm = (SharedMemory(name=self.SHM_NAME, create=True, size=self.SHM_SIZE)
                    if create else SharedMemory(name=self.SHM_NAME))

    # ── 序列化（静态、可独立单测）──────────────────────────────────────────────
    @staticmethod
    def encode(method: str, seqs: list[Sequence] | None,
               finished: set[int] | None = None) -> bytes:
        states = [s.__getstate__() for s in seqs] if seqs is not None else None
        finished_list = sorted(finished) if finished else None
        return msgspec.msgpack.encode((method, states, finished_list))

    @staticmethod
    def decode(data: bytes) -> tuple[str, list[Sequence] | None, set[int] | None]:
        method, states, finished_list = msgspec.msgpack.decode(data)
        seqs = None
        if states is not None:
            seqs = []
            for st in states:
                seq = Sequence.__new__(Sequence)
                seq.__setstate__(st)
                seqs.append(seq)
        finished = set(finished_list) if finished_list else None
        return method, seqs, finished

    # ── 传输 ────────────────────────────────────────────────────────────────
    def broadcast(self, method: str, seqs: list[Sequence] | None = None,
                  finished: set[int] | None = None):
        """rank0：写入 shm 并通知所有子进程。"""
        data = self.encode(method, seqs, finished)
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n + 4] = data
        for event in self.events:
            event.set()

    def recv(self) -> tuple[str, list[Sequence] | None, set[int] | None]:
        """rank>0：等待并读取一条指令。"""
        self.events.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method, seqs, finished = self.decode(bytes(self.shm.buf[4:n + 4]))
        self.events.clear()
        return method, seqs, finished

    def close(self):
        self.shm.close()

    def unlink(self):
        """仅创建方（executor）调用：释放共享内存段。"""
        self.shm.unlink()


class ResultChannel:
    """输出 rank → executor 的单向回传通道（SharedMemory + 单个 Event）。

    载荷为任意 msgpack 可编码对象（"run" 的 token_ids: list[int]|None，
    "num_kvcache_blocks" 的 int）。executor 创建并 recv，输出 worker 打开并 send。
    """
    SHM_NAME = "nanovllm_result"
    SHM_SIZE = 2 ** 20

    def __init__(self, event, create: bool, name: str | None = None):
        self.event = event
        name = name or self.SHM_NAME
        self.shm = (SharedMemory(name=name, create=True, size=self.SHM_SIZE)
                    if create else SharedMemory(name=name))

    def send(self, obj):
        """输出 worker：写入结果并通知 executor。"""
        data = msgspec.msgpack.encode(obj)
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n + 4] = data
        self.event.set()

    def recv(self, alive_check=None, poll: float = 1.0):
        """executor：等待并读取输出 rank 的结果。

        alive_check 非空时改为轮询等待，每 poll 秒调用一次 alive_check()——若某 worker
        子进程已死，alive_check 应抛异常，避免 event 永远不被 set 导致永久阻塞。
        """
        if alive_check is None:
            self.event.wait()
        else:
            while not self.event.wait(poll):
                alive_check()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        obj = msgspec.msgpack.decode(bytes(self.shm.buf[4:n + 4]))
        self.event.clear()
        return obj

    def close(self):
        self.shm.close()

    def unlink(self):
        self.shm.unlink()
