# Worker/Executor 进程隔离对齐 — 测试设计

## 测试矩阵

| 文件 | 覆盖点 | 依赖 |
|---|---|---|
| `test_executor.py`（改） | get_class 按 backend 分派：None+TP1→Uni、None+TP>1→Mp、显式 "mp"@TP1→Mp（隔离）、显式 "uni"@TP2→Uni | CPU |
| `test_rpc.py`（+ResultChannel） | 回传通道收发往返（list[int]/None/int/单元素）；recv 后 Event 复位 | CPU（真 SharedMemory，进程内自收发） |
| `test_rpc.py`（ShmTransport） | 广播三元组编解码（含 finished_seq_ids）不变 | CPU |
| `test_config.py` | distributed_executor_backend 合法性（None/"uni"/"mp"） | CPU |
| `test_sequence.py`（+） | 采样标量随 `__getstate__/__setstate__` 还原（隔离 rank0 采样反序列化 seq 的回归守卫） | CPU |
| 隔离 E2E 对照（脚本，单卡可跑） | `backend="mp"` 单 worker 子进程 vs 内联 `uni`，**temperature=0 真·greedy 输出 token 逐一致** | GPU |

## 关键校验：隔离路径 GPU 对照

单卡即可验证隔离机制（显式 `distributed_executor_backend="mp"` 让 TP=1 也走子进程）：
同一模型、同一 greedy(temperature=0) 提示，内联 `uni` 与隔离 `mp` 两条路径生成的
`token_ids` **逐 token 完全一致** —— 证明：spawn worker、NCCL init(world_size=1)、广播 "run"、
ResultChannel 回传 token、`num_kvcache_blocks` RPC 回填、shutdown 编排全链正确。**实测两 prompt
token_ids 完全相同**（uni 与 mp）。

```python
# 注意：uni / mp 必须各在独立进程跑（uni 的 in-process NCCL 占用 2333，
#       同进程再起 mp 子进程会 EADDRINUSE）；入口须在 if __name__ == "__main__":
sp = SamplingParams(temperature=0.0, max_tokens=32)   # 真·greedy → 确定性
# 进程 A：LLM(path).generate(...)
# 进程 B：LLM(path, distributed_executor_backend="mp").generate(...)
assert out_uni[i]["token_ids"] == out_mp[i]["token_ids"]   # 全部 i —— 已验证通过
```

## 回归

- 全量套件：**237 passed, 4 skipped**（较采样轮 +7：test_rpc ResultChannel 2、test_executor +2、
  test_sequence 采样标量回归 1、test_config 由 TP→backend 改写）。默认 TP=1→UniProc 内联路径
  不变，example.py/bench.py 零改动。

## 限制

- **TP>1（真正多 worker NCCL all_reduce）仍需多卡，未真机验证**——与 H 同级限制。单卡只能验证
  单 worker 隔离（world_size=1，无跨 rank 通信）。
- 单槽 SharedMemory 非 V1 多槽 MessageQueue：依赖 execute_model 同步 + "run" 内 NCCL 集体
  保证时序安全（见 design.md），未实现多槽缓冲 / 失败回调 / 健康监控。
