# 异步调度（async scheduling）对齐 V1 — 调研

> 目标：让 step N 的 GPU 计算与 step N+1 的 CPU 调度/输入构造**重叠**，吃掉每步之间
> CPU 侧（schedule / make_inputs / 采样后处理）的气泡。门控 `Config.async_scheduling`。

## V1 的设计

vLLM V1 `v1/engine/core.py::step()` 在 `async_scheduling=True` 时：
`execute_model(scheduler_output, non_block=True)` 立即返回 future，CPU 不阻塞在 GPU；
调度器在 `_update_after_schedule`（`v1/core/sched/scheduler.py`）里**调度即推进**
`request.num_computed_tokens += num_scheduled_token` 并用 `num_output_placeholders` 给
"尚未产出的 token"占位，使下一步能在当前步结果回来之前就被调度。采样 token 留在 GPU，
下一步的 GPU runner 把上一步 GPU 上的采样张量直接拼进 input_ids（不做 D2H），从而 CPU
路径不被 token 值依赖串行化。**同进程**即可（不要求 EngineCore 独立进程）。

## 两个子问题

| 子问题 | 难度 | 说明 |
|---|---|---|
| **A 调度时推进记账** | 低 | num_computed/长度提前推进 + 占位，使下一步 position/块分配算对 |
| **B 跨步 token 值依赖** | 高 | 采样 token 留 GPU、去掉每步 `.tolist()` 同步、前向回填 input_ids、非阻塞流水 |

真正门槛是 B：nano 的 `ModelRunner.run()` 每步 `sampled.tolist()` 强制 D2H 同步，把相邻
step 串行化；`make_inputs` 又从 `seq.last_token`（CPU）读 decode 输入。要重叠必须让上一步
采样 token 以 GPU 张量形式前向喂给下一步。

## 范围决策（与 V1 的取舍）

| V1 特性 | 是否对齐 | 说明 |
|---|---|---|
| 调度即推进 + 占位 token | ✅ | Sequence.num_pending + advance_after_schedule |
| 采样 token 留 GPU 跨步前向 | ✅ | ModelRunner 两槽（inflight/pending）+ input_ids 就地回填 |
| 非阻塞流水（深度 1） | ✅ | EngineCore 一层在飞步；CUDA 异步天然重叠 CPU/GPU |
| EOS 多调度一步 + 丢弃 | ✅ | 对齐 V1：结束序列被多调度一步，其结果丢弃 |
| 抢占 recompute 与在飞协同 | ✅ | 被抢占序列丢弃在飞占位、干净重算 |
| spec decode 的拒绝回退记账 | ❌ | nano 无投机解码 |
| PP + async（num_output_placeholders 多步） | ❌ | nano 无流水并行；占位深度恒 1 |
| MultiProc / TP>1 的 async | ❌ | 仅 UniProc（TP=1 内联）；与 swap 抢占互斥 |

## 决策依据

1. **默认零回归**：`async_scheduling=False` 默认 → 走原同步 step()，现有路径与测试不变。
2. **单进程足够**：CUDA 默认异步执行，CPU enqueue 下一步 kernel 后即可调度，无需独立进程/线程；
   唯一要求是不在流水中途 `.tolist()` 同步。
3. **互斥项**：swap 抢占会让在飞 token 张量失效；进程隔离需扩展 RPC 载荷传 GPU 张量句柄——
   均暂排除（`__post_init__` 断言）。
