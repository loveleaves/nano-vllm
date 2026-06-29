# EngineCore 进程化（进程拓扑）— 测试

> 单测：`tests/test_core_client.py`（6 例，`-m unit`，免 GPU）。

## 策略

- **busy_loop / _handle_input** 是进程拓扑的真实逻辑，与传输无关 → 用假核心 +
  `queue.Queue` 在进程内直测（ADD/ABORT/EXIT、空闲阻塞、空步不投递、跑到结束）。
- **MPClient 端到端** 用 **fork** 上下文 + 假核心子进程跑通（真跨进程 mp.Queue 收发，
  无 GPU/CUDA）。生产默认 **spawn + 真 EngineCore**，由 GPU 集成覆盖。
  - 为何 fork：spawn 需按模块名 import 子进程目标，pytest 测试模块下不稳；fork 继承内存，
    假核心可内联。CPU 测试未初始化 CUDA，fork 安全（仅有 Python 3.12 多线程 fork 的
    DeprecationWarning，无害）。

## 运行

```bash
source .venv/bin/activate
pytest tests/test_core_client.py -m unit -v
# 全量回归
pytest -m unit -q          # 304 passed
```

## 覆盖矩阵

| 用例 | 验证点 |
|---|---|
| `test_handle_input_add_abort_exit` | ADD/ABORT 改核心状态并返 True；EXIT 返 False |
| `test_busy_loop_runs_to_finish` | 线程跑 busy_loop：ADD→逐步产出 token→finished→EXIT 退出 |
| `test_busy_loop_idle_blocks_then_exits` | 空闲阻塞不产出；EXIT 从空闲态唤醒退出（不空转） |
| `test_delegates_to_core`（InprocClient） | get_output 委派 step、add/abort/has_unfinished/exit 委派核心 |
| `test_end_to_end`（MPClient fork） | 真子进程：两请求各产 2 token、has_unfinished 跟踪、exit 回收 proc |
| `test_abort_clears_unfinished`（MPClient fork） | abort 后 has_unfinished 立即转 False |

## 结果

- `tests/test_core_client.py`：6 passed
- 全量 `-m unit`：**304 passed**，4 deselected，无回归。
- 受影响的既有测试：`test_async_llm`（注入式 fake 改为客户端接口 get_output_async）通过；
  `test_engine_core` / `test_async_scheduling`（直接测 EngineCore 类，未改）通过。

## 手动 GPU 联调（需真模型）

```python
# main.py —— spawn 需 __main__ 守卫
from nanovllm import LLMEngine, SamplingParams
if __name__ == "__main__":
    eng = LLMEngine("~/model/Qwen3-1.7B", enforce_eager=True,
                    multiproc_engine_core=True)        # EngineCore 跑独立子进程
    print(eng.generate(["Hello,"], SamplingParams(temperature=0, max_tokens=16)))
    eng.exit()
```
预期：与 `multiproc_engine_core=False`（InprocClient）逐 token 一致；前端进程不初始化 CUDA。
