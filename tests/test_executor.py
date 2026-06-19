"""Executor 工厂分派单元测试（CPU）。

UniProc/MultiProc 的实例化需 GPU+NCCL，这里只验证 get_class 按
distributed_executor_backend / TP 规模选择后端（用轻量 stub，不构造真实 Config/Worker）。
"""
from types import SimpleNamespace

import pytest

from nanovllm.engine.executor import Executor, MultiProcExecutor, UniProcExecutor


def _cfg(tp=1, backend=None):
    return SimpleNamespace(tensor_parallel_size=tp, distributed_executor_backend=backend)


@pytest.mark.unit
def test_get_class_uniproc_for_tp1_default():
    assert Executor.get_class(_cfg(tp=1)) is UniProcExecutor


@pytest.mark.unit
def test_get_class_multiproc_for_tp_gt1_default():
    for tp in (2, 4, 8):
        assert Executor.get_class(_cfg(tp=tp)) is MultiProcExecutor


@pytest.mark.unit
def test_explicit_mp_backend_forces_isolation_at_tp1():
    # 显式 "mp"：TP=1 也走进程隔离（单卡可测隔离机制）
    assert Executor.get_class(_cfg(tp=1, backend="mp")) is MultiProcExecutor


@pytest.mark.unit
def test_explicit_uni_backend():
    assert Executor.get_class(_cfg(tp=2, backend="uni")) is UniProcExecutor
