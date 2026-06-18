"""Executor 工厂分派单元测试（CPU）。

UniProc/MultiProc 的实例化需 GPU+NCCL，这里只验证 get_class 按 TP 规模选择后端
（用轻量 stub 提供 tensor_parallel_size，不构造真实 Config/Worker）。
"""
from types import SimpleNamespace

import pytest

from nanovllm.engine.executor import Executor, MultiProcExecutor, UniProcExecutor


@pytest.mark.unit
def test_get_class_uniproc_for_tp1():
    assert Executor.get_class(SimpleNamespace(tensor_parallel_size=1)) is UniProcExecutor


@pytest.mark.unit
def test_get_class_multiproc_for_tp_gt1():
    for tp in (2, 4, 8):
        assert Executor.get_class(SimpleNamespace(tensor_parallel_size=tp)) is MultiProcExecutor
