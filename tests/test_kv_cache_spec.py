"""KVCacheSpec / FullAttentionSpec 单元测试（块字节数 / 形状 / 显存→块数）。"""
import pytest
import torch

from nanovllm.engine.kv_cache import FullAttentionSpec


def _spec(block_size=16, num_kv_heads=2, head_dim=8, dtype=torch.float16):
    return FullAttentionSpec(block_size=block_size, num_kv_heads=num_kv_heads,
                             head_dim=head_dim, dtype=dtype)


@pytest.mark.unit
def test_page_size_bytes():
    spec = _spec()
    # 2(K+V) * block_size * num_kv_heads * head_dim * itemsize(fp16=2)
    assert spec.page_size_bytes == 2 * 16 * 2 * 8 * 2


@pytest.mark.unit
def test_kv_cache_shape():
    spec = _spec()
    assert spec.kv_cache_shape(10) == (2, 10, 16, 2, 8)


@pytest.mark.unit
def test_num_blocks_for_memory_roundtrip():
    spec = _spec()
    num_layers = 4
    blocks = 100
    available = spec.page_size_bytes * num_layers * blocks
    assert spec.num_blocks_for_memory(available, num_layers) == blocks
    # 不足一整块的余量向下取整
    assert spec.num_blocks_for_memory(available + spec.page_size_bytes - 1,
                                      num_layers) == blocks


@pytest.mark.unit
def test_dtype_affects_page_size():
    assert _spec(dtype=torch.float32).page_size_bytes == 2 * _spec(dtype=torch.float16).page_size_bytes
