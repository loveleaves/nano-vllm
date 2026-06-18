"""BlockPool 单元测试（块级原语：分配/释放/复用缓存块/哈希注册）。"""
import pytest

from nanovllm.engine.kv_cache import BlockPool


@pytest.mark.unit
def test_initial_state():
    pool = BlockPool(num_blocks=4, block_size=4)
    assert pool.get_num_free_blocks() == 4
    assert not pool.used_block_ids


@pytest.mark.unit
def test_get_new_block_and_deref():
    pool = BlockPool(4, 4)
    bid = pool.get_new_block()
    assert pool.get_block(bid).ref_count == 1
    assert pool.is_used(bid) and pool.get_num_free_blocks() == 3
    freed = pool.deref_block(bid)
    assert freed and not pool.is_used(bid) and pool.get_num_free_blocks() == 4


@pytest.mark.unit
def test_deref_with_refcount_keeps_block():
    pool = BlockPool(4, 4)
    bid = pool.get_new_block()
    pool.get_block(bid).ref_count += 1   # 模拟共享，ref=2
    assert pool.deref_block(bid) is False  # 仍被引用，未释放
    assert pool.is_used(bid)


@pytest.mark.unit
def test_register_hash_and_lookup():
    pool = BlockPool(4, 4)
    bid = pool.get_new_block()
    h = BlockPool.compute_hash([1, 2, 3, 4])
    pool.register_hash(bid, h, [1, 2, 3, 4])
    assert pool.cached_block_id(h) == bid
    assert pool.cached_block_id(BlockPool.compute_hash([9])) == -1


@pytest.mark.unit
def test_reuse_cached_block_increments_or_claims():
    pool = BlockPool(4, 4)
    bid = pool.get_new_block()
    h = BlockPool.compute_hash([1, 2, 3, 4])
    pool.register_hash(bid, h, [1, 2, 3, 4])
    # 已在用：reuse → ref++
    pool.reuse_cached_block(bid)
    assert pool.get_block(bid).ref_count == 2
    # 释放两次回到空闲，再 reuse → 从空闲取出置 ref=1
    pool.deref_block(bid)
    pool.deref_block(bid)
    assert not pool.is_used(bid)
    pool.reuse_cached_block(bid)
    assert pool.get_block(bid).ref_count == 1 and pool.is_used(bid)


@pytest.mark.unit
def test_reallocate_evicts_old_hash():
    pool = BlockPool(1, 4)            # 仅 1 块，强制复用
    bid = pool.get_new_block()
    h = BlockPool.compute_hash([1, 2, 3, 4])
    pool.register_hash(bid, h, [1, 2, 3, 4])
    pool.deref_block(bid)            # 释放，hash 暂留（延迟复用）
    assert pool.cached_block_id(h) == bid
    pool.get_new_block()            # 重新取出同一块 → 应删除旧 hash
    assert pool.cached_block_id(h) == -1


@pytest.mark.unit
def test_fifo_reuse_order():
    pool = BlockPool(2, 4)
    a = pool.get_new_block()
    b = pool.get_new_block()
    pool.deref_block(a)             # a 回到空闲队尾
    pool.deref_block(b)             # b 回到空闲队尾（a 在前）
    assert pool.get_new_block() == a   # FIFO：先释放的先复用
