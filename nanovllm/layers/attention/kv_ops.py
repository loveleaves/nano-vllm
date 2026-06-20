"""KV cache 写入算子（Triton kernel + naive fallback），后端共享。"""
import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:
    @triton.jit
    def store_kvcache_kernel(
        key_ptr,
        key_stride,
        value_ptr,
        value_stride,
        k_cache_ptr,
        v_cache_ptr,
        slot_mapping_ptr,
        D: tl.constexpr,
    ):
        """
        Triton kernel：将新计算的 K/V 写入 KV cache 的指定 slot。

        每个 program 处理一个 token：
          - 从 slot_mapping 读取目标 slot
          - slot == -1 表示无效（CUDA graph dummy token），跳过
          - 向量化读写 D 个元素
        """
        idx = tl.program_id(0)
        slot = tl.load(slot_mapping_ptr + idx)
        if slot == -1:
            return
        key_offsets = idx * key_stride + tl.arange(0, D)
        value_offsets = idx * value_stride + tl.arange(0, D)
        key = tl.load(key_ptr + key_offsets)
        value = tl.load(value_ptr + value_offsets)
        cache_offsets = slot * D + tl.arange(0, D)
        tl.store(k_cache_ptr + cache_offsets, key)
        tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache_triton(key: torch.Tensor, value: torch.Tensor,
                         k_cache: torch.Tensor, v_cache: torch.Tensor,
                         slot_mapping: torch.Tensor):
    """使用 Triton kernel 将 key/value 写入 KV cache。"""
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    store_kvcache_kernel[(N,)](
        key, key.stride(0), value, value.stride(0),
        k_cache, v_cache, slot_mapping, D,
    )


def store_kvcache_naive(key: torch.Tensor, value: torch.Tensor,
                        k_cache: torch.Tensor, v_cache: torch.Tensor,
                        slot_mapping: torch.Tensor):
    """朴素 Python scatter（Triton 不可用时的 fallback）。"""
    block_size = k_cache.shape[1]
    for idx in range(key.shape[0]):
        slot = slot_mapping[idx].item()
        if slot < 0:
            continue
        block_id = slot // block_size
        offset = slot % block_size
        k_cache[block_id, offset] = key[idx]
        v_cache[block_id, offset] = value[idx]


def store_kvcache(key: torch.Tensor, value: torch.Tensor,
                  k_cache: torch.Tensor, v_cache: torch.Tensor,
                  slot_mapping: torch.Tensor):
    """将 K/V 写入 KV cache，优先使用 Triton kernel。"""
    if HAS_TRITON and key.is_cuda:
        store_kvcache_triton(key, value, k_cache, v_cache, slot_mapping)
    else:
        store_kvcache_naive(key, value, k_cache, v_cache, slot_mapping)
