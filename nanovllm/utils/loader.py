import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    """默认权重加载：直接 copy（不做切分）。"""
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    """
    从 safetensors 文件加载模型权重，支持 packed 权重名称映射。

    HuggingFace 格式 → nano-vllm 格式：
      q_proj, k_proj, v_proj → qkv_proj（QKV 拼接）
      gate_proj, up_proj     → gate_up_proj（gate+up 拼接）

    加载流程：
      1. 遍历 *.safetensors 文件的权重名
      2. 检查是否命中 packed_modules_mapping：
         - 命中：重写参数名，调用 param.weight_loader(param, tensor, shard_id)
         - 未命中：调用 param.weight_loader(param, tensor) 或 default_weight_loader
    """
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    files = sorted(glob(os.path.join(path, "*.safetensors")))
    assert files, f"没有找到 safetensors 文件：{path}"

    for file in files:
        with safe_open(file, framework="pt", device="cpu") as f:
            for weight_name in f.keys():
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        # 目标参数不存在则跳过（如模型未实现的子结构：Qwen3.5 的 MTP 头、
                        # VLM 视觉塔等带 gate_proj/up_proj 的权重）
                        try:
                            param = model.get_parameter(param_name)
                        except AttributeError:
                            break
                        loader = getattr(param, "weight_loader", default_weight_loader)
                        loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    try:
                        param = model.get_parameter(weight_name)
                    except AttributeError:
                        continue
                    loader = getattr(param, "weight_loader", default_weight_loader)
                    loader(param, f.get_tensor(weight_name))
